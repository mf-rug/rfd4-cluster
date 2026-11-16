"""Author and validate RFD4-Proteina condition specs.

This is the half of the generator that earns it. The condition-spec YAML is
small but has several ways to be silently wrong, each costing a queue wait to
discover. Every guard below corresponds to a documented trap in
``docs/how-to/input_prep/yaml-json.md``:

* **Selection field count.** ``sel('B/*/N')`` is *not* "chain B, atom N". The
  grammar is ``chain/res_name/res_id/atom_name`` and omitting a field drops
  everything after it, so that string parses as ``res_id=N`` and fails. To
  reach ``atom_name`` every earlier field must be filled: ``sel('B/*/*/N')``.
* **Relative paths resolve against the spec file's directory**, not the
  working directory. Since the job directory is rsynced to a cluster, any
  local absolute path would dangle there -- so the input structure is copied
  into the job directory and referenced by bare basename.
* **A contig must keep at least one preserved segment.** One that is entirely
  expandable is rejected by RFD4 itself.
* **Anything not mentioned is not provided.** Setting only a hotspot gets you
  a hotspot and nothing else -- no fixed coordinates, no known sequence. The
  generator therefore prints what it conditioned, and the caller is meant to
  read it rather than trust intent.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class SpecError(ValueError):
    """A spec that would fail on the cluster, caught locally instead."""


# ---------------------------------------------------------------------------
# contig
# ---------------------------------------------------------------------------

# A preserved segment: chain id, then a residue number or range (A17-131, A52).
# Chain id is non-greedy and must start with a letter: a greedy
# [A-Za-z0-9]+ swallows the residue number, so "A17-131" parses as
# chain="A17", start=-131 -- one residue instead of 115, which
# silently under-sizes the GPU tier.
_PRESERVED = re.compile(r"^(?P<chain>[A-Za-z][A-Za-z0-9]*?)(?P<start>-?\d+)(?:-(?P<end>-?\d+))?$")
# An expandable segment, optionally named: "50-120", "E1:2-20", "E2:10R".
_EXPANDABLE = re.compile(
    r"^(?:(?P<name>[A-Za-z][A-Za-z0-9_]*):)?(?P<lo>\d+)(?:-(?P<hi>\d+))?(?P<flag>[A-Za-z])?$"
)


@dataclass
class ContigToken:
    raw: str
    kind: str  # "preserved" | "expandable" | "break"
    chain: Optional[str] = None
    lo: int = 0
    hi: int = 0  # inclusive worst case

    @property
    def max_residues(self) -> int:
        return 0 if self.kind == "break" else self.hi


def parse_contig(contig: str) -> List[ContigToken]:
    """Parse a contig into tokens, or raise SpecError.

    Worst-case sizing: an expandable range contributes its **upper** bound,
    because a GPU tier is an OOM cliff and must hold the largest design the
    spec can produce, not the average one.
    """
    if not contig or not contig.strip():
        raise SpecError("contig is empty; omit it entirely to keep the whole input")

    tokens: List[ContigToken] = []
    for raw in (t.strip() for t in contig.split(",")):
        if not raw:
            continue
        if raw.startswith("/"):
            tokens.append(ContigToken(raw, "break"))
            continue

        m = _PRESERVED.match(raw)
        if m and not raw[0].isdigit():
            start = int(m.group("start"))
            end = int(m.group("end")) if m.group("end") else start
            if end < start:
                raise SpecError(f"contig token {raw!r}: range runs backwards")
            tokens.append(
                ContigToken(raw, "preserved", chain=m.group("chain"),
                            lo=end - start + 1, hi=end - start + 1)
            )
            continue

        m = _EXPANDABLE.match(raw)
        if m:
            lo = int(m.group("lo"))
            hi = int(m.group("hi")) if m.group("hi") else lo
            if hi < lo:
                raise SpecError(f"contig token {raw!r}: range runs backwards")
            tokens.append(ContigToken(raw, "expandable", lo=lo, hi=hi))
            continue

        raise SpecError(
            f"contig token {raw!r} is not a preserved segment (A17-131), an "
            f"expandable segment (50-120 or E1:2-20), or a chain break (/0)"
        )

    if not any(t.kind == "preserved" for t in tokens):
        raise SpecError(
            "contig has no preserved segment. RFD4 rejects a contig that is "
            "entirely expandable -- `input` exists to be referenced. For a "
            "pure de novo design with no anchor, RFD4's condition spec is the "
            "wrong route entirely."
        )
    return tokens


def contig_max_tokens(contig: str) -> int:
    """Worst-case residue count a contig can produce.

    Named *tokens* rather than residues on purpose: on RFD3 the quantity that
    drove VRAM was token count (residues + ligand atoms + extra chains), and
    keying GPU tiers on stated design length was a trap already hit once. This
    counts residues and is therefore a LOWER BOUND on tokens for any input
    carrying ligands or metals -- pass --max-tokens explicitly for those.
    """
    return sum(t.max_residues for t in parse_contig(contig))


# ---------------------------------------------------------------------------
# selections
# ---------------------------------------------------------------------------

_SEL_CALL = re.compile(r"sel\(\s*['\"](?P<path>[^'\"]*)['\"]\s*\)")


def check_selection(sel: str) -> List[str]:
    """Return a list of warnings for one `select:` string.

    Only the path syntax is checked. The query syntax ("is_ligand and not
    is_protein_backbone") is an arbitrary Python boolean expression over
    annotations and is passed through untouched -- validating it here would
    mean reimplementing AtomWorks.
    """
    warnings: List[str] = []
    for m in _SEL_CALL.finditer(sel):
        fields = m.group("path").split("/")
        if len(fields) > 4:
            warnings.append(
                f"sel('{m.group('path')}') has {len(fields)} fields; the grammar is "
                f"chain/res_name/res_id/atom_name (at most 4)"
            )
            continue
        if len(fields) == 3:
            third = fields[2].strip()
            looks_numeric = (
                third == "*"
                or third.startswith("[")
                or re.fullmatch(r"-?\d+(-\d+)?", third) is not None
            )
            if not looks_numeric:
                warnings.append(
                    f"sel('{m.group('path')}') -- the 3rd field is res_id and must be "
                    f"numeric, so {third!r} will fail. If you meant an atom name, fill "
                    f"every field before it: sel('{'/'.join(fields[:2])}/*/{third}')"
                )
    return warnings


def _residue_selector(chain: str, residues: List[str]) -> str:
    inner = ", ".join(residues)
    return f"sel('{chain}/*/[{inner}]')"


# ---------------------------------------------------------------------------
# spec construction
# ---------------------------------------------------------------------------


@dataclass
class BinderSpec:
    """A protein-protein interface design, the documented default shape.

    Per the upstream agent guidance: target backbone internal coordinates
    (C_CRDGRP), full target sequence (C_SEQ), plus any requested hotspots
    (C_HOT). Internal coordinates rather than absolute ones leave the target's
    global pose free; backbone-only rather than all-atom avoids revealing
    target sidechain identity where it is not wanted.
    """

    name: str
    input_basename: str
    target_chain: str
    target_start: int
    target_end: int
    binder_min: int
    binder_max: int
    hotspots: List[str] = field(default_factory=list)
    hotspot_atoms: List[str] = field(default_factory=list)

    @property
    def contig(self) -> str:
        return (
            f"{self.binder_min}-{self.binder_max}, /0, "
            f"{self.target_chain}{self.target_start}-{self.target_end}"
        )

    def conditions(self) -> Dict[str, Any]:
        c: Dict[str, Any] = {
            # One rigid group, no absolute coordinates: the target keeps its
            # internal geometry but is free to be placed.
            "C_CRDGRP": {
                1: [{"select": f"sel('{self.target_chain}/*/*/[N,CA,C,O]') and not has_nan_coord()"}]
            },
            # The target's sequence is fully known.
            "C_SEQ": {True: [{"select": f"sel('{self.target_chain}')"}]},
        }
        sels = []
        if self.hotspots:
            sels.append({"select": _residue_selector(self.target_chain, self.hotspots)})
        for atomsel in self.hotspot_atoms:
            sels.append({"select": atomsel})
        if sels:
            c["C_HOT"] = {True: sels}
        return c

    def to_dict(self) -> Dict[str, Any]:
        return {
            self.name: {
                "input": self.input_basename,
                "contig": self.contig,
                "conditions": self.conditions(),
            }
        }

    def summary_lines(self) -> List[str]:
        """What was actually conditioned. Read this, not your intent."""
        out = [
            f"design name    : {self.name}",
            f"input          : {self.input_basename} (copied into the job dir)",
            f"contig         : {self.contig}",
            f"  binder       : {self.binder_min}-{self.binder_max} residues, de novo, new chain",
            f"  target       : chain {self.target_chain} residues "
            f"{self.target_start}-{self.target_end} "
            f"({self.target_end - self.target_start + 1} res, preserved)",
            "conditions set :",
            f"  C_CRDGRP     : target backbone N,CA,C,O as one rigid group "
            f"(internal geometry fixed, global pose free)",
            f"  C_SEQ        : target sequence revealed in full",
        ]
        if self.hotspots:
            out.append(f"  C_HOT        : residues {', '.join(self.hotspots)} on chain {self.target_chain}")
        for a in self.hotspot_atoms:
            out.append(f"  C_HOT        : {a}")
        if not self.hotspots and not self.hotspot_atoms:
            out.append("  C_HOT        : (none -- no hotspots requested)")
        out.append(
            "NOT set        : target sidechain coordinates, binder sequence, "
            "secondary structure, H-bond intent. Anything unmentioned is unconditioned."
        )
        return out


def dump_yaml(data: Dict[str, Any], path: Path) -> None:
    """Write a spec dict as YAML.

    `default_flow_style=False` and `sort_keys=False` keep the file diffable and
    in authored order, which is the point of specs living in version control.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise SpecError("PyYAML is required to write a spec") from exc

    class _Dumper(yaml.SafeDumper):
        pass

    # Booleans are condition *keys* (C_SEQ: {true: [...]}) and must stay
    # unquoted `true`, which SafeDumper does by default for bool keys.
    with open(path, "w") as fh:
        fh.write("# Generated by rfd4-setup-job. Edit freely -- it is a plain\n")
        fh.write("# RFD4 condition spec and is read directly by rfd4-generate.\n")
        yaml.dump(data, fh, Dumper=_Dumper, default_flow_style=False, sort_keys=False)


def load_spec(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise SpecError("PyYAML is required to read a spec") from exc
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise SpecError(f"{path}: expected a mapping of design-name -> design")
    return data


def inspect_spec(data: Dict[str, Any]) -> Tuple[int, List[str]]:
    """(worst-case tokens, warnings) for a hand-written spec.

    `global_args` is the one reserved top-level key and is not a design.
    """
    warnings: List[str] = []
    worst = 0
    designs = {k: v for k, v in data.items() if k != "global_args"}
    if not designs:
        raise SpecError("spec defines no designs (only `global_args`?)")

    for name, body in designs.items():
        if not isinstance(body, dict):
            raise SpecError(f"design {name!r}: expected a mapping")
        if "input" not in body:
            warnings.append(f"design {name!r}: no `input` -- RFD4 needs a starting structure")
        contig = body.get("contig")
        if contig:
            try:
                worst = max(worst, contig_max_tokens(str(contig)))
            except SpecError as exc:
                raise SpecError(f"design {name!r}: {exc}") from None
        else:
            warnings.append(
                f"design {name!r}: no `contig`, so the whole input is kept. Size "
                f"cannot be inferred -- pass --max-tokens for correct GPU routing."
            )
        for cond, mapping in (body.get("conditions") or {}).items():
            if not isinstance(mapping, dict):
                continue
            for _value, sels in mapping.items():
                for entry in sels or []:
                    sel = (entry or {}).get("select") if isinstance(entry, dict) else None
                    if isinstance(sel, str):
                        for w in check_selection(sel):
                            warnings.append(f"design {name!r}, {cond}: {w}")
    return worst, warnings


# ---------------------------------------------------------------------------
# local validation
# ---------------------------------------------------------------------------


def find_local_python(explicit: Optional[str] = None) -> Optional[str]:
    """Python interpreter of a local RFD4 install, or None.

    Order: explicit --validate-with, $RFD4_VENV/bin/python, the active venv.
    No path is hardcoded: a machine without a local RFD4 install simply skips
    validation rather than failing.
    """
    import os

    candidates = []
    if explicit:
        candidates.append(explicit)
    if os.environ.get("RFD4_VENV"):
        candidates.append(str(Path(os.environ["RFD4_VENV"]) / "bin/python"))
    if os.environ.get("VIRTUAL_ENV"):
        candidates.append(str(Path(os.environ["VIRTUAL_ENV"]) / "bin/python"))

    for cand in candidates:
        if cand and Path(cand).exists():
            probe = subprocess.run(
                [cand, "-c", "import rfproteina"], capture_output=True, text=True
            )
            if probe.returncode == 0:
                return cand
    return None


def validate_locally(spec_path: Path, python: str, out_dir: Path, timeout: int = 900):
    """Build the annotated CIF from the spec, locally, before submitting.

    This is the whole reason a local generator beats writing job scripts by
    hand: RFD4 resolves a condition spec into an annotated CIF and runs its own
    validation pipeline while doing so, so a malformed selection or an
    impossible contig fails here in seconds instead of after a queue wait.

    Returns (ok, output). Never raises on a *spec* problem -- the caller
    decides whether a failure blocks submission.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    code = (
        "import sys\n"
        "from rfproteina.inference_utils.condition_spec import resolve_condition_spec_inputs\n"
        "from rfproteina.utils.verbosity import configure_verbosity\n"
        "configure_verbosity(False)\n"
        "paths = list(resolve_condition_spec_inputs(sys.argv[1], out_dir=sys.argv[2], validate=True))\n"
        "print('BUILT ' + str(len(paths)) + ' annotated CIF(s)')\n"
        "[print('  ' + str(p)) for p in paths]\n"
    )
    try:
        proc = subprocess.run(
            [python, "-c", code, str(spec_path), str(out_dir)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"validation timed out after {timeout}s"
    except OSError as exc:
        return False, f"could not run {python}: {exc}"

    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, output.strip()


def stage_input(src: Path, job_dir: Path) -> str:
    """Copy the starting structure into the job dir; return its basename.

    A spec's relative paths resolve against the spec file's directory, and the
    job directory is what gets rsynced to the cluster. Copying the structure in
    and referencing it by bare name is the only form that is correct both
    locally (for validation) and remotely (for the run).
    """
    if not src.exists():
        raise SpecError(f"input structure not found: {src}")
    dest = job_dir / src.name
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return src.name


def stage_spec_inputs(data: Dict[str, Any], spec_dir: Path, job_dir: Path) -> List[str]:
    """Copy every structure a hand-written spec references into the job dir.

    A spec's relative `input:` resolves against the spec file's directory, but
    the job directory is what gets rsynced to the cluster -- so a spec that
    works locally dangles remotely unless its structures travel with it. This
    copies them in and rewrites `input:` to bare basenames, which is the one
    form correct in both places.

    Mutates `data`. Returns the basenames staged.
    """
    staged: List[str] = []
    for name, body in data.items():
        if name == "global_args" or not isinstance(body, dict):
            continue
        value = body.get("input")
        if value is None:
            continue
        originals = value if isinstance(value, list) else [value]
        names = []
        for item in originals:
            src = Path(str(item))
            if not src.is_absolute():
                src = (spec_dir / src).resolve()
            if not src.exists():
                raise SpecError(
                    f"design {name!r}: input {item!r} not found (resolved to {src}). "
                    f"Relative paths resolve against the spec file's directory."
                )
            dest = job_dir / src.name
            if src.resolve() != dest.resolve():
                shutil.copy2(src, dest)
            names.append(src.name)
            staged.append(src.name)
        body["input"] = names if isinstance(value, list) else names[0]
    return staged
