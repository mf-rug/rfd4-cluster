"""Cluster-side result summary. Runs inside the job, from an exit path.

Three rules, inherited from the rfd3-cluster helper package:

* **stdlib only** -- it runs under RFD4's venv Python but must not import from
  it, so the generator and the model can never disagree about versions.
* **Never raise into the job.** This is called after the run; a crash here
  would turn a successful job into a confusing failure.
* **Idempotent** -- resubmits are common.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import List, Optional

# RFD4 writes PLAIN .cif by default -- the exact inverse of RFD3's .cif.gz.
# Both are globbed because `compress_outputs=true` switches it.
STRUCTURE_GLOBS = ("*_generated.cif", "*_generated.cif.gz", "*_generated.bcif", "*_generated.bcif.zst")


def _find_run_dir(job_dir: Path) -> Optional[Path]:
    """The directory holding inference_samples/, wherever out_dir pointed."""
    for cand in sorted(job_dir.rglob("inference_samples")):
        if cand.is_dir():
            return cand.parent
    return None


def _structures(run_dir: Path) -> List[Path]:
    samples = run_dir / "inference_samples"
    out: List[Path] = []
    for pattern in STRUCTURE_GLOBS:
        out.extend(sorted(samples.rglob(pattern)))
    return out


def _metrics_rows(run_dir: Path) -> List[dict]:
    path = run_dir / "inference_metrics" / "per_example_metrics.csv"
    if not path.exists():
        return []
    try:
        with open(path, newline="") as fh:
            return list(csv.DictReader(fh))
    except OSError:
        return []


def summarise(job_dir: Path) -> str:
    lines: List[str] = ["--- rfd4 result summary ---"]

    run_dir = _find_run_dir(job_dir)
    if run_dir is None:
        lines.append("NO inference_samples/ directory found anywhere under the job dir.")
        lines.append("The run produced no structures -- check slurm-*.out above.")
        return "\n".join(lines)

    try:
        rel = run_dir.relative_to(job_dir)
    except ValueError:
        rel = run_dir
    lines.append(f"run dir   : {rel}")

    structures = _structures(run_dir)
    lines.append(f"designs   : {len(structures)} structure file(s)")
    if not structures:
        lines.append("  WARNING: inference_samples/ exists but holds no *_generated.* files.")
    for path in structures[:20]:
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        lines.append(f"  {path.relative_to(run_dir)}  ({size} bytes)")
    if len(structures) > 20:
        lines.append(f"  ... and {len(structures) - 20} more")

    rows = _metrics_rows(run_dir)
    if rows:
        lines.append(f"metrics   : per_example_metrics.csv, {len(rows)} row(s)")
        # Provenance: RFD4 records these per design, which is why this toolbox
        # writes no checkpoint manifest of its own.
        first = rows[0]
        for key in ("ckpt_path", "ckpt_name", "seed"):
            if key in first:
                values = {str(r.get(key, "")) for r in rows}
                shown = ", ".join(sorted(values)[:3])
                lines.append(f"  {key:10s}: {shown}")
    else:
        lines.append("metrics   : no per_example_metrics.csv (expected when '~metrics' is set)")

    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    job_dir = Path(argv[0]) if argv else Path.cwd()
    try:
        print(summarise(job_dir))
    except Exception as exc:  # never raise into the job
        print(f"--- rfd4 result summary unavailable: {exc!r} ---")
    return 0
