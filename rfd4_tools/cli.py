"""rfd4-setup-job -- author an RFD4-Proteina job directory for a Slurm cluster.

The split this tool lives inside: it authors inputs and job scripts and never
learns about ssh; `hpcjob` moves them and never learns about RFD4.

What it actually buys you over writing a job.sh by hand:

1. The condition spec is **validated locally** -- RFD4 resolves it into an
   annotated CIF on this machine before anything is submitted, so a bad
   selection fails in seconds instead of after a queue wait.
2. GPU tier and walltime are derived from the design's worst-case size rather
   than guessed, and the QOS cpus-per-GPU ceiling is applied (exceeding it
   makes sbatch reject the job outright).
3. The traps that cost a run each -- the Lmod year-stack, DATA_PATH having to
   be set even when unused, $TMPDIR vanishing at job end -- are encoded once.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from rfd4_tools import cluster as cluster_mod
from rfd4_tools import generate as gen
from rfd4_tools import spec as spec_mod

PKG_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _parse_range(text: str, what: str) -> tuple:
    """"17-131" or "50-120" -> (17, 131)."""
    parts = str(text).split("-")
    try:
        if len(parts) == 1:
            value = int(parts[0])
            return value, value
        if len(parts) == 2:
            lo, hi = int(parts[0]), int(parts[1])
            if hi < lo:
                raise ValueError
            return lo, hi
    except ValueError:
        pass
    raise SystemExit(f"error: --{what} expects N or LO-HI, got {text!r}")


def _hpcjob_supports_jobs_dir() -> bool:
    """Probe once, rather than assuming.

    An hpcjob predating `--jobs-dir` rejects the flag outright. Dropping it
    silently would be worse: the job lands under whatever jobs_dir the registry
    has, which belongs to a different toolbox.
    """
    if not shutil.which("hpcjob"):
        return False
    try:
        out = subprocess.run(
            ["hpcjob", "submit", "--help"], capture_output=True, text=True, timeout=20
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return "--jobs-dir" in (out.stdout or "") + (out.stderr or "")


def _preflight_report(cluster_name: str) -> Optional[dict]:
    """`hpcjob preflight --json`, or None if unavailable.

    Routing reads this; it is never required. A cluster that cannot be reached
    just means no reroute, not a failure to generate.
    """
    if not shutil.which("hpcjob"):
        return None
    for args in (
        ["hpcjob", "preflight", "-c", cluster_name, "--json"],
        ["hpcjob", "preflight", "--cluster", cluster_name, "--json"],
    ):
        try:
            proc = subprocess.run(args, capture_output=True, text=True, timeout=120)
        except (subprocess.SubprocessError, OSError):
            return None
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                return json.loads(proc.stdout)
            except json.JSONDecodeError:
                return None
    return None


def _copy_package(job_dir: Path) -> None:
    """Ship the helper package into the job dir (PYTHONPATH at job time)."""
    dest = job_dir / "rfd4_tools"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        PKG_DIR, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rfd4-setup-job",
        description="Author an RFD4-Proteina Slurm job directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # PD-L1 binder, spec authored from flags, validated locally, not submitted
  rfd4-setup-job pdl1 --target 5o45.cif.zst --target-chain A \\
      --target-res 17-131 --binder 50-120 --hotspots 56,115,123 \\
      --num-replicates 8

  # A spec you wrote yourself; the generator only sizes and wraps it
  rfd4-setup-job mydesign --spec ./my_spec.yaml --max-tokens 450 --submit

  # What the resource model is made of, and which parts are measured
  rfd4-setup-job --show-model
""",
    )
    p.add_argument("name", nargs="?", help="job name (also the Slurm job name)")

    src = p.add_argument_group("design source (pick one)")
    src.add_argument("--spec", type=Path, help="a condition spec you wrote; passed through")
    src.add_argument("--target", type=Path, help="starting structure for binder mode")

    b = p.add_argument_group("binder mode")
    b.add_argument("--target-chain", default="A", help="chain id of the target (default: A)")
    b.add_argument("--target-res", help="target residue range, e.g. 17-131 (default: whole chain)")
    b.add_argument("--binder", default="50-120", help="binder length range (default: 50-120)")
    b.add_argument("--hotspots", help="comma-separated target residue ids, e.g. 56,115,123")
    b.add_argument(
        "--hotspot-atoms",
        action="append",
        default=[],
        help="raw atom-level hotspot selection; repeatable",
    )

    r = p.add_argument_group("run parameters")
    r.add_argument("--num-replicates", type=int, default=1)
    r.add_argument("--batch-size", type=int, default=1)
    r.add_argument(
        "--task",
        default="ppi",
        choices=sorted(gen.RECYCLE_PRESETS),
        help="selects the recycle schedule (default: ppi)",
    )
    r.add_argument("--budget", default="medium", choices=("low", "medium", "high"))
    r.add_argument("--recycle", help="explicit recycle_strategy, overriding --task/--budget")
    r.add_argument("--metrics", action="store_true", help="keep the metrics group (needs oracle envs)")
    r.add_argument("--seed", type=int)
    r.add_argument(
        "--compile",
        action="store_true",
        help="enable the torch.compile warm-up (minutes; only pays off over many designs)",
    )
    r.add_argument("--inference-experiment", help="e.g. quickstart")
    r.add_argument("--verbose-run", action="store_true", help="verbose=true in the job")
    r.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="K=V",
        help="extra Hydra override, repeatable (use ~key to delete)",
    )

    c = p.add_argument_group("cluster")
    c.add_argument("-c", "--cluster", help="cluster name (default: config's default_cluster)")
    c.add_argument("--list-clusters", action="store_true")
    c.add_argument("--max-tokens", type=int, help="override the worst-case size used for routing")
    c.add_argument("--gpu", help="force a gpu_sbatch spec, skipping tier selection")
    c.add_argument("--cpus", type=int, help="cpus-per-task (capped at the QOS ceiling)")
    c.add_argument("--time", help="walltime override, HH:MM:SS")
    c.add_argument("--auto-route", action="store_true", help="move up a tier if the chosen GPU is busy")
    c.add_argument("--jobs-dir", help="remote base dir, overriding the configured one")

    v = p.add_argument_group("validation")
    v.add_argument(
        "--no-validate",
        action="store_true",
        help="skip building the annotated CIF locally (not recommended)",
    )
    v.add_argument("--validate-with", help="python interpreter of a local RFD4 install")

    o = p.add_argument_group("output")
    o.add_argument("-o", "--job-dir", type=Path, help="where to write (default: ./jobs/NAME)")
    o.add_argument("--submit", action="store_true", help="submit with hpcjob after writing")
    o.add_argument("--overwrite", action="store_true")
    o.add_argument("--show-model", action="store_true", help="print the resource model and exit")
    return p


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_clusters:
        default = cluster_mod.default_cluster_name()
        for name in cluster_mod.list_clusters():
            print(f"{name}{'  (default)' if name == default else ''}")
        return 0

    try:
        clus, timing = cluster_mod.load(args.cluster)
    except cluster_mod.ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.show_model:
        print(f"cluster: {clus.name}")
        print("\nGPU tiers (keyed on worst-case TOKEN count, not stated length):")
        for tier in clus.gpu_tiers:
            if tier.get("measured_at"):
                label = f"curve measured to {tier['measured_at']} tokens"
            elif tier.get("fit_cap"):
                label = f"fit extrapolated to this card (cap {tier['fit_cap']})"
            else:
                label = "EXTRAPOLATED"
            print(f"  <= {tier.get('max_tokens'):>5} tokens -> {tier.get('gpu_sbatch'):<16} "
                  f"mem {tier.get('mem'):<7} [{label}]")
        print("\nWall-time model:")
        print(f"  anchor            : {timing.ref_seconds_per_design:.0f}s per design "
              f"at {timing.ref_tokens} tokens / {timing.ref_steps} steps  [MEASURED]")
        print(f"  token exponent    : {timing.token_exponent}  [MEASURED over 700-1200 tokens]")
        print(f"  model load        : {timing.load_seconds:.0f}s")
        print(f"  compile warm-up   : {timing.compile_warmup_seconds:.0f}s (only with --compile)")
        print(f"  safety factor     : {timing.safety_factor}")
        print("\nFrom a 9-point sweep (300-1200 residues). No OOM cliff was reached --")
        print("the sweep ran out of range, not memory -- so each cap is that curve")
        print("solved at 85% of the card's VRAM, not an observed limit. Below ~500")
        print("tokens the run is overhead-dominated and the time model under-predicts.")
        return 0

    if not args.name:
        print("error: a job name is required", file=sys.stderr)
        return 2
    if bool(args.spec) == bool(args.target):
        print("error: pass exactly one of --spec or --target", file=sys.stderr)
        return 2

    job_dir = args.job_dir or Path.cwd() / "jobs" / args.name
    if job_dir.exists() and not args.overwrite:
        if any(job_dir.iterdir()):
            print(f"error: {job_dir} exists and is not empty (use --overwrite)", file=sys.stderr)
            return 2
    job_dir.mkdir(parents=True, exist_ok=True)

    # ---- the spec -------------------------------------------------------
    spec_path = job_dir / "spec.yaml"
    binder: Optional[spec_mod.BinderSpec] = None
    try:
        if args.spec:
            data = spec_mod.load_spec(args.spec)
            staged = spec_mod.stage_spec_inputs(
                data, args.spec.resolve().parent, job_dir
            )
            # Written out rather than copied: `input:` has been rewritten to
            # bare basenames so the spec is correct both here (for validation)
            # and on the cluster (where local paths do not exist).
            spec_mod.dump_yaml(data, spec_path)
            print(f"spec           : from {args.spec}")
            for basename in dict.fromkeys(staged):
                print(f"  staged input : {basename}")
        else:
            basename = spec_mod.stage_input(args.target, job_dir)
            t_lo, t_hi = (
                _parse_range(args.target_res, "target-res") if args.target_res else (1, 0)
            )
            if t_hi < t_lo:
                print("error: --target-res is required in binder mode", file=sys.stderr)
                return 2
            b_lo, b_hi = _parse_range(args.binder, "binder")
            hotspots = [h.strip() for h in (args.hotspots or "").split(",") if h.strip()]
            binder = spec_mod.BinderSpec(
                name=args.name,
                input_basename=basename,
                target_chain=args.target_chain,
                target_start=t_lo,
                target_end=t_hi,
                binder_min=b_lo,
                binder_max=b_hi,
                hotspots=hotspots,
                hotspot_atoms=list(args.hotspot_atoms),
            )
            spec_mod.dump_yaml(binder.to_dict(), spec_path)
            data = spec_mod.load_spec(spec_path)
            for line in binder.summary_lines():
                print(line)

        worst_tokens, warnings = spec_mod.inspect_spec(data)
    except spec_mod.SpecError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for w in warnings:
        print(f"  warning: {w}")

    max_tokens = args.max_tokens or worst_tokens
    if not max_tokens:
        print("error: could not determine design size; pass --max-tokens", file=sys.stderr)
        return 2
    source = "--max-tokens" if args.max_tokens else "worst-case contig expansion"
    print(f"size           : {max_tokens} tokens ({source})")
    if not args.max_tokens:
        print("  (residue count only -- ligands and metals add tokens; pass")
        print("   --max-tokens explicitly for inputs carrying them)")

    # ---- local validation ----------------------------------------------
    if args.no_validate:
        print("validation     : SKIPPED (--no-validate)")
    else:
        python = spec_mod.find_local_python(args.validate_with)
        if not python:
            print("validation     : skipped -- no local RFD4 install found.")
            print("  Set RFD4_VENV=/path/to/rfd4/.venv to validate specs before they")
            print("  reach the queue; without it a bad selection costs a queue wait.")
        else:
            # Sibling of the job dir, not inside it: everything in the job
            # dir is rsynced to the cluster, and the annotated CIF is a local
            # inspection artefact (open it in PyMOL -- it is still a CIF).
            ok, output = spec_mod.validate_locally(
                spec_path, python, job_dir.parent / f"{job_dir.name}.validation"
            )
            if ok:
                built = [l for l in output.splitlines() if l.startswith("BUILT")]
                print(f"validation     : OK -- {built[0] if built else 'annotated CIF built'}")
            else:
                print("validation     : FAILED -- the spec would fail on the cluster too.")
                print("  " + "\n  ".join(output.splitlines()[-25:]))
                print("\nNothing was submitted. Fix the spec and re-run.")
                return 1

    # ---- resources ------------------------------------------------------
    if args.gpu:
        gpu_sbatch, mem, warn, tier = args.gpu, "64GB", False, {}
        print(f"gpu            : {gpu_sbatch} (forced by --gpu; tier check bypassed)")
    else:
        if gen.exceeds_largest_tier(clus, max_tokens):
            largest = clus.gpu_tiers[-1]
            print(
                f"error: {max_tokens} tokens exceeds the largest configured tier "
                f"({largest.get('max_tokens')} on {largest.get('gpu_sbatch')}).",
                file=sys.stderr,
            )
            return 2
        gpu_sbatch, mem, warn, tier = gen.recommend_gpu(clus, max_tokens)
        label = (
            f"measured at {tier['measured_at']} tokens"
            if tier.get("measured_at")
            else "EXTRAPOLATED from a single measurement"
        )
        print(f"gpu            : {gpu_sbatch}  [{label}]")
        if warn:
            print("  WARNING: this tier's cap is a guess. Watch vram.log in the results.")

    if args.auto_route:
        report = _preflight_report(clus.name)
        if report is None:
            print("auto-route     : preflight unavailable; keeping the size-chosen GPU")
        else:
            new_gpu, new_mem, reason = gen.reroute_for_availability(clus, gpu_sbatch, report)
            if reason:
                gpu_sbatch, mem = new_gpu, new_mem or mem
                print(f"auto-route     : {reason}")
            else:
                print("auto-route     : no move needed")

    n_designs = max(args.num_replicates, 1)
    seconds = gen.estimate_seconds(
        timing, n_designs, max_tokens, steps=timing.ref_steps, compile_enabled=args.compile
    )
    walltime = args.time or gen.format_walltime(seconds)
    try:
        partition = gen.find_partition(clus, gpu_sbatch, seconds)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    cpus = gen.cap_cpus(partition, args.cpus)
    if args.cpus and cpus < args.cpus:
        print(f"  note: --cpus {args.cpus} lowered to {cpus}, the QOS ceiling for "
              f"{partition.get('name')} (sbatch would reject the job otherwise)")

    print(f"partition      : {partition.get('name')}")
    print(f"walltime       : {walltime}" + ("" if args.time else f"  (estimate x{timing.safety_factor})"))
    print(f"cpus / mem     : {cpus} / {mem}")

    # ---- write ----------------------------------------------------------
    recycle = args.recycle or gen.recycle_for(args.task, args.budget)
    params = gen.Rfd4Params(
        inputs="./spec.yaml",
        out_dir="./runs/out",
        num_replicates=args.num_replicates,
        batch_size=args.batch_size,
        recycle_strategy=recycle,
        metrics=args.metrics,
        seed=args.seed,
        compile_lengths=args.compile,
        inference_experiment=args.inference_experiment,
        verbose=args.verbose_run,
        extra_overrides=args.override,
    )
    slurm = gen.SlurmParams(
        job_name=args.name,
        time=walltime,
        partition=str(partition.get("name")),
        gpus_per_node=gpu_sbatch,
        mem=mem,
        cpus_per_task=cpus,
    )
    script = gen.build_job_script(clus, slurm, params)
    script_path = job_dir / "job.sh"
    script_path.write_text(script)
    script_path.chmod(0o755)
    _copy_package(job_dir)

    print(f"recycle        : {recycle} (task={args.task}, budget={args.budget})")
    print(f"written        : {job_dir}")

    # ---- submit ---------------------------------------------------------
    jobs_dir = args.jobs_dir or clus.jobs_dir
    cmd = ["hpcjob", "submit", str(script_path), "-c", clus.name]
    if jobs_dir:
        if _hpcjob_supports_jobs_dir():
            cmd += ["--jobs-dir", jobs_dir]
        else:
            print(
                f"\nWARNING: this hpcjob has no --jobs-dir flag, so the job would land\n"
                f"  under the registry's configured jobs_dir instead of {jobs_dir}.\n"
                f"  Upgrade hpcjob, or move the job after submission."
            )

    if args.submit:
        print("\n$ " + " ".join(cmd))
        return subprocess.call(cmd)

    print("\nSubmit with:\n  " + " ".join(cmd))
    return 0
