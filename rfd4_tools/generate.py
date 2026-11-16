"""GPU/walltime recommendation and Slurm job-script templating.

Nothing here knows a partition or GPU name: every choice comes from the
selected cluster's config block, which is what lets the same generator emit a
correct script for any Slurm cluster.

Calibrated against exactly ONE measured point (Snellius, 2026-11-15): 216
residues, quickstart preset, 60 steps -> 2,657 MiB peak VRAM and 150 s of
generation on an A100-40GB. Every scaling law below is carried over from RFD3
and is explicitly unfitted for RFD4; `--show-model` prints which numbers are
measured and which are assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SlurmParams:
    job_name: str
    time: str
    partition: str
    gpus_per_node: str
    mem: str
    cpus_per_task: int = 8
    nodes: int = 1


@dataclass
class Rfd4Params:
    """Hydra arguments for one `rfd4-generate` invocation."""

    inputs: str = "./spec.yaml"
    out_dir: str = "./runs/out"
    num_replicates: int = 1
    batch_size: int = 1
    # Task-aware recycle preset. The CLI default is `thinking_medium`, which
    # upstream is explicit about being a front-loaded alias and NOT a
    # task-aware selector -- so the generator always sets this deliberately.
    recycle_strategy: str = "medium/front_loaded"
    metrics: bool = False
    seed: Optional[int] = None
    # torch.compile warm-up costs minutes and only pays off across many
    # designs, so it is off unless the job is big enough to amortise it.
    compile_lengths: bool = False
    inference_experiment: Optional[str] = None
    verbose: bool = False
    extra_overrides: Optional[List[str]] = None


# Task -> recycle preset, from the upstream schedule table. `thinking_*` are
# aliases for the front-loaded presets, not task-aware defaults, which is
# exactly why they are not used here.
RECYCLE_PRESETS: Dict[str, Dict[str, str]] = {
    "ppi": {"low": "low/front_loaded", "medium": "medium/front_loaded", "high": "high/front_loaded"},
    "dna": {"low": "low/front_loaded", "medium": "medium/front_loaded", "high": "high/flat"},
    "smallmol": {"low": "low/midpeak", "medium": "medium/midpeak", "high": "high/midpeak"},
    "mcsa": {"low": "low/midpeak", "medium": "medium/midpeak", "high": "high/midpeak"},
    "inverse": {"low": "low/flat", "medium": "medium/flat", "high": "high/flat"},
}


def recycle_for(task: str, budget: str) -> str:
    try:
        return RECYCLE_PRESETS[task][budget]
    except KeyError:
        raise ValueError(
            f"no recycle preset for task={task!r} budget={budget!r}; "
            f"tasks: {', '.join(sorted(RECYCLE_PRESETS))}"
        ) from None


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------


def recommend_gpu(cluster, max_tokens: int) -> Tuple[str, str, bool, Dict[str, Any]]:
    """(gpu_sbatch, mem, warn, tier) for the largest design in the job.

    Keyed on token count, not stated design length. On RFD3 the quantity that
    actually drove VRAM was tokens -- residues plus ligand atoms plus extra
    chains -- and a tier table keyed on length was a trap already hit once.
    """
    if not cluster.gpu_tiers:
        raise RuntimeError(
            f"Cluster {cluster.name!r} has no gpu_tiers configured -- add them "
            f"to ~/.config/rfd4-setup/config.yaml."
        )
    for tier in cluster.gpu_tiers:
        if max_tokens <= int(tier.get("max_tokens", 0)):
            return (
                str(tier.get("gpu_sbatch", "")),
                str(tier.get("mem", "32GB")),
                bool(tier.get("warn", False)),
                tier,
            )
    last = cluster.gpu_tiers[-1]
    return (
        str(last.get("gpu_sbatch", "")),
        str(last.get("mem", "32GB")),
        True,
        last,
    )


def exceeds_largest_tier(cluster, max_tokens: int) -> bool:
    if not cluster.gpu_tiers:
        return False
    return max_tokens > int(cluster.gpu_tiers[-1].get("max_tokens", 0))


def reroute_for_availability(cluster, gpu_sbatch: str, report: Dict[str, Any]):
    """Move UP a GPU tier when the size-chosen one is fully busy.

    `gpu_tiers` is ordered by capability, which is the ordering Slurm does not
    expose -- it reports GPU names, not their VRAM. Only ever moves up: a
    tier's cap is an OOM cliff, so a smaller card is never a valid substitute
    however idle it looks.

    Returns (gpu_sbatch, mem, reason); unchanged input and None if no move.
    """
    totals = (report or {}).get("gpu_totals") or {}
    if not totals:
        return gpu_sbatch, None, None

    def free_of(spec: str) -> Optional[int]:
        entry = totals.get(str(spec).split(":")[0].lower())
        return None if entry is None else int(entry.get("free", 0))

    current_free = free_of(gpu_sbatch)
    if current_free is None or current_free > 0:
        return gpu_sbatch, None, None

    names = [str(t.get("gpu_sbatch", "")) for t in cluster.gpu_tiers]
    try:
        idx = names.index(gpu_sbatch)
    except ValueError:
        return gpu_sbatch, None, None

    for tier in cluster.gpu_tiers[idx + 1:]:
        spec = str(tier.get("gpu_sbatch", ""))
        free = free_of(spec)
        if free and free > 0:
            return (
                spec,
                str(tier.get("mem", "32GB")),
                f"{gpu_sbatch} has 0 free; moved up to {spec} ({free} free)",
            )
    return gpu_sbatch, None, None


# ---------------------------------------------------------------------------
# Wall time
# ---------------------------------------------------------------------------


def estimate_seconds(
    timing,
    n_designs: int,
    max_tokens: int,
    steps: int = 60,
    compile_enabled: bool = False,
) -> int:
    """Wall-clock estimate, generous on purpose.

    One anchor (`ref_seconds_per_design` at `ref_tokens`/`ref_steps`) scaled by
    an ASSUMED token exponent. With a single measured point there is no way to
    fit the exponent, so `safety_factor` carries the uncertainty. Over-
    requesting walltime costs scheduling priority; under-requesting costs the
    whole run.
    """
    tokens = max(int(max_tokens), 1)
    scale = (tokens / float(timing.ref_tokens)) ** timing.token_exponent
    step_scale = max(steps, 1) / float(timing.ref_steps)
    gen = n_designs * timing.ref_seconds_per_design * scale * step_scale
    overhead = timing.load_seconds + (timing.compile_warmup_seconds if compile_enabled else 0.0)
    total = (overhead + gen) * timing.safety_factor
    return int(max(total, timing.min_minutes * 60))


def format_walltime(seconds: int) -> str:
    seconds = int(math.ceil(seconds / 60.0) * 60)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def find_partition(cluster, gpu_sbatch: str, seconds: int) -> Dict[str, Any]:
    """Smallest partition that offers this GPU and allows this walltime."""
    hours = seconds / 3600.0
    candidates = [
        p for p in cluster.partitions
        if gpu_sbatch in (p.get("gpus") or []) and hours <= float(p.get("max_hours", 0))
    ]
    if not candidates:
        offering = [p for p in cluster.partitions if gpu_sbatch in (p.get("gpus") or [])]
        if not offering:
            raise RuntimeError(
                f"No partition on {cluster.name!r} offers {gpu_sbatch}. "
                f"Check `partitions:` in ~/.config/rfd4-setup/config.yaml."
            )
        longest = max(offering, key=lambda p: float(p.get("max_hours", 0)))
        raise RuntimeError(
            f"Estimated walltime {format_walltime(seconds)} exceeds the "
            f"{longest.get('max_hours')}h limit of every partition offering "
            f"{gpu_sbatch}. Split the job (fewer replicates) or pick a larger GPU."
        )
    return min(candidates, key=lambda p: float(p.get("max_hours", 0)))


def cap_cpus(partition: Optional[Dict[str, Any]], requested: Optional[int]) -> int:
    """Clamp --cpus-per-task to the partition's QOS ceiling.

    This is not an optimisation: exceeding `cpus_per_gpu` makes sbatch REJECT
    the job with QOSMaxCpuPerJobLimit. Nodes are charged per GPU, so the full
    allowance is taken by default -- asking for less saves nothing.
    """
    ceiling = partition.get("cpus_per_gpu") if partition else None
    if ceiling is None:
        return requested or 8
    ceiling = int(ceiling)
    return ceiling if requested is None else min(int(requested), ceiling)


# ---------------------------------------------------------------------------
# Job script
# ---------------------------------------------------------------------------


def _hydra_override(item: str) -> str:
    """Pass a user override through, shell-quoted.

    RFD4 uses Hydra `key=value` arguments, never `--key value`. Deletions use
    the `~key` form, which must survive the shell intact.
    """
    import shlex

    return shlex.quote(item)


def _rfd4_command(p: Rfd4Params) -> str:
    parts = ["rfd4-generate"]
    if p.inference_experiment:
        parts.append(f"inference_experiment={p.inference_experiment}")
    parts += [
        f"inputs={p.inputs}",
        f"out_dir={p.out_dir}",
        f"num_replicates={p.num_replicates}",
        f"batch_size={p.batch_size}",
        f"recycle_strategy={p.recycle_strategy}",
    ]
    if not p.compile_lengths:
        parts.append("compile_lengths=null")
    if p.seed is not None:
        parts.append(f"seed={p.seed}")
    if p.verbose:
        parts.append("verbose=true")
    if not p.metrics:
        # Drops the whole metrics group: no refolding oracle, no metric
        # dependencies. The AF2/AF3 environments are a separate install.
        parts.append("'~metrics'")
    for item in p.extra_overrides or []:
        parts.append(_hydra_override(item))
    return " \\\n    ".join(parts)


def build_job_script(
    cluster,
    slurm: SlurmParams,
    p: Rfd4Params,
    poll_vram: bool = True,
) -> str:
    """Emit the job script.

    Deliberately NOT built on upstream's `scripts/launch_generate.sh`. That
    script submits from *inside* the cluster (it calls sbatch itself, reads
    `scripts/.user_info_<cluster>`, and wants a repo copy per run), whereas
    this toolbox's model is: author locally, rsync a self-contained job dir,
    sbatch it. Upstream's launcher is the better tool for job arrays and
    dependent metric aggregation -- if you need those, use it on the cluster
    directly and skip this generator.
    """
    module_line = (
        f"module purge\nmodule load {cluster.python_module}\n"
        if cluster.python_module
        else ""
    )
    # Diagnostics live in an EXIT trap, not in a line after the run.
    # `cmd || rc=$?` would survive `set -e` on its own, but a trap also covers
    # an abort from any OTHER line -- a failed `source`, a missing cache dir --
    # which is exactly when the summary is worth having.
    vram_start = ""
    vram_kill = ""
    vram_peak = ""
    if poll_vram:
        # Written to the job dir, NOT $TMPDIR: $TMPDIR is deleted when the job
        # ends, taking any diagnostic with it.
        vram_start = (
            "nvidia-smi --query-gpu=name,memory.total,driver_version "
            "--format=csv,noheader || true\n"
            "( while true; do nvidia-smi --query-gpu=memory.used "
            "--format=csv,noheader,nounits; sleep 5; done ) > vram.log 2>/dev/null &\n"
            "_poll=$!\n"
        )
        vram_kill = '    kill $_poll 2>/dev/null || true\n'
        # `|| echo unknown` because an empty or missing vram.log under
        # `pipefail` otherwise makes the substitution itself the failure.
        vram_peak = (
            '    echo "peak VRAM (MiB): '
            '$(sort -n vram.log 2>/dev/null | tail -1 || echo unknown)"\n'
        )

    return f"""#!/bin/bash
#SBATCH --job-name={slurm.job_name}
#SBATCH --time={slurm.time}
#SBATCH --partition={slurm.partition}
#SBATCH --gpus-per-node={slurm.gpus_per_node}
#SBATCH --nodes={slurm.nodes}
#SBATCH --cpus-per-task={slurm.cpus_per_task}
#SBATCH --mem={slurm.mem}
#SBATCH --output=slurm-%j.out
set -eo pipefail

# Run from the submitted job directory; everything below is relative to it.
_job_dir="${{SLURM_SUBMIT_DIR:-$PWD}}"
cd "$_job_dir"

{module_line}source {cluster.venv_path}/bin/activate

# RFD4 caches. HF_HOME holds the model bundle; DATA_PATH must be set even
# though hf_ckpt_path wins, because configs/inference.yaml interpolates
# ${{oc.env:DATA_PATH}} for the native checkpoint paths and Hydra resolves it
# before deciding it is unused.
export HF_HOME="{cluster.hf_home}"
export DATA_PATH="{cluster.data_path}"
mkdir -p "$HF_HOME" "$DATA_PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Emitted on EVERY bash-controlled exit path, including a `set -e` abort --
# which is the case where losing the diagnostics would hurt most.
_cleanup() {{
    _rc=$?
{vram_kill}    echo "=== rfd4-generate exit=$_rc  elapsed=${{SECONDS}}s ==="
{vram_peak}    # Never fails the job: a parser bug must not turn a good run into a
    # confusing failure.
    PYTHONPATH="$_job_dir" python3 -m rfd4_tools log "$_job_dir" || true
}}
trap _cleanup EXIT

{vram_start}
SECONDS=0
# Absolute path rather than the activated PATH: immune to activate not having
# run and to any PATH ordering surprise on a misconfigured node.
{cluster.venv_path}/bin/{_rfd4_command(p)}
"""
