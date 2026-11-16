"""Cluster configuration for rfd4-setup-job.

Reads ``~/.config/rfd4-setup/config.yaml``, a multi-cluster registry. Nothing
cluster-specific is hardcoded in the Python: adding a cluster is a config edit,
selected with ``--cluster NAME``.

This is deliberately a *separate* registry from ``hpcjob``'s
(``~/.config/hpcjob/clusters.yaml``). hpcjob owns transport facts -- host,
search paths, rsync flags. This file owns model facts -- venv path, module
line, GPU tiers, the RFD4 cache paths. The two never overlap, which is what
lets hpcjob stay model-agnostic.

``{user}`` in any path is replaced with the *cluster* username, resolved from
``ssh_target`` via ``ssh -G`` (which reads ~/.ssh/config without connecting).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

CONFIG_PATH = Path(
    os.environ.get("RFD4_SETUP_CONFIG", Path.home() / ".config/rfd4-setup/config.yaml")
)

# Written on first run. Snellius values are from the install+smoke report
# (2026-11-15); everything labelled EXTRAPOLATED is a guess from one datum and
# should be replaced by a size sweep before anyone trusts it.
DEFAULT_CONFIG = """\
# rfd4-setup cluster configuration (multi-cluster).
# Select with `rfd4-setup-job --cluster NAME`; without it, `default_cluster`.
# Machine-local: holds your paths and allocation ids.
#
# {user} is replaced with the cluster username (resolved via `ssh -G <ssh_target>`).
#
# NOTE: this is NOT hpcjob's registry. hpcjob owns host/search_paths/rsync;
# this owns venv/module/tiers. Keep them disjoint.

default_cluster: snellius

clusters:

  # ---- SURF / Snellius ---------------------------------------------------
  # Replace <PROJECT> with your allocation id (e.g. prjs1234). These paths are
  # per-allocation and must be edited before first use.
  snellius:
    ssh_target: ss
    # Lmod year-stack MUST precede the module, on one `module load` line.
    # Without the `2024` you silently get the 2023 stack and a different Python.
    python_module: "2024 Python/3.12.3-GCCcore-13.3.0"
    venv:
      path: /projects/<PROJECT>/rfd4_venv
    # RFD4 caches. HF_HOME holds the 1.7 GB model bundle and is persistent --
    # left at its default it lands in a backed-up home and is re-fetched per user.
    hf_home: /projects/<PROJECT>/hf_cache
    data_path: /projects/<PROJECT>/rfd4_data
    # hpcjob's own jobs_dir points at another toolbox; RFD4 jobs go here and
    # the generator passes `hpcjob submit --jobs-dir` to say so.
    jobs_dir: /projects/<PROJECT>/rfd4_jobs
    epilog_marker: ""
    # MEASURED, from a 9-point sweep on Snellius (2026-11-16), 300-1200
    # residues at nsteps=20:
    #     peak_MiB = 2077 + 0.01039 * L^2      (max deviation 2.9%)
    # Peak memory is independent of step count -- verified, 3,079 MiB at both
    # 20 and 60 steps from two separate jobs -- so a cheap 20-step probe sizes
    # a 60-step production run correctly.
    #
    # Caps below solve that curve at 85% of each card's VRAM, the same margin
    # RFD3's measured tier sits at. No OOM cliff was actually reached: the
    # sweep ran out of range at 1200 residues, not out of memory, so the caps
    # are a fit extrapolated to the card rather than an observed limit.
    #
    # max_tokens, not max_length: VRAM follows *token* count (residues +
    # ligand atoms + extra chains), and keying tiers on stated design length
    # was a trap rfd3-cluster already hit once.
    gpu_tiers:
      - {max_tokens: 1150, gpu_sbatch: a100_3g.20gb:1, mem: 48GB, fit_cap: 1215}
      - {max_tokens: 1700, gpu_sbatch: a100:1,         mem: 64GB, fit_cap: 1775, measured_at: 1200}
      - {max_tokens: 2400, gpu_sbatch: h100:1,         mem: 96GB, fit_cap: 2549, warn: true}
    # cpus_per_gpu is a QOS CEILING: exceed it and sbatch rejects the job with
    # QOSMaxCpuPerJobLimit. Nodes are charged per GPU, so take the full
    # allowance -- asking for less saves nothing.
    partitions:
      - {name: gpu_mig,  max_hours: 120, gpus: [a100_3g.20gb:1], cpus_per_gpu: 9}
      - {name: gpu_a100, max_hours: 120, gpus: [a100:1],         cpus_per_gpu: 18}
      - {name: gpu_h100, max_hours: 120, gpus: [h100:1],         cpus_per_gpu: 16}

  # ---- RUG / Habrok ------------------------------------------------------
  # DELIBERATELY ABSENT. The RFD4 venv measures 82,862 files -- 16% of the
  # 512k soft inode cap on /scratch -- before weights or outputs. Habrok binds
  # on file count, not bytes. Add a block here only if that budget changes.

# Wall-time model, fitted to the same Snellius sweep (nsteps=20, batch 1):
#   generation_seconds = ref_seconds_per_design * (tokens/ref_tokens)^1.73
# The exponent is MEASURED over 700-1200 residues. Below ~500 residues the
# run is dominated by fixed overhead rather than by size, so the model
# under-predicts there; safety_factor and min_minutes absorb it.
#
# Step scaling is very nearly linear but not exactly: 20 -> 60 steps measured
# 2.6x, not 3x, because a fixed cost sits inside the generation phase.
timing:
  load_seconds: 120          # python start + HF cache check + model load
  ref_seconds_per_design: 39 # MEASURED: 700 tokens, 20 steps, A100
  token_exponent: 1.73       # MEASURED over 700-1200 tokens (was 2.0, assumed)
  ref_tokens: 700
  ref_steps: 20
  # torch.compile warm-up, paid once per job when compile_lengths is enabled.
  compile_warmup_seconds: 420
  safety_factor: 2.0         # lowered from 2.5 now that the curve is measured
  min_minutes: 15
"""


class ConfigError(RuntimeError):
    pass


@dataclass
class Cluster:
    """One cluster's settings, with {user} already substituted."""

    name: str
    ssh_target: str = ""
    python_module: str = ""
    venv_path: str = ""
    hf_home: str = ""
    data_path: str = ""
    jobs_dir: str = ""
    epilog_marker: str = ""
    gpu_tiers: List[Dict[str, Any]] = field(default_factory=list)
    partitions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Timing:
    """Wall-time cost model (see generate.estimate_seconds).

    All of it is measured now, from a 9-point Snellius sweep: the exponent
    over 700-1200 residues, the anchor at 700. Below ~500 residues fixed
    overhead dominates and the model under-predicts, which min_minutes covers.
    """

    load_seconds: float = 120.0
    ref_seconds_per_design: float = 39.0
    token_exponent: float = 1.73
    ref_tokens: int = 700
    ref_steps: int = 20
    compile_warmup_seconds: float = 420.0
    safety_factor: float = 2.0
    min_minutes: int = 15


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ConfigError(
            "PyYAML is required. Install it with: python3 -m pip install pyyaml"
        ) from exc
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return data


def ensure_config(path: Path = CONFIG_PATH) -> Path:
    """Create the config file with annotated defaults if it doesn't exist."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_CONFIG)
    return path


def resolve_cluster_user(ssh_target: str) -> str:
    """Cluster username for {user} substitution, via `ssh -G` (offline)."""
    if "@" in ssh_target:
        return ssh_target.split("@", 1)[0]
    if ssh_target and shutil.which("ssh"):
        try:
            out = subprocess.run(
                ["ssh", "-G", ssh_target], capture_output=True, text=True, timeout=10
            ).stdout
            for line in out.splitlines():
                if line.lower().startswith("user "):
                    return line.split(None, 1)[1].strip()
        except (subprocess.SubprocessError, OSError):
            pass
    return os.environ.get("LOGNAME") or os.environ.get("USER") or ""


def load(cluster_name: Optional[str] = None, path: Path = CONFIG_PATH):
    """Return (Cluster, Timing) for `cluster_name` (or the default cluster)."""
    ensure_config(path)
    data = _load_yaml(path)

    clusters = data.get("clusters")
    if not isinstance(clusters, dict) or not clusters:
        raise ConfigError(f"{path}: no `clusters:` mapping")

    name = cluster_name or data.get("default_cluster")
    if not name:
        raise ConfigError(f"{path}: no `default_cluster` set and no --cluster given")
    if name not in clusters:
        raise ConfigError(
            f"Unknown cluster {name!r}. Configured: {', '.join(sorted(clusters))}"
        )

    raw = clusters[name] or {}
    ssh_target = str(raw.get("ssh_target", "") or "")

    venv = raw.get("venv") or {}
    venv_path = str(venv.get("path", "") if isinstance(venv, dict) else venv or "")

    paths = {
        "venv_path": venv_path,
        "hf_home": str(raw.get("hf_home", "") or ""),
        "data_path": str(raw.get("data_path", "") or ""),
        "jobs_dir": str(raw.get("jobs_dir", "") or ""),
    }
    if any("{user}" in v for v in paths.values()):
        user = resolve_cluster_user(ssh_target)
        if not user:
            raise ConfigError(
                f"Cluster {name!r} uses {{user}} in its paths but the username "
                f"could not be resolved from ssh_target={ssh_target!r}."
            )
        paths = {k: v.replace("{user}", user) for k, v in paths.items()}

    unset = [k for k, v in paths.items() if "<PROJECT>" in v]
    if unset:
        raise ConfigError(
            f"Cluster {name!r} still has the <PROJECT> placeholder in: "
            f"{', '.join(unset)}.\nEdit {path} and set your allocation id."
        )

    cluster = Cluster(
        name=name,
        ssh_target=ssh_target,
        python_module=str(raw.get("python_module", "") or ""),
        epilog_marker=str(raw.get("epilog_marker", "") or ""),
        gpu_tiers=list(raw.get("gpu_tiers") or []),
        partitions=list(raw.get("partitions") or []),
        **paths,
    )

    t = data.get("timing") or {}
    timing = Timing(
        load_seconds=float(t.get("load_seconds", 120)),
        ref_seconds_per_design=float(t.get("ref_seconds_per_design", 39)),
        token_exponent=float(t.get("token_exponent", 1.73)),
        ref_tokens=int(t.get("ref_tokens", 700)),
        ref_steps=int(t.get("ref_steps", 20)),
        compile_warmup_seconds=float(t.get("compile_warmup_seconds", 420)),
        safety_factor=float(t.get("safety_factor", 2.0)),
        min_minutes=int(t.get("min_minutes", 15)),
    )
    return cluster, timing


def list_clusters(path: Path = CONFIG_PATH) -> List[str]:
    ensure_config(path)
    data = _load_yaml(path)
    clusters = data.get("clusters")
    return sorted(clusters) if isinstance(clusters, dict) else []


def default_cluster_name(path: Path = CONFIG_PATH) -> Optional[str]:
    ensure_config(path)
    return _load_yaml(path).get("default_cluster")
