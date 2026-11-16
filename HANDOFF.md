# HANDOFF — installing RFD4-Proteina on a new cluster

For an agent working *on* the cluster. The generator runs on a laptop; the
install does not. Report back in the shape of the config block at the end.

Snellius is already installed (2026-11-15). This document exists for the next
cluster, and to record what that install cost.

## Target paths

Everything lives under the project allocation, never in `$HOME`:

| | |
|---|---|
| venv | `/projects/<PROJECT>/rfd4_venv` |
| HF cache | `/projects/<PROJECT>/hf_cache` (`HF_HOME`, 1.7 GB, persistent) |
| native ckpts | `/projects/<PROJECT>/rfd4_data` (`DATA_PATH`) |
| jobs | `/projects/<PROJECT>/rfd4_jobs` |

`HF_HOME` in particular: left at its default the 1.7 GB bundle lands in a
backed-up home directory and is re-fetched by every new user.

## Install

Run it as a **CPU job**, not on a GPU partition — pip needs no GPU and the
install takes about 17 minutes, which is a GPU-hour burned idle otherwise. A
GPU smoke job can queue in parallel with `--dependency=afterok:<install_id>`.

```bash
module purge
module load 2024 Python/3.12.3-GCCcore-13.3.0   # year-stack FIRST, one line
python3 -m venv "$VENV" && source "$VENV/bin/activate"

# pip <26.2.2 still trusts the compromised eMudhra/emSign CA (Oct 2026).
# The module ships pip 24.0, so bootstrap the hash-pinned wheel BEFORE
# anything else is downloaded.
python3 -m pip install "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"

python3 -m pip install "rfd4-proteina[all-cuda13]"
python3 -m pip check
python3 -c "import torch, rfproteina; print('torch', torch.__version__)"
find "$VENV" -xdev | wc -l    # report this number
```

## Known traps

These accumulate across clusters. Add to them.

1. **The CUDA extra tracks the driver, not the card.** `all-cuda13` vs
   `all-cuda12` is decided by `nvidia-smi`'s CUDA version, not by whether the
   node has an A100 or an H100. Snellius runs driver 595.71.05 / CUDA 13.2 on
   both, so `all-cuda13` is correct there — and it is the one that carries the
   BioIR runtime, so cuda12 is the degraded choice, not the safe one.
2. **The pip bootstrap is not optional** on any module stack shipping pip
   older than 26.2.2. Check with `python3 -m pip -V` before and after.
3. **Lmod year-stack first, on one line.** `module load 2024 Python/...`.
   Without the `2024` you silently get a different Python from the 2023 stack.
4. **`DATA_PATH` must be exported even if unused.** `configs/inference.yaml`
   interpolates `${oc.env:DATA_PATH}` for the native checkpoint paths. A
   non-null `hf_ckpt_path` wins at load time, but Hydra still has to *resolve*
   the interpolation first, so an unset `DATA_PATH` fails the run before the
   model is reached.
5. **`$TMPDIR` is deleted at job end.** Anything diagnostic written there is
   gone before you can read it. Write to the job dir.
6. **QOS caps cores per GPU** and `sbatch` *rejects* a job that exceeds it
   (`QOSMaxCpuPerJobLimit`). On Snellius: `gpu_a100` 18, `gpu_h100` 16,
   `gpu_mig` 9. Nodes are charged per GPU, so request the full allowance.
7. **`uv` may not exist** and may have no module. Plain pip works (16:49
   against upstream's "~10 min" claim). Do not `curl | sh` an installer onto
   shared storage.
8. **Inode budgets.** The venv is **82,862 files**. On a cluster that binds on
   file count rather than bytes — Habrók's `/scratch` is 512k soft — that is
   16 % of the budget before weights or outputs. Check before installing.
9. **`hpcjob --jobs-dir` may not exist** on an older build. The generator
   probes `hpcjob submit --help` once and warns rather than silently filing
   jobs under another toolbox's directory.

## Smoke test

```bash
export HF_HOME=/projects/<PROJECT>/hf_cache
export DATA_PATH=/projects/<PROJECT>/rfd4_data
curl -sLO https://raw.githubusercontent.com/RosettaCommons/RFD4-Proteina/main/docs/examples/inputs/5o45_cropped.cif.zst
cat > binder.yaml <<'YAML'
binder:
  input: 5o45_cropped.cif.zst
  contig: "50-120, /0, A17-131"
  conditions: {C_CRDGRP: {1: [{select: "sel('A/*/*/[N,CA,C,O]')"}]},
               C_SEQ: {true: [{select: "sel('A')"}]}}
YAML
rfd4-generate inference_experiment=quickstart inputs=binder.yaml out_dir=./runs/first
```

Poll VRAM into the **job dir** while it runs:

```bash
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done ) > vram.log &
```

## Report back

The config block this fills in is `clusters.<name>` in
`~/.config/rfd4-setup/config.yaml`:

- `python_module` — the exact `module load` line that worked
- `venv`, `hf_home`, `data_path`, `jobs_dir` — the paths as created
- `partitions` — name, `max_hours`, GPUs offered, and the `cpus_per_gpu` QOS
  ceiling for each
- `gpu_tiers` — **only if you measured them.** One smoke run gives one point
  and no curve; say so rather than filling in a table. A proper sweep means
  ascending design sizes, one process per probe (peak VRAM is per-forward-pass),
  early stop on OOM.
- venv file count, and the inode quota if the cluster has one
- whether compute nodes reach the Hugging Face Hub, or whether the cache must
  be pre-fetched on a login node

State plainly which numbers are measured and which are extrapolated. The
generator prints that distinction to users and cannot do so if the config
lies to it.
