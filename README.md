# rfd4-cluster

`rfd4-setup-job` — author an [RFD4-Proteina](https://github.com/RosettaCommons/RFD4-Proteina)
job directory for a Slurm cluster, from flags or from a condition spec you wrote.

Companion to [`hpcjob`](https://github.com/mf-rug/hpcjob) (transport) and a
sibling of [`rfd3-cluster`](https://github.com/mf-rug/rfd3-cluster). The split
is the same one those two already use: **this tool authors inputs and job
scripts and never learns about ssh; `hpcjob` moves them and never learns about
RFD4.**

## Why this exists, honestly

`hpcjob/docs/NEW-CLUSTER-TOOL.md` §2 says build a generator only when two or
more of four conditions hold. For RFD4 exactly two do:

| Condition | RFD4 |
|---|---|
| Inputs need authoring | **Yes.** The condition spec has at least four documented ways to be silently wrong, each costing a queue wait. |
| Resources depend on the input | **Yes.** Token count drives VRAM and walltime, and `compile_lengths` changes the warm-up economics. |
| Outputs need post-processing | **No.** Upstream's own launcher already does metric concatenation and Foldseek clustering. |
| You will run it tens of times | **Unknown.** Nobody has run RFD4 in anger here yet. |

Two of four is the threshold, not a comfortable margin. If RFD4 turns out to
be a once-a-month tool, this repo is more machinery than the job deserves and
should be collapsed back into a hand-written `job.sh`.

### Why not upstream's `scripts/launch_generate.sh`

RFD4 ships a perfectly good SLURM launcher with job arrays, dependent metric
aggregation and Foldseek clustering. It is the better tool for batch fan-out —
**use it directly on the cluster for that.** It does not fit here because it
submits from *inside* the cluster: it calls `sbatch` itself, reads
`scripts/.user_info_<cluster>`, and wants a repo copy per run. This toolbox's
model is author locally, rsync a self-contained job dir, submit. The two are
complementary, not competing.

## What it actually buys you

1. **Specs are validated locally, before submission.** RFD4 resolves a
   condition spec into an annotated CIF, running its own validation pipeline
   while doing so. `rfd4-setup-job` does that on your machine, so a malformed
   selection fails in seconds with RFD4's own error message instead of after a
   queue wait. This is the single biggest reason the tool is worth having, and
   it needs a local RFD4 install (`RFD4_VENV`) — without one, validation is
   skipped with a warning and you are back to guessing.
2. **GPU tier and walltime are derived, not guessed**, and the QOS
   cpus-per-GPU ceiling is applied. Exceeding that ceiling makes `sbatch`
   *reject* the job, which is a confusing failure to hit by hand.
3. **The traps are encoded once**: the Lmod year-stack, `DATA_PATH` needing to
   be set even though `hf_ckpt_path` wins, `$TMPDIR` vanishing at job end,
   `--jobs-dir` not existing on older `hpcjob` builds.

## Install

> **Known upstream breakage (2026-11-16).** RFD4 0.0.3's published wheel
> allows `atomworks<4`, but atomworks 3.1 moved to Biotite 2's incompatible
> structure API, so RFD4 fails at its entry point with `ImportError: cannot
> import name 'to_cif_string'`. Upstream already fixed this on main (`<3.1`);
> the fix is unreleased.
>
> Whether it bites you depends on platform and extras, not install date: on
> Linux the CUDA extras pull `tmol`/`bionemo-ir`, which hard-pin
> `biotite==1.6.0` and force atomworks back to 3.0.4. Those deps are
> linux-gated, so **a Mac install from PyPI has nothing holding it back and
> will break**, with or without extras. Installing from a clone of RFD4's main
> branch is safe everywhere. When in doubt add
> `"atomworks[ml,openbabel]>=3.0.4,<3.1"` — a range, not `==`, so it matches
> upstream and becomes a no-op on release. This concerns the *model's*
> environment, not this tool's.

```bash
git clone https://github.com/mf-rug/rfd4-cluster.git
ln -s "$PWD/rfd4-cluster/rfd4-setup-job" ~/.local/bin/rfd4-setup-job

# Point it at a local RFD4 install so specs can be validated before submission.
export RFD4_VENV="/path/to/your/rfd4/.venv"
```

PyYAML is the only third-party requirement. If your `python3` lacks it, the
wrapper falls back to `$RFD4_VENV`'s interpreter automatically.

First run writes `~/.config/rfd4-setup/config.yaml`. **Edit it before using
it** — the Snellius block ships with a `<PROJECT>` placeholder and the tool
refuses to run until you replace it with your allocation id.

## Use

```bash
# A PD-L1 binder, spec authored from flags, validated locally
rfd4-setup-job pdl1 \
    --target 5o45_cropped.cif.zst --target-chain A --target-res 17-131 \
    --binder 50-120 --hotspots 56,115,123 \
    --num-replicates 8

# A spec you wrote yourself; the generator sizes it, wraps it, stages its inputs
rfd4-setup-job mydesign --spec ./my_spec.yaml --submit

# What the resource model is made of, and which parts are measured
rfd4-setup-job --show-model
```

Binder mode emits the shape upstream recommends for PPI: target backbone
internal coordinates (`C_CRDGRP`), full target sequence (`C_SEQ`), plus any
hotspots (`C_HOT`). It prints what it conditioned **and what it did not** —
read that summary rather than trusting your intent, because anything a spec
does not mention is simply unconditioned.

For anything binder mode cannot express — motif scaffolding, ligands, DNA,
partial diffusion — write the spec yourself and pass `--spec`. The generator
still sizes it, validates it, stages its structures and wraps it.

## What is measured and what is not

A 9-point sweep on Snellius (300-1200 residues, 2026-11-16) gives the
memory curve the tier table is built on:

```
peak_MiB = 2077 + 0.01039 * L^2        (max deviation 2.9%)
```

**No OOM cliff was reached.** The sweep ran out of range at 1200 residues,
not out of memory, so each tier cap is that curve solved at 85% of the
card's VRAM rather than an observed limit. `--show-model` says which tier
the curve actually reaches and which are extrapolated to the card.

Peak memory is **independent of step count** -- 3,079 MiB at both 20 and 60
steps, from two separate jobs, confirmed on CUDA and MPS. So a cheap 20-step
probe sizes a 60-step production run correctly. (`nsteps` has a floor of 20:
`tail_steps` x `tail_step_multiplier`.)

Generation time scales as **L^1.73** over 700-1200 residues, not the L^2 that
RFD3 behaviour suggested. Below ~500 residues runs are dominated by fixed
overhead rather than size, and the model under-predicts there; `min_minutes`
covers it.

Tiers are keyed on **token count, not stated design length**. On RFD3 the
quantity that drove VRAM was tokens -- residues plus ligand atoms plus extra
chains -- and keying tiers on length was a trap already hit once. The
generator counts residues from the contig, which is a *lower bound* on tokens:
for inputs carrying ligands or metals, pass `--max-tokens` explicitly.

> Upgrading from an earlier version? `~/.config/rfd4-setup/config.yaml` is
> only written when absent, so an existing config still holds the old guessed
> tiers (caps of 400/900/1600 and `token_exponent: 2.0`). Delete it to get the
> measured ones, or edit it against `cluster.py`'s defaults.

## Habrók is deliberately absent

The RFD4 venv measures **82,862 files — 16 % of the 512k soft inode cap** on
Habrók's `/scratch`, before weights or outputs. Habrók binds on file count,
not bytes. Add a config block only if that budget changes.

## Provenance

This toolbox writes **no checkpoint manifest**, unlike `rfd3-cluster`. RFD4
already records `ckpt_path`, `ckpt_name` and `seed` per design in
`inference_metrics/per_example_metrics.csv`, and pins `hf_ckpt_revision` to an
exact Hub commit in its own config. The provenance question is answered
upstream.

## Layout

```
rfd4-setup-job       wrapper; picks an interpreter that has PyYAML
rfd4_tools/
  cli.py             argument surface and orchestration
  cluster.py         the cluster registry (separate from hpcjob's, deliberately)
  spec.py            spec authoring, trap guards, local validation
  generate.py        tier routing, walltime model, job.sh template
  logparse.py        cluster-side result summary (stdlib only, never raises)
verify/test_spec.py  regression tests for the parser
```

`rfd4_tools/` is copied into every generated job directory and invoked with
`PYTHONPATH="$_job_dir"`, so the cluster needs no install of this tool and a
job can never run a different version than the one that generated it.

## Tests

```bash
python3 verify/test_spec.py
```
