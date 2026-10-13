# Own expert + scratch internal scorer

Workspace: `/mnt/project/DriveVLA-M0-iqe-internal-scorer`.
Experiment: `own_query_wa_scorer_seed31415_v1`.
Artifacts: `/mnt/project/iqe-internal-scorer-20261012/own_wa_v1`.

The immutable binding resolves the own Base run to
`/mnt/project/iqe-remote-rolling-cache-20261012/continuation/step_072000.pt`,
SHA256 `412f6c78b57e3bb1daa088ea4862a43b9f67630fcd31c44cd27ecf9a1a81126d`.
It never refreshes during an experiment. This is an offline technical binding,
not production qualification or permission to update ACTIVE.

Run commands from the workspace. Read `/mnt/project/server_dispatch_policy.json`
and check GPU ownership before a new dispatch. The actual worker for this run is
`training-vla-zt-worker-0`. The own expert/scorer work and deployment timing have
finished. All eight local GPUs now serve the user's restored Base dynamic cache.
The commands below are reproduction/recovery instructions, not permission to
dispatch over those processes. Other workloads must remain untouched.

```bash
cd /mnt/project/DriveVLA-M0-iqe-internal-scorer
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/root/miniconda3/envs/ddp/bin/python
CFG=configs/iqe/internal_baseline/own_wa_v1.yaml
OUT=/mnt/project/iqe-internal-scorer-20261012/own_wa_v1

# Resolve once, verify complete checkpoint, freeze roles and configuration.
"$PY" -m iqe.internal_baseline.cli bind-base --config "$CFG"

# One-worker recovery of the complete feature manifest. Existing records are
# checksum/contract checked and reused. No VLM is loaded if none are missing.
CUDA_VISIBLE_DEVICES=0 "$PY" -m iqe.internal_baseline.cli cache-features \
  --config "$CFG" --shard 0 --shards 1 --device cuda:0

# Actual images, label-poisoning and original Base versus K=1 audit (32 scenes).
CUDA_VISIBLE_DEVICES=0 "$PY" -m iqe.internal_baseline.cli audit-online --config "$CFG" --device cuda:0

# Official single-reference Base and difficult-GT labels; fixed 90/10 manifest.
"$PY" -m iqe.internal_baseline.cli build-hard-set --config "$CFG"

# The complete workflow also performs planned boundary resumes at expert250
# and scorer500. Do not launch individual mutating commands while it is active.
CUDA_VISIBLE_DEVICES=0 "$PY" -m iqe.internal_baseline.cli run --config "$CFG" --device cuda:0
```

Individual stages, for inspection/recovery after the workflow has exited:

```bash
# 1,000 expert updates, global128/micro32, independent AdamW1e-4/wd.01,
# warmup50, original dropout/DropPath, validation250 on fixed stage_val.
CUDA_VISIBLE_DEVICES=0 "$PY" -m iqe.internal_baseline.cli train-expert \
  --config "$CFG" --device cuda:0 --resume auto

# Both own candidates on all fixed roles; CPU official labels, 16 workers,
# three retry attempts per invocation. GT is never a deployment candidate.
CUDA_VISIBLE_DEVICES=0 "$PY" -m iqe.internal_baseline.cli build-pair-bank --config "$CFG" --device cuda:0

# 2,000 scorer updates, global128/micro16, AdamW1e-4/wd.01,warmup100.
# First500 pointwise128; then1500 natural64+comparison64. One scheduler.
CUDA_VISIBLE_DEVICES=0 "$PY" -m iqe.internal_baseline.cli train-scorer \
  --config "$CFG" --device cuda:0 --resume auto

# Whole-workflow recovery after a complete verified bank, as actually used.
# Do not rebuild/recapture features or resolve a different latest Base.
CUDA_VISIBLE_DEVICES=0 "$PY" -m iqe.internal_baseline.cli run \
  --config "$CFG" --device cuda:0 --from-stage scorer

# Raw argmax, optional independently calibrated margin, fixed full dev1696.
CUDA_VISIBLE_DEVICES=0 "$PY" -m iqe.internal_baseline.cli evaluate --config "$CFG" --device cuda:0
CUDA_VISIBLE_DEVICES=0 "$PY" -m iqe.internal_baseline.cli diagnostics --config "$CFG" --device cuda:0
CUDA_VISIBLE_DEVICES=0 "$PY" -m iqe.internal_baseline.cli export-bundle --config "$CFG" --device cuda:0
CUDA_VISIBLE_DEVICES=0 "$PY" -m iqe.internal_baseline.cli benchmark --config "$CFG" --device cuda:0

# Separate real-data resume proof, no main-run warm start or optimizer changes.
CUDA_VISIBLE_DEVICES=0 "$PY" -m iqe.internal_baseline.resume_audit --config "$CFG" --device cuda:0 --deterministic
CUDA_VISIBLE_DEVICES=0 "$PY" -m iqe.internal_baseline.audit_training --config "$CFG" --device cuda:0
```

For a new experiment, omit `--resume` on the first trainer invocation. For this
existing experiment, use `--resume auto`; it verifies complete receipts and
selects the highest compatible step. A terminal 1,000/2,000-step checkpoint
returns without further optimizer updates. Restore only within the same
environment/world size/global batch and declared data/Base identity.
The production CUDA backend is not asserted to be bitwise deterministic: the
original real-data comparison and its tiny pre-interruption differences remain
in `REAL_RESUME_AUDIT.json`. The separate `--deterministic` proof retains normal
dropout and the same full LR timeline, and checks parameter/Adam/RNG/cursor/loss/
gradient identity. It does not change the main run's backend or weights.
The actual first expert boundary is
`/mnt/project/iqe-internal-scorer-20261012/own_wa_v1/expert/step_000250.pt`;
subsequent automatic recovery must choose newer completed progress, not return
to that historical boundary. Do not explicitly resume an older checkpoint into
the same run over newer progress.

The feature-cache parallel form used eight independent, globally indexed shards:

```bash
# Only when all selected GPUs are verified available. No automatic preemption.
for rank in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$rank" "$PY" -m iqe.internal_baseline.cli cache-features \
    --config "$CFG" --shard "$rank" --shards 8 --device cuda:0 \
    > "$OUT/cache_${rank}.log" 2>&1 &
done
wait
```

Change Base only in a **new** configuration and output namespace. The following
creates an explicit-checkpoint rerun using the same actual 72k snapshot, with
new caches/candidates and a fresh scorer; edit `checkpoint` to another verified
file within the declared own Base run before binding to choose a different Base.
It does not start training:

```bash
"$PY" - <<'PY'
from pathlib import Path
import yaml
p=Path('configs/iqe/internal_baseline/own_wa_v1.yaml')
c=yaml.safe_load(p.read_text())
c['experiment_id']='own_query_wa_scorer_explicit72k_rerun'
c['output_root']='/mnt/project/iqe-internal-scorer-20261012/explicit72k_rerun'
c['checkpoint']='/mnt/project/iqe-remote-rolling-cache-20261012/continuation/step_072000.pt'
destination=p.with_name('explicit72k_rerun.yaml')
with destination.open('x') as f: yaml.safe_dump(c,f,sort_keys=False)
PY
"$PY" -m iqe.internal_baseline.cli bind-base \
  --config configs/iqe/internal_baseline/explicit72k_rerun.yaml
```

Never point the new experiment at an old `output_root`. `latest_complete` is
also supported, but resolved once by `bind-base`, before score inspection.
This run never accesses final_test/navtest, launches Base training, or updates
formal ACTIVE. Later own-lineage scorer warm start is an explicit tested
interface in `incremental.py`, not enabled for first-round training.

The selected files are `expert/best_step_000500.pt` and
`scorer/best_step_000750.pt`; the completed training states are
`expert/step_001000.pt` and `scorer/step_002000.pt`. The diagnostic deployment
artifact is `bundle/BUNDLE.json`; its scorer parameters are in the same
`bundle/algorithm.pt` as the private expert. It is not an ACTIVE release.

The user's later explicit request to restore the original continuing Base and
dynamic preprocessing was handled separately in
`/mnt/project/DriveVLA-M0-iqe-base-resume-72k-20261013`, with receipts under
`/mnt/project/iqe-remote-rolling-cache-20261012/recovery_72k_20261013`.
The latest instruction also withdraws cache/G-series GPU sharing. Its Base and
cache producers must remain running; they are genuine research, not fillers.
The original local pressure-script restoration is deferred while these real
jobs occupy all GPUs. Its captured command and restore script remain outside
the repository under `own_wa_v1/pressure_restore/`; no private environment is
committed. No G-series process was signalled by the resource relocation.
