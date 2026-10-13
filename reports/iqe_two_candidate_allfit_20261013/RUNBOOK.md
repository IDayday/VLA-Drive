# Full-fit two-candidate experiment

The current preparation and workflow queue are already running on
`training-vlawm-tsinghua-zt2`. Do not duplicate them. Inspect
`/mnt/project/iqe-two-candidate-baseline-20261013/base70k_allfit_v1/runtime/HEAD_QUEUE_STATUS.json`
and the live PID/start-time receipts first. Reserve GPU1 for the queued head
workflow after its context worker releases it; other jobs retain their assigned GPUs.

The commands below are the actual interfaces, for stage execution after the
existing owner has exited or for reconstructing the experiment. Run in:

```bash
cd /mnt/project/DriveVLA-M0-iqe-two-candidate-all-fit-20261013
IQE_PY=/root/miniconda3/envs/ddp/bin/python
IQE_CFG=configs/iqe/two_candidate_baseline/base70k_allfit_v1.yaml
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Binding, splits and the explicit Scorer scope were already completed, before
training or full-population scores. Repeating these commands verifies/reuses
their fixed identities; it must not refresh Base or replace the original config.

```bash
"$IQE_PY" -m iqe.two_candidate init-baseline --config "$IQE_CFG"
"$IQE_PY" -m iqe.two_candidate build-baseline-splits --config "$IQE_CFG"
"$IQE_PY" -m iqe.internal_baseline.expansion scorer-all-fit \
  --config "$IQE_CFG" --reason 'User confirmed all 90892 legal fit scenes; expert exposure reported separately'
"$IQE_PY" -m iqe.internal_baseline.expansion reuse-context \
  --source-config /mnt/project/DriveVLA-M0-iqe-two-candidate-baseline-v1/configs/iqe/two_candidate_baseline/base70k_v1.yaml \
  --config "$IQE_CFG" --workers 2
```

The running three workers correspond to these commands on physical GPUs1/2/3.
Each uses microbatch4 and its fixed disjoint manifest slice. CPU official scoring
has 16 workers, a 12-hour execution limit, and chunks of at most 512 scenes.

```bash
CUDA_VISIBLE_DEVICES=1 "$IQE_PY" -m iqe.two_candidate cache-baseline-context --config "$IQE_CFG" --shard 0 --shards 3 --device cuda:0
CUDA_VISIBLE_DEVICES=2 "$IQE_PY" -m iqe.two_candidate cache-baseline-context --config "$IQE_CFG" --shard 1 --shards 3 --device cuda:0
CUDA_VISIBLE_DEVICES=3 "$IQE_PY" -m iqe.two_candidate cache-baseline-context --config "$IQE_CFG" --shard 2 --shards 3 --device cuda:0
CUDA_VISIBLE_DEVICES= "$IQE_PY" -m iqe.internal_baseline.expansion score-stream --config "$IQE_CFG" --max-seconds 43200 --chunk-size 512
```

After complete inputs, the existing queue executes the following continuous
workflow on GPU1. This includes one expert (10,000 updates), its frozen pair bank,
one scratch Scorer (20,000 updates), fixed dev evaluation, diagnostics, research
bundle, actual-consumption exposure audit and online latency. It does not run
Navtest, train another Base, add another expert or update ACTIVE.

```bash
CUDA_VISIBLE_DEVICES=1 "$IQE_PY" -m iqe.two_candidate run-two-candidate-baseline --config "$IQE_CFG" --device cuda:0
```

Equivalent individual stages, only with no active workflow owner:

```bash
"$IQE_PY" -m iqe.two_candidate build-expert-data --config "$IQE_CFG"
CUDA_VISIBLE_DEVICES=1 "$IQE_PY" -m iqe.two_candidate train-baseline-expert --config "$IQE_CFG" --device cuda:0
"$IQE_PY" -m iqe.two_candidate evaluate-baseline-expert --config "$IQE_CFG"
CUDA_VISIBLE_DEVICES=1 "$IQE_PY" -m iqe.two_candidate build-pair-bank --config "$IQE_CFG" --device cuda:0
CUDA_VISIBLE_DEVICES=1 "$IQE_PY" -m iqe.two_candidate train-baseline-scorer --config "$IQE_CFG" --device cuda:0
CUDA_VISIBLE_DEVICES=1 "$IQE_PY" -m iqe.two_candidate calibrate-baseline-margin --config "$IQE_CFG" --device cuda:0
CUDA_VISIBLE_DEVICES=1 "$IQE_PY" -m iqe.two_candidate evaluate-two-candidate-baseline --config "$IQE_CFG" --device cuda:0
CUDA_VISIBLE_DEVICES=1 "$IQE_PY" -m iqe.two_candidate export-research-bundle --config "$IQE_CFG" --device cuda:0
```

Recovery is available only for genuine interruptions; no artificial pause/resume
experiment is scheduled. `--resume auto` resolves the latest complete compatible
head checkpoint within this experiment, preserving its optimizer, sampler and
learning-rate timeline. No head checkpoint exists yet at this launch report.

```bash
CUDA_VISIBLE_DEVICES=1 "$IQE_PY" -m iqe.two_candidate train-baseline-expert --config "$IQE_CFG" --device cuda:0 --resume auto
CUDA_VISIBLE_DEVICES=1 "$IQE_PY" -m iqe.two_candidate train-baseline-scorer --config "$IQE_CFG" --device cuda:0 --resume auto
```

For another Base, copy the configuration to a new experiment ID/output root and
explicit checkpoint or lineage-bound `latest_complete`, initialize once, and
build new contexts and labels. Do not reuse this Base's language/planning context
or score identities for another checkpoint.

Regression command actually run:

```bash
OMP_NUM_THREADS=2 "$IQE_PY" -m pytest -q \
  tests/iqe/test_scorer_all_fit_scope.py \
  tests/iqe/test_all_fit_context_reuse.py \
  tests/iqe/test_two_candidate_protocol.py \
  tests/iqe/test_two_candidate_budget.py
```

Result: 81 passed, one existing pynvml deprecation warning, 1.62 seconds.
