# PlanReg-WM-V1 formal training

The formal experiment consists of two runs only:

- `formal_base_init_wm`: standalone Base InternVL3-2B initialization.
- `formal_vqa_init_wm`: standalone Driving-VQA InternVL3-2B initialization.

They train for the same 27 full epochs over 103,288 trainval records, use the
same seed-specific random planning/action/scorer/world-model artifact, and load
no M0/full-agent checkpoint. Both enable correct-future, GT-conditioned world
model supervision from optimizer step zero. The older bootstrap, E0-E7, and
R1-R3 scripts remain reproducibility utilities but are not formal launchers and
are not run in this comparison.

## Required artifacts

Formal launch refuses to start without all of the following:

1. The paired VLM audit at
   `reports/planreg_wm_v1/formal_vlm_initialization_audit.json`.
2. A seed-matched `shared_planreg_init_seedN.pt` whose paired Base/VQA runtime
   state is bitwise identical.
3. An input-only cache manifest proving exactly 103,288 complete records and no
   cached VLM/Q-Former/register/EMA outputs.
4. Completed throughput evidence referenced by the shared
   `formal_training_layout_lock.json` and the immutable lock itself.
5. A clean git worktree and explicit standalone Base/VQA VLM paths.
6. Matching cross-node runtime fingerprints (Python ABI, PyTorch,
   Transformers, PEFT, Lightning, EpisodeDrive source, and InternVL
   `trust_remote_code` source hashes).

Prepare or re-audit the VLM pair and shared state with:

```bash
python scripts/audit_formal_vlm_initialization.py \
  --base "$PLANREG_BASE_VLM_PATH" \
  --driving-vqa "$PLANREG_VQA_VLM_PATH" \
  --output reports/planreg_wm_v1/formal_vlm_initialization_audit.json \
  --load-runtime-classes

python scripts/create_shared_planreg_initialization.py \
  --seed 0 \
  --architecture-config navsim/planning/script/config/common/agent/episode_drive_planreg_wm_formal_base.yaml \
  --output /absolute/shared_planreg_init_seed0.pt \
  --metadata-output reports/planreg_wm_v1/shared_trainable_init_seed0.json
```

If Driving-VQA is a PEFT adapter, run
`scripts/prepare_merged_vqa_vlm_init.py` first. A dense VQA checkpoint is
normalized without adapter merging; both paths prove fixed-input forward parity
and absence of agent/action/scorer keys.

## Throughput lock

Run all layouts on the Base VLM with the same cache and shared initialization:

```bash
bash local_planreg_wm_v1/benchmark_formal_8x2.sh
bash local_planreg_wm_v1/benchmark_formal_8x4.sh
bash local_planreg_wm_v1/benchmark_formal_16x2.sh
bash local_planreg_wm_v1/benchmark_formal_16x4.sh
bash local_planreg_wm_v1/benchmark_formal_16x6.sh
bash local_planreg_wm_v1/benchmark_formal_16x8.sh

python scripts/select_formal_training_layout.py \
  --metrics-root reports/planreg_wm_v1/throughput \
  --output reports/planreg_wm_v1/formal_training_layout_lock.json
```

The 16-GPU scripts use `vla-zt` and `vla-zt2`. They force one shared Python
3.9/package tree and one shared Hugging Face code cache, then reject any
cross-node fingerprint difference before model construction. No script
preempts a busy GPU.
Each benchmark runs 20 warmup and 300 measured optimizer steps of the complete
PlanReg-WM graph. The selector enforces finite losses/gradients, no OOM/deadlock,
peak allocation below 72 GiB, peak reservation below 76 GiB, and bounded p90
step-time jitter. Among layouts within 95% of peak sample throughput, it chooses
the smallest global batch to retain more optimizer updates without materially
increasing total wall time. BaseInit and VQAInit must use the same lock.

The selected 16x8 layout uses split SDPA for the mathematically read-only
register attention, disables vision gradient checkpointing, and uses two exact
PDM proposal partitions per scene. Exact detached CPU metric targets run
concurrently with the no-grad EMA future-vision forward. All settings are
recorded in the lock and applied by the formal launchers; they cannot drift
between BaseInit and VQAInit.

Before benchmarking, the real-data gate uses 16 disjoint train and 16
validation scenes (32 total), executes two train batches and one validation
batch, requires all three future horizons in the WM batches, checks finite
losses/gradients, exports a student, and performs current-only inference:

```bash
bash local_planreg_wm_v1/smoke_formal_real_data.sh 0
```

## Formal launch

```bash
PLANREG_LAYOUT_LOCK=/absolute/formal_training_layout_lock.json \
PLANREG_SHARED_INIT=/absolute/shared_planreg_init_seed0.pt \
PLANREG_BASE_VLM_PATH=/absolute/InternVL3-2B-base \
bash local_planreg_wm_v1/train_formal_base_init_wm.sh 0

PLANREG_LAYOUT_LOCK=/absolute/formal_training_layout_lock.json \
PLANREG_SHARED_INIT=/absolute/shared_planreg_init_seed0.pt \
PLANREG_VQA_VLM_PATH=/absolute/InternVL3-2B-driving-vqa \
bash local_planreg_wm_v1/train_formal_vqa_init_wm.sh 0
```

When the selected layout occupies both machines, queue the paired runs in
their declared order with the same lock and shared state:

```bash
PLANREG_LAYOUT_LOCK=/absolute/formal_training_layout_lock.json \
PLANREG_SHARED_INIT=/absolute/shared_planreg_init_seed0.pt \
PLANREG_BASE_VLM_PATH=/absolute/InternVL3-2B-base \
PLANREG_VQA_VLM_PATH=/absolute/InternVL3-2B-driving-vqa \
bash local_planreg_wm_v1/train_formal_dual_init_sequential.sh 0
```

Automatic resume is disabled. Lossless continuation requires an explicit
`RESUME_CHECKPOINT` from that exact output directory and matching identity
hashes. The launchers compose both resolved configs and reject every difference
except VLM identity, variant label, experiment name, and output directory.

At global batch 32, peak LRs are `2e-4` for planning adapter, fusion, generator,
scorer, and predictor; `1e-4` for Q-Former; and `4e-5` for vision Q/V LoRA.
They scale by the square root of actual global batch subject to configured
caps. AdamW uses weight decay 0.01/0.0, 5% warmup from 1% peak, cosine decay to
10% peak, and norm clipping at 1.0.

## Deployment and evaluation

Epoch 27 is fixed before seeing Navtest. Export and evaluate with:

```bash
python scripts/export_planreg_student_checkpoint.py \
  /absolute/epoch_27_final.ckpt \
  /absolute/formal_epoch27_student.ckpt \
  --resolved-config /absolute/resolved_hydra_config.yaml

bash local_planreg_wm_v1/evaluate_formal_checkpoint.sh \
  base /absolute/formal_epoch27_student.ckpt
```

The evaluator verifies a student-only checkpoint, performs a mandatory public
four-scene gate, then runs selected-trajectory Navtest PDMS and resumable
official scoring of all 64 candidates. “Oracle@64” is reported as an offline
upper-bound diagnostic, alongside regret, proposal distribution, six PDM
components, duplicates/clusters, register rank, gates, and latency. Evaluation
is BF16 VLM plus FP32 action/scorer and accepts only the current front frame;
EMA, predictor, and future keys are absent.

PlanReg-WM-V1 does not implement multi-trajectory consequence modeling. Its
future-predictor API carries K, but formal supervision is always K=1 from the GT
trajectory and real future frames; the scorer never reads predicted future
registers.
