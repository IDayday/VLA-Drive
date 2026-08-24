# GP-SQ3D-Mix repair

## Scope and provenance

This branch implements **GP-SQ3D-Mix (Geometry-Preserving
Scene-Conditioned 3D-Mix)** as a new framework. It was forked from
`feature/sq-3d-mix` at
`25b2c50a05740e751a848640475e2522e07298a6`; the legacy framework, classes,
100k checkpoints, report and result CSVs remain loadable and unchanged.

The code checkout is
`/mnt/zhangt_workspace/project/VLA-Drive-GPSQ3DMix` inside DLC. Durable
datasets, weights, caches and historical runs remain under
`/mnt/zhangt_workspace/project/DriveDreamer-Policy`. These paths intentionally
differ: DLC executes the branch visible through the GP worktree while reading
shared artifacts from the original asset root.

## 1. Legacy SQ-3D-Mix failure confirmation

The prior full-navtest result remains a negative result: the legacy 100k model
has PDMS `0.886634` and EPDMS `0.881808`; at 100k its semantic gate is `1.0`,
effective geometry weight is `0.0`, and the 16 scene queries have pairwise
cosine `1.0`. The matched 90k PDMS delta is `-0.002320`, with paired 95%
bootstrap CI `[-0.005338, 0.000611]`. These values are copied only by reference
from `SQ3DMIX_GATED_20260821.md`; that report and its three source CSVs were not
edited.

The failure mechanism is architectural, not simply an insufficient learning
rate. The 16 learned scene queries collapsed to one direction, the scalar
semantic/geometry convex gate learned the semantic shortcut, and uncentered
tokens changed the Action DiT conditioning even when they carried no useful
scene-specific geometry. In addition, rank-stream inference noise made earlier
cross-topology intervention comparisons confounded. GP-SQ3D-Mix removes the
convex semantic branch, preserves per-slot camera/UV/ray metadata, centers the
geometry readout against the train-set slot mean, and writes only a bounded
residual into the existing eight action queries.

The required same-checkpoint, same-split and same-noise 2k intervention
completed with 2,000/2,000 successful scenes in every mode:

| mode | PDMS | EPDMS |
| --- | ---: | ---: |
| real | 0.883327 | 0.890599 |
| zero | 0.877250 | 0.885803 |
| shuffled | 0.883619 | 0.890920 |

| paired comparison | metric | mean delta | paired bootstrap 95% CI | trajectory L2 mean / median | identical ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| real - zero | PDMS | +0.006078 | [+0.001156, +0.011156] | 0.916051 / 0.874623 | 0.0% |
| real - zero | EPDMS | +0.004797 | [-0.000397, +0.010196] | 0.916051 / 0.874623 | 0.0% |
| real - shuffled | PDMS | -0.000291 | [-0.000876, +0.000002] | 0.004178 / 0.000668 | 0.0% |
| real - shuffled | EPDMS | -0.000320 | [-0.000964, +0.000002] | 0.004178 / 0.000668 | 0.0% |

This is the missing causal confirmation. Replacing VGGT by zero changes every
trajectory and measurably hurts PDMS, so the legacy VGGT route is not literally
dead. But synchronously replacing each scene's VGGT input by another scene's
input barely changes its trajectory, and shuffled is marginally better rather
than worse on both metrics. The model therefore responds mainly to the generic
presence/distribution of VGGT features, not to their scene correspondence. The
old architecture learned an input-distribution effect without useful
scene-conditioned geometry, which explains why an active route did not produce
a policy improvement.

The committed raw summaries are
`results/legacy_sq3dmix_interventions_2000.csv` and
`results/legacy_sq3dmix_interventions_2000_paired.csv`. The immutable external
run is
`/mnt/zhangt_workspace/project/DriveDreamer-Policy/navsim_exp/legacy_sq3dmix_interventions/gp2k`;
its manifest records checkpoint SHA256
`3dcf03d3c92c2f076f81b798783f52ff9c3a24d81d90deac5cc63f4eb3a98732` and
datalist SHA256
`f9b2bec623ab49625d48ca29f22fb174283798aca348ff4932470d6725ef6173`.

The causal test uses SHA256-derived per-token initial noise and a global,
nearest-action-target shuffled scene mapping constructed before rank sharding.
Consequently real/zero/shuffled results are invariant to world size, rank,
batch size and worker count. A complete 12,146-scene rerun was not attempted on
the single available PPU.

## 2. Stage A representation and utility gate

The new route removes the previous semantic shortcut and duplicated context:

- a parameter-free masked scene mean excludes padding and all eight action
  tokens;
- each front/left/right view is pooled independently to 6x10 while retaining
  view ID, UV, ego-frame ray origin and normalized ray direction;
- one shared 2048-to-512 adapter constructs both real and slot-mean reference
  memory;
- a bounded scene-conditioned retention gate initializes exactly at 0.10;
- a centered eight-query cross-attention reader writes only a bounded residual
  into the original action queries;
- the Action DiT receives exactly `[B,8,2048]` and `extra_context=None`.

Stage A reuses one `FlowMatchingState` and one matched Action-DiT dropout stream
for baseline, real and shuffled losses. Only the adapter, gate and reader are
trainable. The action-only checkpoint is mandatory and is loaded through a
strict, explicit warm-start API.

Status: **not run**. This host exposes one PPU, while the fixed launcher
requires 8 PPU x batch 4 x accumulation 1 (effective batch 32). The immutable
full-train slot statistics also have not yet been computed from the 103,288
sample dense cache. No Stage A gate has therefore passed; this is an unevaluated
gate, not a failed scientific result.

## 3. Stage B matched policy pilot

Stage B is implemented as a matched 10k pair: an action-only continuation
control and GP-SQ3D-Mix continuation start from the same action-only checkpoint,
seed, ordered datalist, global batch, optimizer and scheduler. Qwen and the
state input adapter remain frozen. Action DiT uses LR `1e-6`, GP modules use
`3e-5`, the retention projection uses `1e-5`, and `scale_logit` uses `1e-6`.
Checkpoints are saved every 2k and selected jointly by real PDMS,
real-minus-shuffled paired PDMS and fidelity to the control.

Status: **not run**, because Stage A has not passed. Both the training and
evaluation launchers read the Stage A JSON decision and fail closed unless
`all_passed` is exactly true.

## 4. Formal long training

No 100k (or other formal long) GP-SQ3D-Mix training was launched. The current
decision is **NO-GO for Stage B and NO-GO for long training** until all fixed
Stage A gates pass. Stage B itself must then pass deterministic intervention,
paired PDMS/control, bounded-residual, nonzero-gradient and non-collapse gates
before a long run can be recommended.

## Reproduction sequence

All commands must run from the DLC-visible GP worktree and source machine paths
from ignored `env.local.sh`:

```bash
bash 18-eval_legacy_sq3dmix_interventions.sh --world-size 2 --batch-size 4
bash 19-compute_gp_sq3dmix_slot_stats.sh
bash 20-train_gp_sq3dmix_stage_a.sh
bash 21-eval_gp_sq3dmix_stage_a.sh --run-dir "$GP_STAGE_A_RUN_DIR"

# These remain fail-closed until Stage A passes.
bash 22-train_gp_sq3dmix_stage_b.sh
bash 23-eval_gp_sq3dmix_stage_b.sh \
  --gp-run "$GP_STAGE_B_RUN_DIR" \
  --control-run "$GP_STAGE_B_CONTROL_RUN_DIR" \
  --gate-report "$GP_STAGE_A_GATE_REPORT"
```

Every launcher supports `--dry-run`, refuses to overwrite an experiment
directory, records the code commit and relevant hashes, and uses `per_token`
inference noise. Training defaults to 8 PPU with per-device batch 4 and
effective global batch 32.

## Validation

- `compileall`: pass.
- New GP-SQ3D-Mix tests: 20 passed.
- QwenOFT/SQ-3D-Mix regression tests: 55 passed.
- Dense VGGT cache/pooling regression tests: 18 passed, 1 skipped because
  the optional GPU-only condition was unavailable.
- Scripts 18 through 23: all dry-run paths pass.
- Legacy fixed-2k intervention: 6,000 predictions and all six PDMS/EPDMS
  scoring jobs completed; every mode has 2,000 successful and zero failed
  scenarios.
- Preserved legacy report/CSV SHA256 values match the source commit.
