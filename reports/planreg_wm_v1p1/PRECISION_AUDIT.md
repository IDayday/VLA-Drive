# V1.1 precision audit

Evidence boundary: this directory separates old reported findings, source changes,
synthetic tests, new real-model checks, and experiments not yet run. No new full
Navtest score or matched no-WM causal result is claimed.

## Starting point and isolation

- Repository: IDayday/VLA-Drive.
- Sole base: `d9ca73f3d61f059285fcbf12a5bc81177ee350d7`.
- Source branch: `audit/planreg-wm-v1-comprehensive-20260904`.
- Implementation branch: `feature/planreg-wm-v1p1-precision-local-readout`.
- Isolated worktree: `/mnt/project/DriveVLA-M0-planreg-v1p1-precision-local-readout`.
- Original worktrees, main, old weights, old candidate banks and audit images
  were not modified. Their pre-existing untracked/modified files were left alone.

All six requested source reports were read before implementation. Their BF16 EMA
loss-of-increment finding is an **old checkpoint-based finding**, not a newly
repeated training result here.

## Implemented storage contract

Frozen base VLM tensors may remain BF16. All trainable leaves are stored FP32:
Q/V LoRA A/B, registers, readout, neck/norms, fusion, Q-Former, generator, scorer,
predictor. Residual conversions are explicit. Training uses BF16 autocast, not
whole-agent `.bfloat16()`. An initialized PlanReg agent rejects dtype downcasts.

FP32 AdamW moments are inspected after actual optimizer execution. Interval-500
logs contain parameter update/weight ratios, changed fractions, LR, teacher/student
distance and master/copy changed counts. This audit is not a per-step full-parameter
walk. Gradient finiteness and actual parameter changes are different checks.

EMA tracks only the trainable visual/readout subset, with FP32 persistent buffers:
`master += (1 - momentum) * (student.float() - master)`. Immutable frozen vision
base weights are not EMA-updated. The teacher has no LLM, Q-Former or planner.
Its forward copy is generated from master; a BF16 copy never overwrites master.
Device movement preserves master dtype and low bits. Checkpoints contain a
versioned byte-encoded schema, exact parameter-name mapping and FP32 masters.

EMA updates only after the actual optimizer post-hook fires. Accumulated
microbatches and skipped optimizer execution do not update EMA. Schedule position
uses Lightning's restored optimizer/global-step convention; a skipped attempt
can advance Lightning's attempted-step counter but never advances EMA tensors.
The callback avoids a second update when the Lightning optimizer hook owns it.

Legacy BF16 teacher values can seed migration only. Reports explicitly mark
`legacy_bf16_ema_history_unrecoverable=true`; past lost increments are not recovered.
An explicit old numerical update function remains available for audit replay.
New V1.1 runs create their teacher after VLM + shared-init restoration.

## Executed checks

- Synthetic: 1,000 sub-BF16-ULP updates accumulate in FP32 against an FP64
  reference; the forward copy eventually crosses a BF16 ULP.
- Synthetic: dtype conversion cannot downcast masters; resume trajectory is
  bitwise equivalent; legacy migration is explicit.
- Synthetic: actual AdamW steps change BF16-base/FP32-LoRA trainable leaves;
  base/K remain unchanged; moments are FP32.
- Real Lightning CPU integration: accumulation=2 produces exactly one EMA
  update per optimizer step; duplicate callback suppressed; skipped-step test.
- Two-rank CPU/Gloo DDP: same-batch diagnostic leaves pending ordinary backward,
  rank equality, optimizer gradients and RNG intact. This is not a 16-GPU run.
- Real InternVL3-2B: see `REAL_RUNTIME_SMOKE.json`, `PRECISION_CONTRACT.json`
  and `REAL_UPDATES_150.json`. The four-update smoke verifies all 24 adapted
  blocks have nonzero gradients, LLM has none, and actual moments are FP32.
- Real student-only export/reload: current-only trajectory difference is zero.
- Real old epoch33 four-scene FP32 replay: proposal max difference zero and
  identical selected indices (`LEGACY_REPLAY.json`). This is not new PDMS scoring.

Actual initial trainable count after freezing inactive heads is 23,926,559 in
465 tensors. Shared initialization also retains the 40 dormant-head tensors for
compatible loading: 29,292,415 parameters / 505 tensors in the artifact. Base/VQA
effective trainable initial states are bitwise identical. Q/V LoRA: 24 blocks,
48 Q/V adapters, 96 A/B Linear modules, 3,145,728 trainable parameters.

Do not interpret zero changes in some upstream predictor norm/embedding tensors
after only four tiny warmup updates as proof that FP32 is broken: the residual
output starts at zero and masks the first upstream gradient. The longer numerical
check reports actual changed and unchanged tensors, without replacing that
measurement with a finite-gradient claim.

The completed 150-update run observed changes in **all 465** effective trainable
tensors and retained FP32 storage/moments throughout. `TEACHER_DRIFT.json` also
compares real teacher outputs with the same shared-initialized teacher: future
target change RMS≈0.232, master-to-initial L2≈0.182, and master/forward-copy equality
is exact. The source checkpoint was not modified. Thus this is evidence of online
target change, not only a nonzero optimizer gradient.

## Retained failure evidence

External artifact root: `/mnt/project/DriveVLA-M0-formal-runs/v1p1_audit_20260905`.
`real_runtime_v1` completed four updates but its control helper mishandled FP64
GT trajectory dtype; fixed by explicit input conversion. `real_runtime_v2`
reached export but its audit-only Hydra field assignment failed; fixed with
`OmegaConf.update(..., force_add=True)`. `real_runtime_v3` passed. The first
legacy replay lacked `NUPLAN_MAPS_ROOT`; the second passed. These failed attempts
remain outside Git with their logs and are not reported as full passes.

The first diagnostic retained full eager attention activations, peaking near
71 GiB for microbatch two. The final low-frequency diagnostic uses temporary
non-reentrant per-block checkpointing, not a global torch monkey-patch; normal
training retains its original checkpointing. See the longer real audit for
the final-path memory observation. It is not a layout throughput benchmark.
