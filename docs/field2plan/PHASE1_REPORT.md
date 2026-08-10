# Field2Plan Phase 1 Report

Date: 2026-08-08 UTC

## Completed scope

- Added the opt-in `QwenOFT_Field2Plan` framework. The legacy `QwenOFT`
  framework and its default configuration remain unchanged.
- Added a single-source trajectory codec, calibrated camera contract, scoped
  visual feature tap, teacher-free geometry writer, trajectory-tube reader,
  zero-initialized physical-space refiner, diagnostics, and weighted loss
  aggregation.
- Added strict proposal-cache loading infrastructure. The Phase-1 debug path
  alone may generate a frozen online proposal; the main configuration fails
  fast when its cached proposal is missing.
- Added optional inference diagnostics without changing the evaluator-facing
  `{token}.npy` trajectory files.
- Added debug/main configurations, a coordinate sanity visualizer, and a
  non-interactive two-device launcher which preserves effective batch size 32.

No external teacher is imported or run. The geometry writer receives only
current visual features and camera calibration; it has no future action input.

## Modified and added files

- `starVLA/model/framework/QwenOFT_Field2Plan.py`
- `starVLA/model/modules/field2plan/{types,trajectory_codec,camera_geometry,visual_feature_tap,geometry_field_writer,semantic_writer,trajectory_tube_reader,trajectory_refiner,losses,diagnostics}.py`
- `starVLA/dataloader/field2plan_cache.py`
- `starVLA/dataloader/navsim_dataset.py`
- `starVLA/model/framework/__init__.py`
- `starVLA/training/train_starvla.py`
- `infer.py`
- `starVLA/config/training/cfg_field2plan_mvp.yaml`
- `starVLA/config/training/cfg_field2plan_mvp_debug.yaml`
- `scripts/field2plan/01_debug_mvp.sh`
- `tools/field2plan/visualize_coordinates.py`
- `tests/field2plan/`
- `artifacts/field2plan/baseline_manifest.example.json`

## Actual tensor contract

- Baseline/final normalized action: `[B,M,8,4]`, columns
  `[x_normalized,y_normalized,sin(theta),cos(theta)]`; inference removes `M=1`
  and returns `[B,8,4]`.
- Physical draft and bounded delta: `[B,M,8,3]`, columns `[x_m,y_m,theta_rad]`.
- Camera intrinsics / ego-to-camera / image size: `[B,V,3,3]`,
  `[B,V,4,4]`, `[B,V,2]`; MVP uses `V=3` in order
  `cam_f0,cam_l0,cam_r0`.
- Visual feature map: `[B,V,2048,Hf,Wf]`.
- Geometry field: `[B,Cg,Ny,Nx]`; debug is `[B,64,24,24]`, main is
  `[B,256,64,64]`.
- Tube points: `[B,M,8,S,3]`; current config has `S=6` from three lateral by
  two longitudinal offsets.
- Waypoint context: `[B,M,8,Cr]`; debug `Cr=64`, main `Cr=256`.
- Refiner output projection is exactly zero initialized, so initial
  `final_action == draft_action` and `delta_norm == 0`.

The field projection is a channel-last pointwise MLP. An earlier Conv2d
projection reproducibly faulted in the real PPU BF16 convolution backward
kernel at a 24x24 field. The MLP preserves the Phase-1 per-anchor projection
role and passed the real two-device backward smoke.

## Commands actually executed

```bash
pytest tests/field2plan -q
python -m compileall -q starVLA/model/modules/field2plan \
  starVLA/model/framework/QwenOFT_Field2Plan.py \
  starVLA/dataloader/field2plan_cache.py tools/field2plan
bash -n scripts/field2plan/01_debug_mvp.sh
git diff --check
NCCL_DEBUG=WARN FIELD2PLAN_DEBUG_NUM_PROCESSES=2 \
  bash scripts/field2plan/01_debug_mvp.sh
```

## Test results

- PASS: 32 CPU tests in `tests/field2plan`.
- PASS: compileall, launcher syntax, whitespace check, and static scans for
  bare `except:` and unconditional `.cuda()` in new paths.
- PASS: real two-device PPU / Accelerate / DeepSpeed ZeRO-2 / BF16 smoke,
  one optimizer step, effective batch `2 devices * 2 batch * 8 accumulation =
  32`. Observed finite `plan_loss=0.0066855`, `field_valid_ratio=0.32224`,
  `tube_valid_ratio=1.0`, and `delta_norm=0.0`; exit code 0.
- NOT RUN: a full Field2Plan training experiment or NAVSIM score. Phase 1 has
  no trained Field2Plan checkpoint and must not be assigned a PDMS result.

## Known risks

- Main training intentionally cannot start until Phase 2 produces and validates
  a baseline draft cache.
- The configured lidar-to-planning-ego transform is explicit identity for the
  audited NAVSIM data. Projection overlays look consistent on the checked mini
  sample, but broader coordinate QA remains required before a full experiment.
- Visual feature spatial layout is tied to the audited local Qwen3-VL wrapper;
  different checkpoints/grid layouts must fail contract checks rather than be
  silently reshaped.
- The two-device smoke validates execution, gradients, and strict degeneration,
  not scientific improvement.

## Next stage

Proceed to Phase 2 in this order: baseline draft cache and manifest validation,
DA3 cache adapter/schema, geometry auxiliary targets and losses,
supervision-by-access controls, then CPU and GPU acceptance. VGGT remains an
explicit lazy-import skeleton unless a local repository and checkpoint are
actually supplied.
