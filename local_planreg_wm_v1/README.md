# PlanReg-WM-V1 local experiments

These launchers use `agent=episode_drive_planreg_wm_v1`, the frozen public M0
checkpoint, and the exact DrivoR scorer implementation pinned in the code.
They never search for a checkpoint: fresh runs start from
`PLANREG_BASE_CHECKPOINT`; continuation requires an explicit
`RESUME_CHECKPOINT=/absolute/path.ckpt`.

The local defaults point to the currently audited NAVSIM installation. Portable
runs can override `PLANREG_NAVSIM_LOG_ROOT`, `PLANREG_SENSOR_BLOB_ROOT`,
`PLANREG_TRAIN_METRIC_CACHE`, `PLANREG_NAVTEST_METRIC_CACHE`,
`NUPLAN_MAPS_ROOT`, `PLANREG_BASE_CHECKPOINT`, and `PLANREG_VLM_PATH`.

Examples:

```bash
# Inspect the fully resolved command without launching.
DRY_RUN=1 bash local_planreg_wm_v1/train_e3_register_qvlora_wm.sh 0

# Two-batch small-data smoke.
SMOKE_SPLIT=1 SMOKE_SCENES=32 \
  bash local_planreg_wm_v1/train_e3_register_qvlora_wm.sh 0

# Full navtrain; TRAIN_TEST_SPLIT=trainval is also supported.
PLANREG_TRAIN_TEST_SPLIT=navtrain \
  bash local_planreg_wm_v1/train_e3_register_qvlora_wm.sh 0

# Lossless Lightning continuation (optimizer/scheduler/global step/EMA buffers).
RESUME_CHECKPOINT=/absolute/path/to/last.ckpt \
  bash local_planreg_wm_v1/train_e3_register_qvlora_wm.sh 0
```

E0, E2 and E3 accept seeds 0/1/2. E1 and control experiments E4–E7
intentionally accept seed 0 first. Every real launch records the git commit,
dirty status, redacted environment, resolved Hydra config, command, and log in
`OUTPUT_DIR/run_metadata/`; optimizer groups are printed into that log.

Evaluation takes explicit label/checkpoint pairs, derives its explicit seed from
the `_seedN` suffix, and uses FP32 trainer precision:

```bash
bash local_planreg_wm_v1/evaluate_all.sh \
  e0_semantic_exact_scorer_seed0=/path/e0.ckpt \
  e3_register_qvlora_wm_seed0=/path/e3.ckpt
```

`collect_candidate_metrics.py` only summarizes a previously scored immutable
candidate NPZ. Its best-of-K field is an offline oracle candidate-bank upper
bound, not deployable PDMS. Use `compare_experiments.py` to assemble reports;
promotion still requires held-out validation before complete Navtest.
