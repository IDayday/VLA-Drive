# Tests and evidence boundaries

All commands ran in `/mnt/project/DriveVLA-M0-iqe-review-1e876da` with `/root/miniconda3/envs/ddp/bin/python`, unless noted.

| Check | Actual result | Complete output |
|---|---|---|
| Fixed-start historical tests rerun |85 passed,1 skipped,5 warnings,16.23s|evidence/baseline_pytest.log |
| All readiness repairs |116 passed,1 skipped,4 warnings,30.17s|evidence/final_pytest_release.log |
| Final complete suite including checkpoint monitoring/schedule |121 passed,1 skipped,4 warnings,30.32s|evidence/final_all_tests.log |
| Added explicit navtest entry regression |120 passed,1 skipped,4 warnings,36.59s|evidence/navtest_entry_regression.log |
| Final Navtest/Base targeted checks |7 passed|evidence/navtest_final_targeted.log |
| Idle-host/hold schedule tests |5 passed|evidence/navtest_schedule_test.log |
| Qualified real GPU/NCCL |1 passed,7 deselected; real image framework probe PASS|evidence/gpu_qualification.log |
| Real diagnostic workflow |expert1 → Scorer-only2 → expert3 PASS, then all-stage checksum reuse|REAL_WORKFLOW.json; REAL_STAGE_RESUME.json |
| Real optimizer-boundary resume |IDs/loss/actual gradients/parameters/Python+CPU RNG exact|REAL_EXACT_RESUME.json |
| Real saved bundle restore |PASS, restored expert0/1/3 and exact bundle identity|real_bundle_restore.log |
| Real standalone Scorer auto-resume |selected completed step2 and reused verified result|real_scorer_auto_resume.log |
| Base13k real evaluation |4 real images/GT/cache scores; model unchanged; unqualified freeze rejected|BASE_EVALUATION.json; formal_freeze_rejected.log |
| Real residual_v2 |zero init,2 actual updates, nonzero delta, Base unchanged|RESIDUAL_V2_PROBE.json |

The default skipped test requires explicit GPU authorization; it was separately run with `IQE_RUN_GPU_TESTS=1` on idle `training-vlawm-zt`, GPUs0,1. Gloo and NCCL tests check unequal valid denominators and exact optimizer-boundary continuation. The GPU framework probe uses generic pretrained Qwen initialization, so it is **not** mislabeled as trained-S0 or two-round integration. The actual trained formal checkpoint was independently evaluated on real images; the multi-round engineering run used the existing1-step diagnostic Query Base and actual frozen image features/GT/official contexts.

Reproduction:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /root/miniconda3/envs/ddp/bin/python -m pytest tests/iqe -q
# Only after a fresh idle-resource check on the authorized host:
ssh training-vlawm-zt 'cd /mnt/project/DriveVLA-M0-iqe-review-1e876da && CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 bash scripts/iqe/gpu_qualification.sh /mnt/project/iqe-review-runtime-1e876da/real_qualification.yaml /mnt/project/iqe-review-runtime-1e876da/gpu_probe'
```

Intermediate failing logs are preserved. An early resolver Path/string serialization problem was fixed; a residual test comparing different train/eval execution modes was corrected to the fixed frozen Base definition; historical-best fork behavior received a regression. No numerical tolerance was widened after observing a failure. No tensor fixture is counted as real scientific evidence. `git diff --check` and Python compilation passed. No new full expert/Scorer training or formal serving activation occurred.

Navtest evaluation was subsequently explicitly authorized by the user. Its complete population result is recorded separately; incomplete progress is never reported as full PDMS.

Full official navtest on the complete14000-step formal-training checkpoint: **12146/12146 valid scores**,136 logs, PDMS85.2969324/100, no nonfinite trajectories; all8 inference shards passed frozen parameter/buffer hashing and exited0. [Result](NAVTEST_REPORT.md).
