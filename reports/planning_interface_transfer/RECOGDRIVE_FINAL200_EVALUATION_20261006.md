# ReCogDrive final200 evaluation started (2026-10-06)

Official Stage2 completed 200 epochs at 2026-10-06 06:25:48 UTC: 65,400
trainer steps, 65,372 actual Adam updates, 16,729,600 scene exposures and
157.332601 formal-training GPU-hours. Stage1 extraction and earlier smoke/failure
costs are separately preserved. Final checkpoint SHA256 is
`ba78c77ea6db3a9eb5b8ac7f3aca111a2239b8de904273d478141f9ca5f2e998`.
Training source remains `8160b510fe2422587cd1409efb0e0a9cc0fabadd`.

Four real scenes passed original-agent inference and official metric parity,
both maximum errors exactly zero. This is a diagnostic, not Navtest performance.
The first attempt retained an ndarray/Tensor comparison error; the next safely
paused on an occupied shared host lease. Neither updated weights nor stopped
an existing evaluation. All attempts remain available outside Git.

Full run `final200_navtest_seed42_v1` is RUNNING on eight idle A800 GPUs at
training-vlawm-zt4: complete 12,146 scenes/136 logs, current-only inference.
Evaluation source `57130ca69b81b7652ec487f75a98ae518fa7f657` is immutable and
distinct from this report commit. Public Stage1 passes full weight hash and
tensor-inventory checks. Stage2 strictly loads all FP32 master tensors. The
original Stage1 computes in BF16; its full hidden interface and Stage2 compute
are FP32, with TF32 off. Original five-step DDIM, one candidate and official
postprocessing are retained. Training seed0 and scene-bound sampling seed42
are separate. No future trajectories, teachers or metric inputs enter inference.

The latest installed navtest-pdms-evaluation skill (2026-10-05) supplies unchanged
official parallel v1 scoring and an independent complete official submission
replay. Every token and factor must agree within1e-8 raw. CPU numerical
fingerprints match its existing qualification. The full PDMS is **PENDING**;
the smoke score is not a benchmark. No second training seed or five-run result
is claimed. Existing optimized-training observers and sources are untouched.

Registration/artifacts:
`/mnt/project/recogdrive-stage2-evaluation-artifacts/20261006/`.
Export has a finite three-hour per-rank guard; official CPU replay has a fixed
finite population. All partial/failed receipts are retained.

Resume only after the existing controller exits, using the same registration:

```bash
cd /mnt/project/VLA-Drive-recogdrive-evaluation-source-57130ca
/root/miniconda3/envs/navsim/bin/python -m tools.recogdrive_stage2.run_evaluation \
  --plan /mnt/project/recogdrive-stage2-evaluation-artifacts/20261006/formal_registration_v1.json
```

Do not duplicate the live controller. Models, raw data, trajectory banks and
metric caches remain outside Git; this report contains only aggregate evidence.
