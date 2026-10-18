# CPU preparation and review

Working directory:
`/mnt/project/VLA-Drive-st-consequence-road-proxy-20261018`.

```bash
cd /mnt/project/VLA-Drive-st-consequence-road-proxy-20261018
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/miniconda3/envs/ddp/bin/python tools/st_world_policy/diagnose_road_direction.py --help
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  timeout 600 /root/miniconda3/envs/ddp/bin/python -m pytest -q tests/st_world/test_road_margin_proxy.py
```

For another preparation, supply a **new** small output path; never delete or
replace the existing immutable `PREPARATION.json`:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/miniconda3/envs/ddp/bin/python tools/st_world_policy/diagnose_road_direction.py \
  --mode prepare --output /tmp/road_direction_review_preparation_20261018.json
```

This returns metadata only, not a completed experiment or scoring permission.
There is deliberately no CLI real-run option until a separate Lead admission and
independent review bind the exact invocation/assets/resources. The admitted thin
runner can reuse `CommandRoadCost`, `direction_variants`, and
`score_diagnostic_variants`; it must provide the durable global call-budget and
result persistence callbacks. Do not invoke these against real assets under the
previous feasibility-audit read allowance.
