# Full navtest PDMS + EPDMS on PAI-DLC

This launcher performs one full DriveVLA-M0 Base/no-memory inference pass over
all 12,146 `navtest` scenes. It evaluates those exact 8-pose trajectories with
two separate official protocols:

- NAVSIM v1.1 PDMS, aggregate row `average`;
- NAVSIM v2 one-stage EPDMS, aggregate row `average_all_frames`.

EPDMS is not a renamed PDMS result. It uses the external NAVSIM v2 devkit and
its own metric cache. Metric-cache generation is disabled; missing or partial
caches fail before model inference.

## Runtime contract

The job uses one worker and one PPU. It reuses the installed vendor runtime and
never invokes `pip`, changes PyTorch/Triton, or installs another flash-attn.
The expected installed flash-attn is
`2.8.2+v0.1.0.ppu2.1.0.oe`; the launcher verifies this exact version while the
Base config continues to set `use_flash_attn=false`.

Configure machine-local paths in ignored `env.local.sh` following
`env.local.example.sh`. The additional required values are:

```bash
export DRIVEVLA_NAVTEST_TOKEN_LIST=/shared/path/test_meta.json
export DRIVEVLA_NAVSIM_V2_ROOT=/shared/path/navsim-v2
export PDMS_METRIC_CACHE_PATH=/shared/path/metric_cache_navtest_v1_1
export EPDMS_METRIC_CACHE_PATH=/shared/path/metric_cache_navtest
export DRIVEVLA_DLC_EVAL_ROOT=/shared/path/drivevla_dual_metric_runs
```

Configure the `dlc` CLI authentication outside the repository. The submit
wrapper accepts either a dedicated `DLC_RESOURCE_ID` or a `DLC_WORKER_SPEC`;
the existing audited PPU image may be supplied through `DLC_WORKER_IMAGE` (the
current DSW image is also available as `DOCKER_IMAGE_URL`).
The wrapper defaults to the current DSW region and constructs the matching
`pai-dlc.<region>.aliyuncs.com` endpoint; both remain CLI-overridable.

## Commands

Choose one stable run ID and keep it for retries:

```bash
cd /path/to/DriveVLA-M0
source load_env.sh
export RUN_ID=base-full-navtest-20260828
```

No-write local protocol check:

```bash
bash scripts/run_base_navtest_dual_metrics_dlc.sh \
  --run-id "$RUN_ID" \
  --dry-run
```

Inspect the exact DLC submission without allocating a job:

```bash
bash scripts/submit_base_navtest_dual_metrics_dlc.sh \
  --run-id "$RUN_ID" \
  --dry-run
```

Submit a runtime/assets-only PPU preflight job:

```bash
bash scripts/submit_base_navtest_dual_metrics_dlc.sh \
  --run-id "${RUN_ID}-preflight" \
  --preflight-only
```

Submit the formal complete evaluation:

```bash
bash scripts/submit_base_navtest_dual_metrics_dlc.sh \
  --run-id "$RUN_ID"
```

Retry after an interruption, reusing only verified complete artifacts:

```bash
bash scripts/submit_base_navtest_dual_metrics_dlc.sh \
  --run-id "$RUN_ID" \
  --resume
```

If an invalid artifact must be replaced, `--overwrite` first moves the exact
run directory into a sibling `.superseded/` archive. It does not recursively
delete shared results.

## Outputs and acceptance

The run is complete only when both official CSVs contain exactly 12,146 unique
scene tokens, zero failed rows, a finite aggregate, and the expected aggregate
row name. Stable outputs are:

```text
<output-root>/
├── protocol.json
├── predictions/
│   ├── submission.pkl
│   └── test/
│       ├── <token>.npy
│       └── inference_manifest.rank0.json
├── results/
│   ├── pdms.csv
│   └── epdms.csv
├── summary.csv
├── summary.json
├── logs/
└── work/
```

Read the final scores with:

```bash
python - <<'PY'
import json, os
from pathlib import Path

run = Path(os.environ["DRIVEVLA_DLC_EVAL_ROOT"]) / os.environ["RUN_ID"]
result = json.loads((run / "summary.json").read_text())
print("PDMS", result["PDMS"]["score"])
print("EPDMS", result["EPDMS"]["score"])
print("scenes", result["scenarios"])
PY
```
