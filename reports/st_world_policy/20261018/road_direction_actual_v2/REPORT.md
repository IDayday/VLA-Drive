# Real road-direction diagnostic: incomplete validity, no gain claim

The registered 32 TRAIN scenes were actually processed on
`training-vla-zt-worker-0` using the original NAVSIM reference engine. The
experiment retained its fixed scene selection, perturbations and numerical
tolerances. It performed no model update and initialized no CUDA device.

| Outcome | Scenes / requested 32 |
|---|---:|
| Complete original/negative/positive/random comparison | 4 |
| Zero/nonfinite-gradient guard | 12 |
| Moving-corner boundary/nearest-segment tie | 7 |
| Official geometry parity failure | 9 |

All original 32 candidates had DAC=1. The road-violation repair denominator is
therefore zero. The 28 unsuccessful directions are retained, not replaced or
counted as valid zero scores. The CLI exits 2 (`PARTIAL_OR_UNDEFINED`).

On the **four common valid scenes only**, negative-gradient motion reduced the
custom executed road cost relative to positive motion by 0.00004450 m and to
fixed random motion by 0.00001301 m. However, its official PDMS change versus
the unchanged original was **−0.005545 percentage points**, from a valid-subset
original mean of 85.58408. NC/DAC/TTC did not change in that subset. This is not
a full-population score, repair result, policy-learning result or safety proof.

The result does not qualify this proxy for policy supervision. First diagnose
the real geometry mismatch while preserving the 1e−9 tolerance and original
failure evidence. Also distinguish a genuinely undefined geometric derivative
from an overly conservative tie guard, and report weak gradients on already
safe commands. No new scene selection or reward/threshold change has been
executed on the basis of these scores.

## Actual cost and state

- End-to-end supervisor wall: 42.8084 s; worker wall: 38.6775 s.
- Parent + worker + tracker observed peak RSS: 1,516,310,528 bytes.
- Verified logical asset reads: 26,113,939 bytes.
- New official calls: 12; original valid scores were reused.
- Worker exited normally; owned-worker exit confirmed; no cleanup error.
- All full per-scene errors, predictions, variants and score records remain in
  the runtime output named in `RESULTS.json`. They are not copied into Git.

Actual command (runtime frozen at `83679937fe9702be39f0184154fdf9ebdc4eb98a`):

```bash
cd /mnt/project/VLA-Drive-st-consequence-diagnostics-runtime-v2-20261018
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
/root/miniconda3/envs/ddp/bin/python tools/st_world_policy/run_road_direction.py \
  --execute \
  --registration /mnt/project/st-world-policy-20261016/audits/consequence/road_direction_fixed32_v2_controls/REGISTRATION.json \
  --admission /mnt/project/st-world-policy-20261016/audits/consequence/road_direction_fixed32_v2_controls/ADMISSION.json \
  --admission-sha256 f35086651ca4147a6d3da698bbe7a9ab357b788a9c7fb6f361f4d524226f0bdf
```

This consumed admission is an execution record, not a rerun command. Existing
attempts and failures must remain immutable. No automatic retry was launched.
