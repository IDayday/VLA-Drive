# Independent road-direction correctness review

Reviewed implementation: `ec9fbb32b67532bd82058533eb89ae469c4c81d1`.
Reviewer: `/root/road_proxy_correctness_review`; this reviewer did not write the
implementation. Exact source hashes are in `REVIEW.json`.

**Geometry and preparation: VERIFIED on CPU fixtures. Real direction experiment:
NOT_RUN. Real dispatch: BLOCKED pending the thin execution runner, its independent
review, and Lead resource admission. This is not a formal training approval.**

The review reran all 30 implementation tests and added nine independent behavior
tests: **39 passed**, one installed-package deprecation warning, no skips.
Pytest wall was 8.43 seconds; wrapper wall 9.875 seconds; measured child CPU
9.331 seconds and peak rusage RSS 885,190,656 bytes. No model, real scene, metric
cache, GPU or official scoring call was used. Actual installed NAVSIM coordinate
and interpolation functions were exercised only with synthetic geometry.

The independently checked chain is the original normalized XY/sin/cos decoder,
periodic yaw, official timestamp truncation/interpolation, rear-axle transform,
four oriented corners, signed distance to the full allowed-road polygon union,
and the fixed mean over 41 times × 4 corners. Holes and disconnected components
remain present. A 2,300-segment polygon also checks the bounded nearest-segment
search across multiple chunks against Shapely's full boundary distance.

Behavior tests verify correct gradient direction inside a hole and outside the
road, decreasing cost with a small negative-gradient move, no duplicate-polygon
weighting, zero action gradient at fixed t0, geometric rigid-frame equivariance,
periodic anchored yaw derivatives, finite differences through all 32 raw
coordinates, deterministic equal-norm perturbations and isolated random state.
Boundary/tie and saturated-zero-gradient cases remain undefined rather than
manufacturing useful directions. No safety-dependent filtering changes the cost
denominator. The fake scoring test proves call charging and failure persistence
for the expected backend-return path only, not true score equivalence.

The anchored decoder is a defensible **explicit local numerical adapter**, not
an assertion that CPU float64 and saved GPU float32 outputs are identical. Its
origin is the exact stored physical request; its derivative is the original
decoder's derivative. However, the primitive intentionally accepts any supplied
finite anchor. The independent `test_anchor_arbitrary_offset_is_currently_caller_admission_boundary`
demonstrates that a 100m mismatch is not rejected by this primitive. Therefore a
real execution runner must validate the raw/physical provenance and a tight
predeclared discrepancy gate **before using this adapter**.

The implementer and Lead were given the following pre-data admission policy:

- The saved raw origin must be exactly a widening of FP32 values.
- XY discrepancy must fit the known FP32 constant-rounding terms plus half-ULP
  multiply and add roundoff, using maximum adjacent spacing and an explicit
  float64 evaluation allowance. This accommodates the actual eager decoder
  algebra; it must not be replaced by a large global relative tolerance.
- Periodic yaw discrepancy is limited to `8*float32_eps`, namely
  `9.5367431640625e-7` radians. This is a fixed numerical admission tolerance,
  **not** a proven error bound for every GPU `atan2` implementation. For the
  registered vehicle it is a micrometer-scale footprint perturbation. Actual
  differences have not been read and may fail this gate.
- Preserve offsets, reject out-of-envelope scenes and retain their denominator.
  No post-result tolerance relaxation or scene replacement.

The remaining execution review must bind exact rollout/request/cache hashes,
all 32 fixed TRAIN scenes, vehicle agreement, each scene's native interpolation
and four-corner parity, durable call limits and every failure receipt. The new
runner must not repeat a model forward or create a policy update. The current
preparation CLI alone cannot execute that experiment and does not claim it can.

All original 32 trajectories already have DAC=1. Even a successful future
experiment can only support a limited local direction or safety-preservation
observation on this fixed set. The differentiated function excludes LQR/bicycle;
executed custom margin, official DAC and PDMS remain distinct. The feature-side
pause is respected.

Reproduce this bounded CPU review:

```bash
cd /mnt/project/VLA-Drive-st-road-proxy-correctness-20261018
/root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261018/road_direction_correctness_v1/run_cpu_review.py
```

The wrapper enforces one numerical thread, hidden CUDA devices, a child CPU
limit, wall/RSS monitoring, and logs complete pytest output. Its child process is
owned by this review; no campaign process is signaled. This command overwrites
only this review's repeatable CPU test output; it does not create real results.
