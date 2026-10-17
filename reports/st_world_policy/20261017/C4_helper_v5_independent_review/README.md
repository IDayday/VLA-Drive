# Independent C4 fixed-dev helper v5 review

**PASS for the narrow additive evaluation-control change.** Reviewed candidate:
`0fed55bd7d1c39121117a49af4ec845b3950b8a3`, exact clean worktree
`/mnt/project/VLA-Drive-st-c4-dev-preparation-20261017`.
Reviewer: `/root/st_consequence`, not the author of this helper/preparation.
This is not another algorithm/training review or permission to bypass Lead
resource admission. No production source was edited during this review.

## Scope and findings

The change adds C4's actual registration/approval/launch/runtime binding and
only its already registered512/1024/4096 full1696-scene,16-log dev endpoints.
It does not add a reward, arm, sampling rule, checkpoint-selection rule,
trajectory candidate, control reevaluation, Navtest access or automatic retry.

All preparation and helper manifest files match their exact Git blobs. The
helper's fixed five-file C4 pin set matches the actual preparation. Actual C4
registration, config, approval fingerprint, initial launch, three existing
training-review receipts and source6714 identity match runtime
`4e023061b8ac16114cc1f9e3b5e440cb2c3a23d524d562aa1276270567f5fbbb`.

The v4-to-v5 diff changes only fixed C4 pin declarations and the existing
additive loader/static-binding cases. All unrelated function bodies, complete
export admission body, score main, routing, lock, retry and budget paths are
unchanged. Existing sixteen job metadata results match v4, including W's
fdf25f99 training /6714 evaluation distinction. Old source and preparations
are preserved.

Reviewed invariants remain:

- Only fixed512/1024/4096, complete1696 dev, world1,seed42,Euler10 and original
  FP32-master evaluation. Matched C0 controls remain at the same step.
- Original exact host/GPU UUID/environment route and current dispatch policy;
  no extension from C4's training-host placement.
- Four shared export slots and two shared score slots, two CPU workers/score.
- 1728 total score admissions across bounded retry, at most two export and two
  score invocations; failed/unknown durable admissions consume budget.
- Native COMPLETE self-hash, exact identity/tag/step/world/inventory; exporter
  identity/model readers still perform the full payload hash check before use.
- Complete rank0 predictions and attempt RESULT for all1696 scenes, matching
  identity/route, zero failed export scenes and no surviving exporter before
  scoring; the rank lock is held through scoring.
- Original finite timeouts and route lifetime checks; no new background
  controller, automatic resume, training, ACTIVE update or dispatch here.

## Actual checks

1. Re-executed the owner's unmodified AST-only validator into an independent
   output: **153/153 checks passed**,2.27seconds. It never calls launcher main.
2. Added seven independent checks: exact manifests/Git blobs, actual training
   bindings, actual512 receipt/inventory/stat, guard positive case and two new
   native-boundary negative cases. **All passed**,0.80seconds.
3. Wrong native receipt SHA is rejected even after recomputing the surrounding
   checkpoint metadata hash.
4. A re-sealed513-step native/checkpoint metadata fixture is rejected against
   the fixed512 intent. This is distinct from the owner's existing raw-intent
   wrong-step test. Fixtures stay in memory and never create runtime evidence.

Real native endpoint inspected:

- Tag: `step_000512_1792223517823880689`, completed512,world4.
- COMPLETE SHA256: `9c63e6296ea92085f8848546e5c5a049e68e4059aa038f10170e9d64f09ecd36`.
- receipt_identity: `e458fc917bf18743e421d4dfc3d5ad3091a04d0f99c1b3bd71f4b7d65ea63525`.
- Entire declared native file inventory exists with exact recorded byte sizes.
  No model/optimizer payload was opened or rehashed, and no model was loaded.

Only small CPU metadata was read. No GPU, scoring, dispatch, SSH, new Navtest,
native conversion or mutation of author/runtime outputs occurred. These tests
validate control paths; they are not a real C4 dev result or live free-slot check.

## Exact candidate bytes for Lead

Export helper SHA256:
`ad0ee24987bdba9ee3fbc46a5efbdb1445b0a6a69d4d588fc32e960738b876bf`.
Score helper SHA256:
`2e62c84a2eecde3e9af585d0a6fb77f9ff33d27a2ec76cd0fa2b594b3a8f15c3`.
Policy SHA256:
`ed465e3bedc8974c0bc43edd64ed73f55090010c314cccce8b9c9718c3e2e1b2`.

Lead can integrate/deploy these exact reviewed bytes under the existing
protocol, then separately bind the actual endpoint and a fresh authorized
route. A complete512 checkpoint does not bypass an occupied export slot.
No future1024/4096 native receipt is guessed or approved by this report.

## Reproduction

Use a **new** result path because the owner validator intentionally uses
exclusive creation:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  timeout 120 /root/miniconda3/envs/ddp/bin/python \
  /mnt/project/VLA-Drive-st-c4-dev-preparation-20261017/reports/st_world_policy/20261016/evaluation/C4_HELPER_v5/VALIDATE_CANDIDATES.py \
  --output /tmp/C4_helper_v5_independent_checks_20261017_new.json

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  timeout 120 /root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261017/C4_helper_v5_independent_review/review_native_boundary.py
```

Evidence is saved in `OWNER_CHECKS_REPRODUCED.json`,
`NATIVE_BOUNDARY_REVIEW.json` and their complete stdout/stderr logs.
