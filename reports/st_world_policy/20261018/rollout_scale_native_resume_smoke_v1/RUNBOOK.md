# Lead-only real native resume SMOKE

Current state: code prepared; real dispatch NOT_STARTED. Runtime751/active
formal run are protected. This experiment writes only its new SMOKE namespace.
Never pass its weights to a formal initializer.

Metadata-only commands (no model construction/GPU query/optimizer/scoring):

```bash
cd /mnt/project/VLA-Drive-st-rollout-native-resume-smoke-20261018
/root/miniconda3/envs/ddp/bin/python resources/st_rollout_resume_smoke_v1.py --help
/root/miniconda3/envs/ddp/bin/python resources/st_rollout_resume_smoke_v1.py status
CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/ddp/bin/python resources/st_rollout_resume_smoke_v1.py plan
```

On the actual authorized zt6 host, Lead can generate real software/policy/lease
metadata using the existing real UUID layout. This is an executable metadata
command; its new output path is planned, not an existing approved admission:

```bash
CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/ddp/bin/python \
  /mnt/project/VLA-Drive-st-rollout-native-resume-smoke-20261018/resources/st_rollout_resume_smoke_v1.py draft-binding \
  --output /mnt/project/st-world-policy-20261016/smoke/rollout_scale_native_resume_v1_admission \
  --resource-layout /mnt/project/VLA-Drive-st-native-resume-resources-20261018/reports/st_world_policy/20261018/rollout_scale_native_resume_resources_v1/RESOURCE_LAYOUT.json \
  --resource-layout-sha256 d5b4dbd394b9ef2563b8bffee4bc0da2336d4e74235844ed99d75ce2b97d2dfb \
  --valid-for-seconds 7200
```

The draft is WAITING_REVIEWS. It creates no SMOKE run and launches nothing. Lead
must pin independent correctness/resources reviews of its actual binding SHA
before issuing ADMISSION.json. Admission schema is
st_rollout_resume_smoke_admission_v1, statusAPPROVED, issuer, binding,
binding_sha256 and reviews.correctness/resources. Each pinned review must have
statusPASS, matching review_kind/binding_sha256, nonempty pinned
evidence_artifacts, distinct reviewer, neither Lead nor implementer. There is no
third new formal gate; original formal approval logic is unchanged.

Binding contains controller SHA, source751, exact LIMITS/root/TRAIN32 manifest,
actual environment/policy/authorization/resource-layout pins, both actual leases
and route.host/hostname/gpus/UUIDs/valid_from_unix/valid_until_unix. UUID visibility
strings in the lease exactly match those passed to the workers. The draft
metadata tool supplies these values; no fake UUID, placeholder lease or latest
model selection is accepted.

The following commands require actual independently reviewed admission path and
file SHA from Lead; no such approved file is supplied by this implementation:

```bash
: "${SMOKE_ADMISSION:?actual reviewed admission path required}"
: "${SMOKE_ADMISSION_SHA256:?actual admission file SHA256 required}"
/root/miniconda3/envs/ddp/bin/python resources/st_rollout_resume_smoke_v1.py prepare \
  --admission "$SMOKE_ADMISSION" --admission-sha256 "$SMOKE_ADMISSION_SHA256"
# Same global3600s clock across initial and resume, no automatic retry:
timeout --signal=TERM --kill-after=180s 3600s /root/miniconda3/envs/ddp/bin/python \
  resources/st_rollout_resume_smoke_v1.py initial \
  --admission "$SMOKE_ADMISSION" --admission-sha256 "$SMOKE_ADMISSION_SHA256"
# Inspect actual PAUSED4/A and PAUSED2/B, complete native2 and owned EXITs first.
timeout --signal=TERM --kill-after=180s 3600s /root/miniconda3/envs/ddp/bin/python \
  resources/st_rollout_resume_smoke_v1.py resume \
  --admission "$SMOKE_ADMISSION" --admission-sha256 "$SMOKE_ADMISSION_SHA256"
CUDA_VISIBLE_DEVICES='' timeout --signal=TERM --kill-after=60s 120s \
  /root/miniconda3/envs/ddp/bin/python resources/st_rollout_resume_smoke_v1.py compare
```

prepare creates two actual strict SMOKE registrations/configs in
/mnt/project/st-world-policy-20261016/smoke/rollout_scale_native_resume_v1/.
initial/resume use original751 train_flow_policy.run through the explicit
versioned observer worker, not a second trainer. STOP handling preserves newer
operator stops and never targets the formal run. No phase creates new retries.
Failure preserves partial native directories and receipts; do not delete them
to reset an invocation budget. compare writes COMPARISON.json with exact-replay,
replay-divergence or A/A-inconclusive verdict and precise evidence boundaries.

Local CPU fixture command (not real native/GPU evidence):

```bash
/root/miniconda3/envs/ddp/bin/python reports/st_world_policy/20261018/rollout_scale_native_resume_smoke_v1/run_cpu_checks.py
```
