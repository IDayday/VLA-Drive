C0-4096 is checkpoint-ready and waits for a fresh allowed GPU and immutable v3 route. The actual training RESULT is COMPLETE / MAX_STEPS at 4096; native tag is step_004096_1792214769814548316. This is the predeclared common endpoint, not a checkpoint selection. No new dev evaluation has run.

READY.json binds the original fixed intent, actual training registration/approval/launch/identity, final RESULT and canonical native COMPLETE receipt. Receipt SHA256 is c8655892cff8292b60325b405767294c04fb1445f952a2fd858c084188e818f0. All nine native filenames/sizes match, totaling 11,473,306,120 bytes. Payload checksums were not recomputed; the existing frozen exporter verifies them before use.

The original dev manifest and metric index hashes match their pins, with exactly the same 1696 token/log pairs across 16 logs. Source 6714b1e9fb2e0c8059b90b8b4d48cd40796fe345 is clean and its policy-bound source files match. The deployed v3 helpers match the reviewed bytes. At the observation there was no evaluation route, registration, export, score or admitted invocation for this job.

Lead must create the actual ROUTE.json at /mnt/project/st-world-policy-20261016/formal/evaluation/registrations/C0_step004096_dev1696_seed42_v1/ROUTE.json using the existing [v3 route schema](/mnt/project/st-world-policy-20261016/audits/evaluation/C3_ROUTE_HELPER_v1/ROUTE_SCHEMA.md). The route binds this exact receipt, fixed policy, runtime identity, pinned environment and one actual authorized host/GPU UUID. Allowed canonical hosts are training-rl-zt2, training-vla-zt and training-vlawm-tsinghua-zt. Export/resume/score retain that host and route hash. This preparation makes no fresh-idle claim; occupied training GPUs must not be shared or preempted.

After actual route and fresh admission, Lead uses the reviewed helper, with the variables populated from that route:
```sh
: "${GPU_INDEX:?Set from the actual reviewed ROUTE.json}"
: "${ROUTE_SHA256:?Set from the actual immutable ROUTE.json}"
/root/miniconda3/envs/ddp/bin/python /mnt/project/st-world-policy-20261016/resources/launch_formal_dev_export_v3.py --job-id C0_step004096_dev1696_seed42_v1 --checkpoint-tag step_004096_1792214769814548316 --gpu "$GPU_INDEX" --route-sha256 "$ROUTE_SHA256"
```

Only after complete1696 export, successful export RESULT and exporter exit, on the same host:
```sh
: "${ROUTE_SHA256:?Use the same immutable export route}"
/root/miniconda3/envs/ddp/bin/python /mnt/project/st-world-policy-20261016/resources/launch_completed_dev_score_v3.py --job-id C0_step004096_dev1696_seed42_v1 --route-sha256 "$ROUTE_SHA256"
```

COMMANDS.json retains exact argv and source-command provenance. The helper performs registration/admission; this queue is not an execution authorization. No direct child-CLI bypass, latest fallback, route migration or automatic retry is introduced.

World1/rank0, seed42, FP32 master/TF32-off/SDPA, one Euler-10 trajectory, 1696 scenes and the common scoring campaign ID remain fixed. The scoring cap is 1728 total admissions with two workers and at most two invocations; unknown admissions remain charged. Campaign concurrency remains four exports and two scorers. The per-job output reservation is 1 GiB; existing independent cache/checkpoint/evaluation disk reservations remain enforced. Route admission needs at least 7860 seconds for export or 4260 seconds for scoring. The route should cover the intended export/score lifecycle and any allowed retry; its identity is not renewed silently.

No scientific result or treatment gain is claimed. The 512/1024 diagnostics and all prior C0 results remain intact. No GPU, scoring, model load, navtest or new training was performed. The ready audit contains only small metadata checks and file stats. C4's separate running training is outside this preparation.

Owned report directory: reports/st_world_policy/20261016/evaluation/C0_4096_ready_v1. Parent commit: 6b04a54f662633cc1ddfc309921a13cc8c0eb5c9.
