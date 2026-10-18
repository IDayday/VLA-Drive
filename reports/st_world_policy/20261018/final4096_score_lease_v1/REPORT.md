# C0–C4 final4096 status and score-only execution continuation

Read-only observation: 2026-10-18 13:45–13:50 UTC. Source base of this handoff is
`98b189408fe67dc66e2315ab049be9a607a390c9`; Lead inspected at `bec3889c`.
No model/checkpoint tensors, image, metric payload or per-scene CSV was opened.
Checkpoint files were stat-checked against the small COMPLETE receipt inventory;
their payload hashes were not recomputed. No analysis/bootstrap was run.

| Arm | Training | Final native receipt | Export | Official score | Next dependency |
|---|---|---|---|---|---|
| C0 | COMPLETE4096 | Complete,9 files stat-match | Complete1696 | COMPLETE1696 | Reuse control |
| C1 | COMPLETE4096 | Complete,9 files stat-match | Not started | Not started | Fresh fixed export route |
| C2 | COMPLETE4096 | Complete,9 files stat-match | Complete1696 | Not started | Fresh score execution lease |
| C3 | COMPLETE4096 | Complete,9 files stat-match | Complete1696 | Not started | Fresh score execution lease |
| C4 | COMPLETE4096 | Complete,9 files stat-match | Not started | Not started | Fresh fixed export route |

All five RESULT/status receipts report MAX_STEPS and unchanged frozen encoder.
Those receipts establish terminal artifacts; this review does not infer live PID
state from historical process IDs. `STATUS.json` contains exact pins and tags.

C1's export checkpoint tag is `step_004096_1792295346368343264`, COMPLETE SHA256
`96fff4a62a41d727e32bc9bd0c76b07a5d3baa7cdc3964d229289fc9ad2cf22a`.
C4's tag is `step_004096_1792296601511370672`, COMPLETE SHA256
`b817e986afaafc3a132d3c856f08f341730bebb0e4d4b517eca3736719369b4b`.
They reside under their existing
`/mnt/project/st-world-policy-20261016/formal/consequence/runs/C{1,4}_fulltrain_seed42_official_4096_v1/checkpoints/`.
No C1/C4 final4096 evaluation registration or output existed at inspection.

## Completed results already present

All entries below use1696 valid dev scenes; units here are PDMS multiplied by100.
These are copied from existing COMPLETE JSON receipts, not recomputed CSV means.

| Arm | 512 diagnostic | 1024 diagnostic | 4096 common endpoint |
|---|---:|---:|---:|
| C0 | 90.989352 | 91.076647 | 91.138335 |
| C1 | 91.169425 | 91.071694 | Pending |
| C2 | 91.204624 | 91.018238 | Pending |
| C3 | 90.992712 | 91.009236 | Pending |
| C4 | 90.989579 | 91.060582 | Pending |

Existing paired1024 analyses for C2/C3/C4 are COMPLETE1696. Their exact stored
statistics and pins are in `EXISTING_RESULTS.json`; no new bootstrap or selection
was done. C0 final4096 has45 zero-score scenes, NC0.990566, DAC0.982311,
TTC0.960495 and progress0.853145. Pending final endpoints must not be replaced by
earlier diagnostic results or compared against a different-step control.

## Why C2/C3 cannot simply invoke v5 now

Their immutable routes are:

- C2: `/mnt/project/st-world-policy-20261016/formal/evaluation/registrations/C2_step004096_dev1696_seed42_v1/ROUTE.json`,
  SHA256 `2ce918f0668a0c2657d6cb4a0bb78ceaa1b699f069fca39387c38dfc49a9687c`.
- C3: same path with C3, SHA256
  `dacafaff012bb505fd760e1c02010bf7832de0fac0d5e59ce6ee8df27e8b9d01`.

They expired at08:36:01 and08:54:41 UTC. Original
`/mnt/project/st-world-policy-20261016/resources/launch_completed_dev_score_v5.py`
(SHA256 `2e62c84a2eecde3e9af585d0a6fb77f9ff33d27a2ec76cd0fa2b594b3a8f15c3`)
checks the actual current time twice: via `validate_route_v3` at lines541–545,
and again at603 immediately before durable score admission. Each requires4260s
remaining. Thus the old command is now blocked by expired execution authority,
despite its valid export. Changing old ROUTE/REGISTRATION would invalidate export
lineage and is not permitted.

## Minimal wrapper

`score_with_execution_lease.py` executes the entire pinned v5.main unchanged.
With no `--score-lease`, it calls the original v5 behavior without installing any
override. An explicit lease is supported only for these C2/C3 final4096 jobs.

With a lease, its replacement `check_route_time_v3`:

- pins the unchanged original route bytes and requires the same route object;
- binds the same job, hostname, source and environment and exact v5 helper;
- requires two workers,1728 total calls across resumes, original two score
  invocations and two shared score slots,3600s native scoring,4200s outer timeout
  plus60s cleanup, CUDA hidden and the same one-thread CPU environment;
- requires an explicit finite window of at most6h, real `now` inside the window,
  and at least4260s remaining at **both original check sites**;
- rereads immutable lease/route pins at both sites, rejects mutation, and grants
  no export, auto-retry, new host, new job or changed scoring protocol.

The wrapper does not modify the old timestamps, mock time or remove the v5
registration/native/export/runtime/resource/host checks. Original lock ownership,
two-admission accounting, current host environment, read-complete-export, no-live-
exporter check, CPU/RAM/free-space checks and exact scoring argv all execute in
the original main. It only appends the new lease and wrapper pins to fresh
ADMISSION/LAUNCH receipts through the same immutable writer; prior receipts are
never rewritten. No GPU context is authorized.

`C2_LEASE_TEMPLATE.json` and `C3_LEASE_TEMPLATE.json` have null times deliberately.
They are unissued templates and safely fail validation. Lead must issue a fresh
immutable lease on the actually authorized, original export host
`training-vlawm-tsinghua-zt-worker-0` after current resource review. An arbitrary
host migration is not part of this patch.

## Commands after Lead issues each actual lease

The actual wrapper help passes. Once the concrete lease path and its SHA are
issued, use the following exact endpoint commands on that original host; shell
variables are mandatory inputs, not fabricated receipt paths:

```bash
: "${C2_SCORE_LEASE:?set to issued immutable C2 lease path}"
: "${C2_SCORE_LEASE_SHA:?set to its verified SHA256}"
/root/miniconda3/envs/ddp/bin/python \
  /mnt/project/VLA-Drive-st-spatial20k-evidence-20261018/reports/st_world_policy/20261018/final4096_score_lease_v1/score_with_execution_lease.py \
  --job-id C2_step004096_dev1696_seed42_v1 \
  --route-sha256 2ce918f0668a0c2657d6cb4a0bb78ceaa1b699f069fca39387c38dfc49a9687c \
  --score-lease "$C2_SCORE_LEASE" --score-lease-sha256 "$C2_SCORE_LEASE_SHA"

: "${C3_SCORE_LEASE:?set to issued immutable C3 lease path}"
: "${C3_SCORE_LEASE_SHA:?set to its verified SHA256}"
/root/miniconda3/envs/ddp/bin/python \
  /mnt/project/VLA-Drive-st-spatial20k-evidence-20261018/reports/st_world_policy/20261018/final4096_score_lease_v1/score_with_execution_lease.py \
  --job-id C3_step004096_dev1696_seed42_v1 \
  --route-sha256 dacafaff012bb505fd760e1c02010bf7832de0fac0d5e59ce6ee8df27e8b9d01 \
  --score-lease "$C3_SCORE_LEASE" --score-lease-sha256 "$C3_SCORE_LEASE_SHA"
```

No `--resume` on the first score invocation: none existed at inspection. If an
actual partial score later exists, original v5 resume/identity/two-attempt checks
apply; the lease does not add another allowance.

C1/C4 should use the existing export v5 with their exact tags above and newly
issued original routes, followed by the original score v5 while those routes
remain valid. No new exporter is needed.

COMMON4096 `run_common_once.py --arm C2/C3 --bind` still lacks treatment
`official_score/IDENTITY.json` and `COMPLETE.json`; do not execute it early. After
official COMPLETE1696/0invalid/0unresolved, bind and run the existing independently
reviewed single-pair frontend. C0 is reused. No rerun of its export or score.

## Tests and current boundary

9 source/JSON-only tests passed in0.023s. They verify unchanged no-lease delegation,
finite/fresh windows, both original check sites, changed route/lease rejection,
wrong host/source/environment/budget/score scope rejection, GPU-visible rejection,
additive immutable provenance, and the unchanged native two-admission refusal.
They do not import v5's model stack or launch a child.

```bash
python reports/st_world_policy/20261018/final4096_score_lease_v1/test_score_lease.py
python reports/st_world_policy/20261018/final4096_score_lease_v1/score_with_execution_lease.py --help
```

Independent correctness review was requested. No fresh execution lease, score
job, export job, payload analysis, bootstrap or runtime change was dispatched by
this subagent. Matched-probe recovery stays unchanged and unextended.
