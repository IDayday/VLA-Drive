# Independent bounded wiring review, 2026-10-18

Reviewer: `/root/spatial20k_correctness`. Implementation authors: Lead and the evidence/temporal implementation agents. The reviewer edited only this report directory. No dispatch, GPU work, checkpoint payload hashing, model load, large index/CSV read, scoring or bootstrap was performed.

## Conclusions

All three **code wiring** reviews pass. This does not approve an actual resource lease, assert GPU qualification, or report completed scoring/training. Original model mathematics and historical qualification evidence are inherited, not re-tested.

| Entry | Exact reviewed SHA256 | Result |
|---|---|---|
| Evidence `final4096_score_lease_v1/score_with_execution_lease.py`, author commit `5939719c` | `388a5407bf1d2f9f52e5162c3db76e43845049ce94a5a1a95e57556d9137d7d1` | PASS, 9 independently rerun CPU tests |
| Lead `spatial20k_resume_invocation_v1/launch.py` | `c4593d8e3e5444ee2b31afc2043836a5742658ba777be0ed1176030e681ce6a8` | PASS, two arms each 1 positive + 18 negative metadata checks, plus source wiring checks |
| Temporal `temporal_T0_profile4_actual_binding_v1/controller.py`, author commit `3b29e8041e8a2dcca6fc5f6c3379b385d9cc641b` | `e2e6fd89c9c18ef12a3c067a1c3366c1868acafb72e8be6b07f7980ced3790b9` | PASS, 21 independently rerun CPU tests |

## Score execution lease

The wrapper retains the pinned v5 main function. Both original time-check sites validate a fresh explicit score-only lease against the actual unchanged route object/file, host, source, environment and helper. It does not alter the original timestamps or the actual clock. The receipt seam adds provenance to immutable admission/launch records only.

Inspection of `/mnt/project/st-world-policy-20261016/resources/launch_completed_dev_score_v5.py` confirms retained checks for complete 1696-scene exports, exporter exit, policy/environment, rank/dispatch locks, two shared campaign scoring slots, maximum two durable invocation attempts, 1728 cumulative calls, two CPU scoring workers and the 4200+60 second outer bound. Its native scoring command and labels are unchanged. `--resume` retains its old explicit behavior; there is no automatic retry or export path in this wrapper.

Independent tests covered wrong route/source/host/helper/limits, nonfinite/future/short windows, pin mutation between checks, GPU-visible execution, immutable additive receipt handling and unchanged no-lease delegation. The lease must still be independently issued for the actual host. Large export completeness is checked by v5 at launch, not certified anew by this source-only review.

## Spatial same-run continuation

Small actual SP2/SP3 registration, config and original resource lease JSONs were read and hash checked. Source remains `fdf25f99b702754787904215bf1c2a2b7515ea75`, topology world4/micro4/global32, 20000-step horizon and stop, 28800-second native invocation budget. Synthetic in-memory validation admissions used expected steps 11762/11582; they were never published as approvals.

The entry checks the existing registration/source/config and run identity, refuses an operator stop, acquires per-run dispatch and four GPU FD locks, probes the native run lock, and rejects occupied GPUs. It calls the original `resolve_latest` with identity/world4, requires its selected receipt to equal the explicitly bound `BUDGET_PAUSED` receipt, then creates one exclusive invocation directory. The original resolver verifies receipt identity, exact rank inventory, bytes and every payload hash; this reviewer inspected its source but did not run it on real checkpoints. The actual argv includes `--resume`, retains native optimizer/RNG/cursor loading and the original scheduler, and passes lock FDs to the bounded 29400+180-second child.

Fresh checkpoint accounting must retain gross4143GiB = credited-on-disk + remaining, full768GiB cache growth, separate10GiB metadata and1TiB floor. A fresh accounting/admission record is still required. This implementation does not itself authorize a resource receipt or alter the scientific registration. Same-backend native four-rank restoration evidence remains inherited; no duplicate large test was performed.

## T0 actual prepared-input binding

The wrapper binds the actual successful CPU preparation/input hashes and the unchanged `38f86a15a92171e583bb5019cce89a63fb7a1d43` runtime. The inherited full-population supervisor and T3 owned-process read guard remain pinned. Recipe checks require full original101592 TRAIN population, world4/micro1/accumulation8/global32, four updates/128 exposures, original20000-step clock and warmup1000. It is PROFILE only, with no formal-init permission, no resume and no automatic retry. The config is copied without an output/recipe mutation.

The four GPU locks/UUID checks, existing-output protection, original owned-process cleanup, four-rank telemetry,192GiB sampled RSS guard,1TiB sampled logical-read guard,80GiB inclusive output hold and3600/3720/3900+180-second bounds remain. Tests cover modified recipes, synthetic invalid admissions, insufficient affinity, duplicate/busy GPUs, reservation arithmetic, missing admission and forbidden resume. Sampled guards are not advertised as hard kernel quotas.

**Remaining actual dependency:** `RESOURCE_ADMISSION_TEMPLATE.json` is explicitly `TEMPLATE_NOT_ADMITTED`, with null reviewer/UUID/affinity/time and the earlier3701166716312-byte remaining balance. It must not be dispatched. The independent resource owner must bind the newly reconciled actual balance, add this profile's80GiB hold separately, inspect the real host and issue a fresh admission. Code PASS is not evidence that world4 capacity already passed.

## Reproducible CPU commands

```bash
cd /mnt/project/VLA-Drive-st-spatial20k-evidence-20261018
python -B reports/st_world_policy/20261018/final4096_score_lease_v1/test_score_lease.py

cd /mnt/project/VLA-Drive-st-spatial20k-correctness-20261018
python -B reports/st_world_policy/20261018/bounded_actual_wiring_correctness_v1/test_sp_resume_metadata.py

cd /mnt/project/VLA-Drive-st-temporal-profile-prepare-20261018
python -B -m pytest -q -p no:cacheprovider reports/st_world_policy/20261018/temporal_T0_profile4_actual_binding_v1/test_controller.py
```

See `TEST_OUTPUT.txt` for actual rerun output. All imported entry modules are inert until `main`; no reviewed launcher/controller `main`, `common`, `prepare`, `run` or `execute` was invoked.
