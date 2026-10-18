# Host amendment and bounded diagnostic launch review

Status: **PASS_BOUNDED_FOUR_RANK_SMOKE_ONLY**. The v1 plan at commit
`12d1a468986b700140a45548668e6195067d7090` remains unchanged. This addendum
supersedes its host/GPU proposal and binds the actual4-update SMOKE registration
and launcher in REVIEW.json. PROFILE remains NOT_APPROVED. No formal training
approval is implied and this reviewer has dispatched no task.

Lead's fresh observation found other real distributed jobs starting on zt4 and
zt. Their initially small GPU allocations are not idle/pressure-job evidence.
Those jobs were protected, with no termination. The replacement target is
`training-vlawm-tsinghua-zt5-worker-0`, GPUs0–3; GPUs4–7 remain unallocated by this
trial. Policy already allows this host. The pinned fresh snapshot is actually
timestamped2026-10-18T17:31:12.644319UTC:8×A80080GB, all0MiB/0% and no reported
GPU applications; GPUs0–3 haveNV8 connectivity and NUMA0 CPU affinity. CPU0–15
are allowed. MemAvailable969.43GiB, shared project free3370.18GiB; maps exist.
Torch2.5.1cu124/DeepSpeed0.16.9/FlashAttention2.7.4.post1 match the previous
environment. Visible v1 CPU quota is unlimited; hidden ancestor limits and
filesystem quota are not certified. CPU/IO PSI are unavailable. The raw process
inventory remains outside git; sanitized pins are in HOST_AMENDMENT.json.

The scientific and execution recipe is unchanged: one
`scene_center_rollout_std_v1` candidate, world4/micro4/accumulation2/G32, G4,
MC50/chunk5, ten Euler steps, repeat8 BC, original S0 and101,592 TRAIN population,
4096 learning-rate clock and warmup100. The scope-only manifest derivatives must
retain original population/order and parent provenance. SMOKE remains4 updates
over32 fixed scenes; PROFILE is one240-update complete pipeline with two20+100
windows. This is a host move, not another topology or hyperparameter experiment.

Historical zt4 latency/memory evidence remains useful for reservation planning.
It does not certify zt5's actual speed or support a pure normalizer timing
attribution across hosts. The real4-rank SMOKE tests the new host/communication
path. The one PROFILE establishes new complete pipeline timing; an incomplete
run is reported without automatic retry or budget extension.

Actual detached runtime is
`/mnt/project/VLA-Drive-st-rollout-scale-runtime-v2-20261018`, clean at
`751fcfa2b41549ae16f7b3204aef8dd18b912016`. The older39e1819 runtime is retained
and is not used. New normalizer wall timing is now recorded per rank; it includes
dependency preparation, votes/statistics/hash/wait, not pure NCCL time.

Actual SMOKE retains4 updates,32 scenes,128 scene presentations,512 primary
scores and640 total admissions,900s trainer bound,1200s TERM /180s KILL grace,
four global CPU scorers and CPU0–15. Source/registration/launcher and monitor
hashes are exact in REVIEW.json. The existing10800s lease is an admission window;
it does not extend the single invocation's hard timeout or authorize a second
invocation. `save_every=2` saves native0/2/4 for subsequent bounded inspection.
With future PROFILE0/240, five intended directories leave one contingency in the
original six-directory/64GiB native reserve. No automatic resume is enabled.

The final launcher rejects any selected-device compute process even at0MiB/0%,
checks UUID/idle/source/allowed host/RAM/disk, holds per-GPU/dispatch locks, and
requires exact `{source_sha,registration,launcher}` binding in both independent
reviews. It rechecks after locks, rejects duplicate output and preserves the
immutable preparation budget through a separate EXECUTION_BINDING migration
receipt. New monitor hash is validated before spawn. Only the launched process
group is subject to the existing timeout; no other PID is signalled.

The five-second owned monitor sums its process subtree RSS and matching driver
GPU processes, checks output/free-space/logs, and requests only this run's
STOP_REQUESTED at an optimizer boundary. It sends no signals. RSS is a sampled
shared-page upper bound, not an instantaneous cgroup quota. It is possible to
cross a threshold between samples; outer timeout remains the bounded backstop.
For this four-step SMOKE, Lead retains explicit live supervision that all four
devices remain this job's and driver PIDs are mapped to its process tree.

Two limits explicitly prevent extending this approval to PROFILE: the current
monitor accounts output per-stage, not both stage roots combined; selected GPU
processes not mapped to the owned tree are not included in its GPU total.
Before PROFILE, a new separately pinned launcher/monitor must account the
combined96GiB/six-native budget and record unknown/foreign selected-device
processes with an owned safe-stop request. Preserve this frozen SMOKE version.
The PROFILE admission also requires an independently verified actual4-rank
SMOKE completion receipt, not just successful CPU tests or a status filename.

Independent synthetic checks against the actual `fresh_host` code reproduced the
original real-process exclusion bug (6 passed,1 failed, pre-fix SHA662664ad).
The same7 checks all pass after the narrow fix (final SHAae0c45f2,0.017s).
They cover healthy/foreign host, dirty source, low RAM/disk, busy GPU and
zero-usage real-process cases. No real subprocess/GPU/model was invoked by these
AST tests. The real metadata gate additionally exits0 with DIAGNOSTIC_ONLY,
identity `fbf2ca497ad9f1407c22c2621008adeeb349e9255735ad2d2eb01b9a9afa07b7`,
formal_launch_allowed=false and weights_usable_for_formal_init=false.

Lead's actual permitted invocation, only after fresh checks and independent
correctness admission, is the existing launcher:

```bash
/root/miniconda3/envs/ddp/bin/python \
  /mnt/project/st-world-policy-20261016/resources/launch_rollout_scale_diagnostic_v1.py \
  launch --stage SMOKE --admission "$ACTUAL_SMOKE_ADMISSION"
```

The admission path is assigned by Lead; this reviewer does not manufacture it.
No PROFILE, FORMAL, DEV/navtest, legacy rerun, model/data change or automatic
follow-up is covered by this signed Systems scope.
