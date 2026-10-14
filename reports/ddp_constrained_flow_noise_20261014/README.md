# DDP constrained Flow-Noise: execution record

Source start: `6edb05d576b40a9718421e86c06d560f184a11c6`.
Branch: `feature/ddp-learned-noise-constrained-grpo-20261014`.

**User-requested interim Navtest, 2026-10-14:** both methods at matched
raw25,398 / update2030 have now completed all 12,146 scenes with original
single-candidate 10-step ODE, seed2101 and canonical v1.1. PDMS is
SFT **88.9116**, C_SDE **89.2592** (**+0.3476**), N_FLOW_NOISE **89.2429**
(**+0.3313**). Both paired log-bootstrap intervals are positive, but NC/TTC
violations increase: NC **231 / 251 / 260**, TTC **601 / 640 / 689**.
DAC and the total zero count improve. Neither method meets safety preservation;
learned noise has not shown additional deployed score benefit at this matched
checkpoint. This is **12.5% of the formal budget**, not a completed training
endpoint or a 92-PDMS result. The explicit interim Navtest observation is
disclosed and does not change training or checkpoint selection.
[Full test result, safety transfers and execution evidence](interim_navtest_raw025398.md).

Earlier execution update at **2026-10-14 18:49 UTC**: **both formal lines are running**.
C resumed its complete recovered update1850 state after a targeted numerical
repair passed real-density/full-head-gradient and eight-rank inner-resume
qualification. It has now completed update1888 / raw23648; no restart from SFT
was needed. N is at update3212 / raw40086. Current complete-collection throughput
is N **2.52 raw scenes/s** (about 2.37x its earlier window) and C **1.45/s**
(about 1.97x). Remaining pure training is approximately **18h / 34.4h**;
including saves and evaluation, provision **20–24h / 36–40h** under current
throughput. C's estimate uses only its first 16 steady-state collections and
will need a longer observation. These are observed windows, not paired speed
guarantees or driving-performance results.
[Measured speed, C recovery, costs and actual commands](progress_speed_recovery_20261014.md).

Earlier execution update at 2026-10-14 18:12 UTC: existing batch-PDMS code has been
adapted and deployed to N_FLOW_NOISE from its complete native update2722 /
raw33958 checkpoint. All eight ranks have continued through update2774 /
raw34606. Batch simulation retains independent canonical reference/candidate
aggregation, including EP normalization. Equality passed for 512 real policy
candidates under different batch splits, order and distractors; both variants'
eight-rank learning/optimizer/RNG/ODE exports and an old-to-new execution
interruption recovery also passed. Warm CPU scoring improved 2.68x. The first
16-collection production windows show only 1.050 to 1.070 raw scenes/s (1.019x),
so the scoring microbenchmark is not claimed as whole-training acceleration.
An earlier execution change initially showed 2.18x throughput, but update time
subsequently rose from about 2 to 5 seconds before the batch adapter started;
its cause remains unresolved. C's 826-step exact recovery is complete; its
reproduced probability-layout fault remains paused, not repaired by these
performance changes. Neither formal endpoint is complete.
[Performance evidence, source reuse and commands](performance/README.md).

Status correction at 2026-10-14 16:43 UTC: **neither formal run is complete**.
N_FLOW_NOISE is live at 2,294 updates / 28,638 raw exposures (14.095% of its
two-pass budget). C_SDE exited at 16:09 UTC after 1,850 updates because the
fresh-chain probability consistency check failed. The previous statement that
both runs continued was based on stale training logs and was incorrect.
An isolated eight-GPU recovery is replaying the original saved chains from the
last complete checkpoint at update1,024; all ranks have reproduced the first
92 updates exactly in the recorded numerical fields. Recovery is not yet
complete, and no probability threshold has been relaxed. All 32 census worker
processes are live; the committed population snapshot now covers **28,601 /
101,592** scenes. [Current results, failure and recovery evidence](progress_and_conclusions_20261014T1643Z.md).

The earlier verified frozen-SFT census snapshot covers **17,455 / 101,592**
TRAIN scenes: all-full groups **33.824%**, all-zero **0.052%**, other ties
**0.046%**. Final zero-advantage groups are **22.698%** under fixed initial
multipliers; nearly all are saturated full-score groups. Of non-tied PDMS
groups, **86.70%** have only EP varying among the official score components.
This is incomplete population coverage, not a final full-TRAIN estimate or a
trained-policy deployment score. [Verified snapshot and failure causes](train_population_snapshot_20261014T1615Z.md).

Both methods have reached their first registered raw25,398 full-DEV observation:
ODE PDMS is C **94.2246**, N **94.5460**, versus SFT **93.6192**.
TTC violations are **4 / 4 / 1**. Both fail safety selection, without triggering
the registered severe-pause boundary. N's better DEV score does not carry over
as an advantage over C on the matched interim Navtest above.
[C full-DEV evidence](C_SDE_DEV_raw025398.json), [N full-DEV evidence](N_FLOW_NOISE_DEV_raw025398.json).

Earlier observation at 2026-10-14 15:35 UTC: the
user-requested extra full-DEV observation of both saved update1024 checkpoints
is complete: 1,696 scenes / 16 logs, original single-candidate 10-step ODE,
seed2101, canonical v1.1. PDMS is SFT **93.6192**, C_SDE **94.4840**,
N_FLOW_NOISE **94.4822**. TTC violations are **1 / 4 / 10**, respectively;
neither trained checkpoint meets the safety eligibility rule. Learned noise
has not demonstrated an extra ODE score gain at this observation. These extra
observations stay out of preregistered checkpoint selection, while the TTC
evidence is retained for the existing consecutive-degradation pause rule.
[Analysis, efficiency findings and next action](interim_update1024_analysis.md),
[C_SDE evidence](interim_update1024_C_SDE_ODE.json),
[N_FLOW_NOISE evidence](interim_update1024_N_FLOW_NOISE_ODE.json),
[paired comparison](interim_update1024_N_minus_C.json),
[observation registration](interim_update1024_registration.json).
Core and real-learning acceptance passed; all 101,592 TRAIN condition and
canonical metric assets are sealed. [Run-start snapshot](formal_start.json).

Read-only analysis of the first 512 saved behavior collections per method
confirms a substantial efficiency problem: only 64.19% / 64.28% of raw groups
enter PG. Zero final advantages account for 25.71% / 25.14%; effective refill
surplus accounts for another 10.10% / 10.58%. Across both methods, 145,952 of
408,064 candidate scores did not enter PG. Some zero groups still contribute
to first16 reference retention and unbiased-by-filter dual statistics, so this
is a PG-utilization count, not a claim that all associated work had no use.
Over 99.7% of tied-PDMS groups have all16 candidates at score1; all-failure
groups are only 3 / 4. [Signal and measured cost breakdown](interim_update1024_signal_efficiency.json).

The earlier 14:48 UTC live training snapshot covers 1,572 / 1,782 all-rank
completed updates and 19,600 / 22,272 raw scene exposures. The unequal prefixes
are progress evidence, not a matched method-effect comparison.
[Snapshot](formal_progress_20261014T1448Z.json),
[training curves](learning_curves_20261014T1448Z.png).

The user additionally requested the entire TRAIN population, not an extrapolation
from the first training prefix. A separate **read-only frozen-SFT C_SDE G16
census** of all 101,592 TRAIN scenes is now running on 32 qualified GPU workers.
It performs no updates and does not replace either formal line or ODE evaluation.
The three explicitly authorized temporary hosts expire at **2026-10-14
23:07:18 UTC**, with native record-boundary stopping from 22:47:18. Permanent
host bans outside this task-scoped lease remain unchanged. This full census is
not yet complete; its upfront 1,625,472-candidate budget is not free training
acceleration. [Execution/expiry evidence](full_TRAIN_census_start.json).

The only formal comparisons are legacy correlated, score-corrected `C_SDE`
and diagonal learned `N_FLOW_NOISE`. Both retain the complete 359-tensor DDP
action model (819,503,620 parameters), G16/K10, training seed 42 and original
single-candidate ODE evaluation seed 2101. The new line additionally trains the
RLinf noise MLP. Fixed SFT weight SHA-256:
`9a26685aa3838a2e1b89ab2d92997664afde4b259fed6683851f42781ad16eb4`.

The old all-BF16 condition cache is excluded. The replacement uses the original
VLAAgent execution: BF16 Qwen, FP32 history/action prefix, canonical processor,
prompt and action-token extraction, before trainable `qwen_proj`. It contains
only token identity and observation features. Qualification on real TRAIN
scenes passed exact cache/direct ODE comparisons at batch 1/2 and candidate
chunk 1/16, including the original inference call. Identity:
`80a2dd4b417a3154cbbf575044daea586d945b54e014eee22269d7bd48d39e0b`.
An earlier qualification attempt failed because the launch environment lacked
the deterministic CuBLAS setting; its evidence is preserved. Full TRAIN cache
generation and missing canonical v1.1 metric-cache production are complete.

The 34 targeted core/runtime/diagnostic tests passed on the authorized Tsinghua zt4 host. They cover
upstream outputs/gradients, time/sign conversion, densities, KL and sigma
gradients, score-function arithmetic, component advantages, dual direction,
collector ordering/caps/resumption and geometric costs. This is arithmetic
acceptance, not evidence of driving improvement. Real full-head learning and
inner-epoch recovery have now passed the bounded qualification below. The two
formal curves and their ODE outcome comparison remain unfinished.

Commands already executed from this worktree:

```bash
python -m scripts.analysis.constrained_flow_noise.lock_upstream --output /mnt/project/constrained-flow-noise-20261014/upstream --lock configs/flow_rl/upstream_lock.json
python -m pytest tests/constrained_flow_noise/test_core.py -q
python -m torch.distributed.run --standalone --nproc_per_node=8 -m scripts.analysis.constrained_flow_noise.build_condition_cache --population /mnt/project/flow-safe-improvement-20261013/population/manifest.json --output /mnt/project/constrained-flow-noise-20261014/condition_cache --priority-tokens /mnt/project/constrained-flow-noise-20261014/cache_priority.json
python -m scripts.analysis.constrained_flow_noise.build_training_metric_cache --population /mnt/project/flow-safe-improvement-20261013/population/manifest.json --cache-root /mnt/project/constrained-flow-noise-20261014/metric_cache_v1 --reuse-root /mnt/project/flow-safe-improvement-20261013/metric_cache_v1 --output /mnt/project/constrained-flow-noise-20261014/metric_cache_build --workers 16
```

Python environment: `/mnt/project/flow-safe-improvement-20261013/venv/bin/python`.
Launches set `CUBLAS_WORKSPACE_CONFIG=:4096:8`, CPU BLAS/OMP threads to one,
and validate the current dispatch policy. Private run records and failure
evidence are under `/mnt/project/constrained-flow-noise-20261014`; they are not
uploaded. No original data, features, weights or private scene images are committed.

Upstream boundaries: [patch inventory](../../starVLA/rl/constrained_flow_noise/vendor/PATCHES.md).
Formal calibration, cost budget, optimizer/resume and outcome sections will be
updated with measured evidence; a target PDMS above 92 is not assumed achieved.

The finite calibration is complete: 128 predeclared TRAIN scenes, G16 each,
one C_SDE reference and at most two noise ranges. The initial discrete-std range
`[0.0105409, 0.1]` was rejected because its median requested-XY second difference
was 2.80 times C_SDE (registered limit 2). The only allowed half-range passed
and is locked to `[0.0052704627669472965, 0.05000000000000001]`.

| Initial stochastic policy | PDMS (0–100) | Joint NC/DAC/TTC full pass | Pairwise ADE (m) | Median XY second difference (m) |
| --- | ---: | ---: | ---: | ---: |
| C_SDE | 91.5867 | 96.1426% | 0.1322 | 0.2111 |
| N_FLOW_NOISE original range, rejected | 87.7386 | 91.1621% | 0.3396 | 0.5905 |
| N_FLOW_NOISE selected half-range | 91.5540 | 96.0449% | 0.1797 | 0.3514 |

These are calibration means over actual stochastic candidates, **not trained
ODE deployment results**. More geometric diversity is not yet evidence of
useful driving exploration. No DEV/Navtest data selected the range.
The fixed road/object thresholds are 0.1720945/0.9274561 metres, the registered
positive-margin 5th percentiles among C_SDE candidates fully passing all three
official safety metrics. Common unfiltered cost budgets are
`[0, 0.03271484375, 0.005859375, 0.02867518074, 0.01117817716]`.
The first three are event rates; the latter two are bounded geometric costs.
A zero empirical NC budget does not establish zero true collision probability.

The eight-GPU storage probe confirmed that existing DeepSpeed converts a
separate FP32 noise group to BF16. The implementation therefore retains the
audited uniform BF16 storage and FP32 optimizer masters, and evaluates the
noise module through differentiable FP32 parameter views and FP32 fixed bounds.
Actual full-DDP kernel checks passed: legacy C_SDE chain unchanged; eager versus
CUDA graph chain exactly equal; noise-disabled means exactly equal to original
Euler; sigma-only gradients reached all six MLP tensors and shared DiT; both
sigma-only and mean-only changes altered ratios. Flat versus serial initial
mean-ratio errors were at most `1.02e-5` (acceptance `1e-4`), with no PPO update
performed in that kernel test. A first graph attempt failed on constructing a
constant during capture; fixed FP32 bound tensors are now prepared before capture.
Only configuration constants are cached, never noise outputs or trainable hidden states.

Both real eight-GPU learning qualifications completed 32 raw TRAIN scenes,
512 newly scored trajectories, two behavior collections and four Adam updates.
They used the complete original action head, genuine canonical feedback,
both inner epochs, independent reference KL, GT replay and the five dual
optimizers. C_SDE had finite nonzero full gradients for all 359 tensors;
N_FLOW_NOISE for all 365 tensors, including the six noise tensors. Actual FP32
masters changed in every optimizer group. At update 2, 335/359 and 338/365 BF16
shadow tensors changed, respectively; small FP32 updates below a BF16 rounding
unit must not be called missing optimizer updates or claimed to change every
forward tensor. [Aggregate evidence](acceptance_learning.json).

Continuous versus inner-interrupted N_FLOW_NOISE finished with bitwise-equal
actor/noise weights, all eight FP32 optimizer partitions, every rank's RNG,
dual optimizer, stream/inner cursors, KL controller and original ODE export.
A separate interruption after the first16 raw groups passed the same complete
state comparison, including every saved chain, mean/std/logprob, advantage,
lambda snapshot, canonical metric and simulated consequence (only execution
timings excluded). Reconstruction using DeepSpeed's public FP32 checkpoint
utility additionally confirmed that **all 365 FP32 master tensors actually
changed**, including every original action tensor and every noise tensor.
The exported qualification model also ran the original VLAAgent from real
current observations: no noise head, teacher, training cache or scorer is
required. All frozen upstream tensors retained their exact original hash.
These qualification weights will not initialize either formal line.

The matched original-observation SFT DEV baseline is complete for all 1,696
scenes / 16 logs, one output each at scene-bound seed 2101:
PDMS **93.6192**, NC 100.0000, DAC 99.2335, TTC 99.9410, EP 85.8065,
Comfort 99.8821, DDC 98.8797 (all on a 0–100 reporting scale); 13 zero-score
scenes. [Full paired-analysis contract](SFT_DEV_original_seed2101.json).
This is **untrained SFT on DEV**, not evidence that RL improved or achieved
the Navtest research target. DEV is unseen by RL updates; SFT may have seen it.
The log bootstrap does not include training- or inference-seed uncertainty.

The four-update N_FLOW_NOISE qualification export also completed all 1,696
original-observation ODE DEV evaluations. PDMS changed by only **+0.000342
points**, with the same 13 zeros and unchanged safety event counts. This is
an end-to-end export/evaluation plumbing check, not a meaningful learning gain
or a formal method comparison. [Qualification-only results](qualification_ODE_seed2101.json).

Full-model cost profiling completed for each variant without the expensive
all-tensor qualification probes: 128 raw scenes each, 10 C_SDE versus 12
N_FLOW_NOISE updates because their valid-group boundaries differ. Measured
update time was 56.34/57.06 seconds; actual profile wall time including its
end checkpoint/export was 280.92/289.18 seconds. These measurements occurred
under shared GPU/cache load. Extrapolating component times with ideal rank
balance gives 85–86 hours per two-pass line; extrapolating the complete measured
wall interval gives 124–128 hours. These are conditional cost projections,
not promised completion times or a claimed algorithmic speed-up.

The [common budget](formal_budget.json) is now locked before formal outcomes:
203,184 unfiltered scene exposures and 3,250,944 freshly scored trajectories
**per variant**, with shared target 16/cap 32 group collection and two inner
epochs. The requested finite budget is retained; no resource expiry or GPU-hour
cap was supplied. All four DEV milestones, TRAIN monitoring and one-time cache
costs are accounted separately. Recovery points are saved every 1,024 updates
plus milestone/interruption boundaries and all are retained. Fixed DEV safety pauses use the original
precalibration rule: a single +2 percentage-point event-rate increase, or
two successive +0.5-point increases with positive lower paired 95% bounds.
Additionally, sustained numerical exploration saturation is defined before
formal training as at least 95% of sigma values within 1% of one log-std bound
for 20 consecutive collections. A trigger preserves the chain and checkpoint
and pauses that variant; it never silently changes bounds or safety budgets.

Before formal training, inspection of the actual SFT configuration identified
one replay-mode correction: the original DiT uses dropout 0.2. The first bounded
qualifications and profiles had retained the former RL trainer's eval-mode
replay. Formal GT replay now uses original SFT train mode, with independent
FM and dropout RNG and exact restoration of the PG eval modes. This applies
equally to both lines; no reward, sigma range or raw budget changed.
[Recorded addendum](formal_replay_addendum.json).
Both corrected eight-GPU real-learning checks completed four updates, with
all 359/365 gradients finite and nonzero. Corrected-mode interruption recovery
also passed exact actor/noise, eight FP32 optimizer partitions, RNG, dual,
cursors, saved chains and original ODE export comparisons. The corrected
export passed real-observation inference with noise head absent and all upstream
weights unchanged. [Corrected recovery evidence](corrected_resume_acceptance.json).
Earlier numerical and evaluation evidence is retained with
its original execution identity, rather than silently relabelled.
[Corrected replay checks](replay_mode_learning_v3.json).

The saved TRAIN calibration was also audited in physical units, with the
original scalar XY normalizers and the bicycle's body-frame velocity convention.
Median first-half-second requested mean-velocity discrepancy was 0.356 m/s
for C_SDE, 0.474 for the rejected range, and 0.373 for the selected range;
all simulated t0 states were identical within each scene. This does not claim
continuous request derivatives. Two read-only analysis mistakes (a scalar
normalizer reshape and an extra rotation of already body-frame velocity) were
corrected with superseded evidence retained; neither affected scoring,
calibration selection or learning. [Physical calibration audit](calibration_physics.json).

At 2026-10-14 06:20 UTC, all required acceptance receipts are PASS. The C_SDE
launcher is assigned to `training-vlawm-tsinghua-zt3`, and N_FLOW_NOISE to
`training-vlawm-tsinghua-zt4`. They wait for full TRAIN cache publication and
fresh eight-GPU memory checks, without preempting unrelated owners. Metric
production has sealed 100,892/101,592 TRAIN scenes; the final log is still
being computed. The corrected condition population has finished production.
Neither a queued launcher nor the four-update qualification is a formal result.

Formal training source is frozen at commit
`8275f6b06878fba3f082d2ac213ca70557c21753`, with 83 exact source/config hashes
in the private `formal_sources.json`. Later report/evaluation adapters do not
change those training files. These two commands have actually been launched
on their respective hosts:

```bash
python -m scripts.analysis.constrained_flow_noise.launch_formal --variant C_SDE --run-root /mnt/project/constrained-flow-noise-20261014 --population /mnt/project/flow-safe-improvement-20261013/population/manifest.json --output /mnt/project/constrained-flow-noise-20261014/formal_C_SDE_seed42_v1 --source-lock /mnt/project/constrained-flow-noise-20261014/formal_sources.json
python -m scripts.analysis.constrained_flow_noise.launch_formal --variant N_FLOW_NOISE --run-root /mnt/project/constrained-flow-noise-20261014 --population /mnt/project/flow-safe-improvement-20261013/population/manifest.json --output /mnt/project/constrained-flow-noise-20261014/formal_N_FLOW_NOISE_seed42_v1 --source-lock /mnt/project/constrained-flow-noise-20261014/formal_sources.json
```

Recovery uses the same variant/output/source-lock command with `--resume`
pointing to that run's actual `latest_checkpoint.json` path. It does not reset
the optimizer, lambda, raw cursor or scheduler. A registered safety pause is
not automatically resumed.

The new `lock_final` adapter implements the already registered safe DEV rule
for these two lines only. It checks all eight endpoint states and every fixed
DEV observation before sealing any Navtest selection. Endpoints are always
kept separate from DEV selections, and identical exported weights reuse one
evaluation. Maximum final budget is five unique models (SFT, two endpoints,
two selections), one seed 2101 each. No new Navtest prediction or score has
been accessed for selection. Its read-only invocation was executed and
correctly returned `WAITING_FOR_FORMAL_ENDPOINTS`; two targeted tests passed.

`evaluate_final` reuses the existing original-observation locked Navtest actor
and canonical single-candidate worker. Its CLI smoke passed; actual Navtest
execution remains gated by the endpoint lock. `summarize_runs` was exercised
on both real corrected qualifications and produced CSV/PNG curves in the
private smoke directory, explicitly marked qualification-only. Stochastic
candidate metrics and original ODE DEV metrics occupy separate tables/panels.
A first reporting smoke encountered the legacy C_SDE log's absent entropy
field; this is now retained as unavailable rather than filled with zero.
Covariance-aware C_SDE entropy remains available in its real monitor summary.

Both formal jobs started at 07:05 UTC on the assigned eight-GPU hosts and
completed the fresh 128-scene initial monitor, initial atomic recovery point,
and genuine on-policy actor updates. At the 07:19 snapshot, all eight ranks
had completed 26 C_SDE updates (320 raw / 208 accepted scenes) and 36
N_FLOW_NOISE updates (448 raw / 288 accepted scenes). Different collection
boundaries and shared-host speeds are recorded rather than treated as equal
update budgets. Both continue toward the same 203,184 raw-scene endpoint.
No post-update formal DEV point is available yet; the first is raw 25,398.

Full-cache publication has content fingerprint
`51f04cab23a0714d67e50d0dd2aa64f8ce18230f061d61c77f293f5491c227aa`.
Every condition payload was loaded/validated and every metric payload's SHA
was verified. The last metric-only verification used eight bounded reader
threads and took 290.08 seconds. Earlier serial reading, its elapsed cost and
source are preserved; only that owned read-only finalizer was restarted after
the complete condition seal existed. No cache content, training source,
calibration value or numerical execution setting changed. Measured allocated
directory sizes are approximately 3.9 GiB for corrected conditions, 31 GiB for
new v1 caches, and 2.8 GiB for the shared preserved old v1 cache directory.

The independent fixed TRAIN monitor at **zero actor updates** gives the
following starting stochastic policies. These 128 scenes are separate from
the calibration set, and this table is not DEV or deployed ODE performance:

| Initial policy | Stochastic PDMS (0–100) | Joint safety pass | Zero final-A groups | Pairwise ADE (m) | Mean XY second difference (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| C_SDE | 91.7721 | 97.6074% | 24/128 | 0.1341 | 0.2378 |
| N_FLOW_NOISE | 91.8595 | 97.6074% | 23/128 | 0.1800 | 0.3713 |

Higher initial diversity includes additional jitter and does not establish
useful exploration or a learned improvement. The actual run identities are
`86d20f6c67e548634aa29d7e0091a2fada50253b91473e1bc1a9f0802e4b4cc9`
and `eb9d087b349dfa49183d0322e7394bb75259684a87ef41d66c0c395ade3f55b1`.
The fixed-point original ODE DEV evaluations execute within each trainer;
the separate `finish_registered` process waits for both valid endpoints before
locking and running final Navtest. It never restarts training or changes a
safety pause. The main goal remains incomplete until the actual endpoints,
single-ODE evaluations and result analysis are delivered.

The paired `compare_ode` adapter reuses sealed canonical scores and the
existing log-cluster bootstrap. It has been run on the actual 1,696-scene
qualification/SFT pair and reproduces the earlier +0.000342-point difference;
this is marked qualification-only. Formal N_FLOW_NOISE minus C_SDE intervals
will use paired scene differences, rather than subtraction of two separate
confidence intervals.

The curve exporter also aggregates the saved velocity-call counters, separately
for rollout, frozen reference, PG/retention, GT replay and the fixed TRAIN
monitor. Completed inner updates are counted from the intersection of all eight
rank logs, so a scored collection does not automatically count as two updates.
Padding remains part of compute cost; accepted scene exposure and GT replay
exposure are separate. [The accounting check](compute_accounting_acceptance.json)
uses both actual 32-scene qualification runs: each has 512 candidate scores,
5,120 rollout calls, 24 frozen-reference calls, 126 PG/retention forward and
backward calls, and 64 GT replay forward and backward calls. These are logical
velocity calls with examples counted separately, not GPU kernel/FLOP counts.
Each cold graph-creating rank additionally executes three warmups and one
capture; that setup is explicitly separate from candidate/update counts.
Repeated or partially failed attempts require their preserved receipts rather
than being inferred from the final logical stream. The 83 frozen training and
evaluation source files still match the formal source seal; this change only
extends postprocessing, and the waiting final controller will use it for the
final curve export.
