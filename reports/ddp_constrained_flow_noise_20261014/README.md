# DDP constrained Flow-Noise: execution record

Source start: `6edb05d576b40a9718421e86c06d560f184a11c6`.
Branch: `feature/ddp-learned-noise-constrained-grpo-20261014`.
Status: core and real-learning acceptance passed; two formal launchers queued
behind complete cache sealing. No formal training result yet.

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
generation and missing canonical v1.1 metric-cache production are running.

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
