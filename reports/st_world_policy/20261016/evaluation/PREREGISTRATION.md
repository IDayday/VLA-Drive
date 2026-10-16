# A8 evaluation preregistration

Status: **definitions registered; all formal SP0–SP3 and C0–C4 results NOT_RUN**.
This is an analysis contract, not a training approval. The implementation starts
from `9250884e51b7cfc45a68331ca037b32a26ec375c` in the isolated Evaluation worktree.
`PREREGISTRATION.json` is the machine-readable companion. Its empty `formal_runs`
map deliberately blocks FORMAL analysis until the lead binds reviewed real runs.

## Population and initialization

The existing full current-only population manifests were compared with the
original `reports/ddpolicy_vehicle_from_scratch/NAVTRAIN_PARTITION.json`:

| Role | Scenes | Logs | SELECTED_SCENES.json SHA256 |
|---|---:|---:|---|
| Train | 101592 | 1176 | `58dffa070d434d15cb9adf6382010520aff57f4aa624aa081412b70e6f28b7ea` |
| Dev | 1696 | 16 | `4ce40893281f3cf246989cbbac6bbd80c40c06ec937a4405a5a7c353c6dd0573` |

Both contain each original token once, with the original token/log mapping.
Train/dev share neither scenes nor logs. Exact paths, source identities,
completion receipts and a sorted scene/log fingerprint are recorded in
`POPULATION_VERIFICATION.json`. This audit read metadata only. It did not rehash
images, weights or metric-cache payloads and did not read Navtest.

All arms start from the original S0 100k FP32 masters bound by
`/mnt/project/st-world-policy-20261016/bindings/BASE_BINDING.json`, binding identity
`3b910a6d4245580d77f868e4e0af5faf2fc7fccc25c561c935f219ab1bf7f099`, parent checkpoint
`78f870647015194798af067eefc8d8fedb6e59e6db2f3ae719c63b783871446f`.
The parent's historical internal arm name is C1; this is not the new consequence
C1 treatment. All matched arms reset their optimizer at the common S0 start.
The parent has already seen these training logs. Results concern continuation
exploration, not independent from-scratch generalization.

## Fixed arms, contrasts and checkpoints

| Arm | Teacher pixels W×H | Target grid W×H | W/view | Supervised positions/view |
|---|---|---|---:|---:|
| SP0 (`S-P0`) | 256×192 | pooled 8×6 | 48 | all 48 |
| SP1 (`S-P1`) | 384×288 | pooled 8×6 | 48 | all 48 |
| SP2 (`S-P2`) | 384×288 | native 24×18 | 48 | uniform 96 |
| SP3 (`S-P3`) | 384×288 | native 24×18 | 96 | same uniform 96 |

All four use the same original current three-view student input and only the
current DINO auxiliary task. Historical future/MAE supervision is disabled in
this registered spatial pilot. SP1−SP0 measures teacher input resolution under
the matched target grid. SP2−SP1 measures the **fine-grid supervision scheme**,
which also changes supervised position count. SP3−SP2 measures W capacity under
the same target/sampling protocol. All three are primary spatial contrasts;
unregistered comparisons are descriptive.

| Arm | Original FM/BC | Additional policy reward | Primary comparison |
|---|---|---|---|
| C0 | retained | none | continued-training control |
| C1 | retained | canonical total PDMS | C1−C0 |
| C2 | retained | `−any(road violation)` | C2−C0 |
| C3 | retained | canonical NC−1 | C3−C0 |
| C4 | retained | `−any(TTC violation)` | C4−C0 |

The C groups share frozen Qwen, trainable original DiT/existing projections,
the original BC exposure, common data order and seed, and original ten-step
deployment. Training four candidates does not authorize deployment selection.
FPO/feedback mathematics remain owned by their existing implementation/evidence
cards. C5 is not authorized by this registration and requires a separate locked
reward/progress definition after the C0–C4 evidence.

| Family | Fixed diagnostic/screening points | Common endpoint | LR clock |
|---|---|---|---|
| Spatial | 5000 optimizer updates, all four arms | 20000, retained contrast pairs | one shared 20000-update schedule |
| Consequence | 512 and 1024 optimizer updates | 4096, all C0–C4 | one shared 4096-update schedule |

Common AdamW starting recipe: LR `1e-5`, minimum LR `5e-7`, betas `[.9,.95]`,
epsilon `1e-8`, weight decay `.001`, gradient clip `1`. Warmup is 1000 spatial
updates and 100 consequence updates. These are proposed engineering choices
registered from the existing configs, not conclusions from their PROFILE
weights. The LR horizon is the common endpoint, including at the earlier pause;
resume carries optimizer, scheduler, RNG and data position. No restart of the
schedule at 5k, no independent 5k cosine control, no best-checkpoint substitution.

The source contract §12.2 says: “共同S0起点，计划20k更新，5k阶段暂停；仅入围对继续至共同20k；不得把5k当从头充分训练”.
It does not specify two primary contrasts or authorize a four-arm union. The
conservative budget interpretation registered here advances **at most one
contrast, exactly its two endpoint arms**, after lead/Evidence review before
formal results exist. No contrast union expands this into three or four 20k arms.

Spatial shortlist rule: after all four complete 5k evaluations, rank the three
registered paired PDMS differences by descending full-dev mean. Retain at most
one contrast and **both ends**; exact ties use SP1−SP0, SP2−SP1, SP3−SP2.
Nonpositive results
may still receive the finite common endpoint to resolve uncertainty, but are
not called improvements. Report every 5k arm and each selected 20k pair.
Unresolved/invalid pairs are not rankable. Missing core controls cannot be
replaced by the untrained S0 or a different step. All safety/progress diagnostics
must accompany the shortlist; a PDMS ranking does not establish safety.
This deterministic shortlist is a newly proposed analysis choice for lead and
Evidence review before any formal run, not an approval issued by A8.

The 512/1024 C diagnostics check finite learning, failure modes and budget; they
do not select a best epoch or eliminate a low-scoring arm. All C0–C4 continue to
4096 unless a recorded technical failure, resource limit or approval withdrawal
prevents it. Such runs retain FAILED/BLOCKED/NOT_RUN status. No one-sided budget
extension follows a disappointing score. A later seed is a separate preregistered
replication. Initial training seed and inference seed are both 42, with distinct
RNG streams. Resampling logs is not a substitute for repeating training.

## Outcomes and uncertainty

The primary estimand for every registered contrast is
`sum_scene(PDMS_treatment − PDMS_control) / 1696`, at the same fixed checkpoint.
PDMS and its six components remain in official `[0,1]` units. Every table contains
the mean of PDMS, NC, DAC, TTC, EP, Comfort and DDC, and all paired differences.
DDC remains reported even though its canonical score weight is zero. No new
weighted score or component formula replaces `score_world`.

Uncertainty: sort log names, use NumPy `Generator(PCG64(20261017))`, draw 16 logs
with replacement 2000 times, and keep every scene of each selected log with its
multiplicity. Use the **same draws for both policies and all metrics/contrasts**.
Each draw is a ratio of sums: sum of paired scene differences divided by the
number of sampled scenes. Unequal log sizes must not become an unannounced mean
of log means. Equal-log means are a separately labelled sensitivity analysis.
Report the linear-quantile 2.5th/97.5th percentiles, not a scene-IID bootstrap.
The deterministic draws follow the sorted full log list; reading CSVs in a
different order cannot change results.

These are unadjusted exploratory intervals, conditional on the S0, training
seed, inference seed and dev selection. No familywise-significance claim or
post-selection confirmatory claim is supported. Only 16 dev clusters are
available. Empty strata return `null` mean/interval with denominator zero.
Bootstrap draws that contain no member of a stratum are counted as undefined,
never filled with zero; the interval states how many draws were defined.

Fixed failure and regression diagnostics:

- Zero means canonical PDMS `<=1e-8`, matching the scorer summary. Report the
  full 2×2 zero/nonzero transition table. Zero→nonzero repairs and nonzero→zero
  new failures each have a full-population rate and a conditional denominator
  (control zeros or control nonzeros). Report net repairs minus new failures per
  full population with paired log intervals. Legal zero scores always remain.
- High-score scenes are defined from the matched control at that checkpoint:
  PDMS `>=.9`. Report any decline beyond `1e-8`, a decline strictly greater than
  `.05`, high→zero, their counts/full-population/conditional rates, and all
  component deltas in the fixed high-score stratum. Control-zero and middle
  strata are also reported, without replacing the full population.
- Report EP change and the rate of EP drops strictly greater than `.05`.
  `safety_gain_with_ep_drop` counts a gain beyond `1e-8` in any NC/DAC/TTC plus
  such an EP drop; it is an exchange diagnostic, not a causal claim.
- Report raw executed progress in meters, fixed-reference progress and mean
  executed speed. The stopping proxy requires reference progress `>=5m`,
  candidate raw progress `<=1m`, and mean executed speed `<=.5m/s`. Speed is the
  mean Euclidean XY speed at the 40 executed states `t=.1…4.0s`, excluding t0.
  Report each policy's count, full-population and moving-reference rates, their
  paired difference and newly flagged stops. These thresholds are transparent
  exploratory choices, not a universal definition of unsafe stopping.
- ADE is mean XY Euclidean displacement from the original GT over all eight
  command waypoints `.5…4.0s`; FDE uses the 4.0s XY endpoint. Both use meters,
  original label normalization/ego frame, no yaw penalty, and no tracker-space
  substitution. GT labels are read only by this offline analysis, after the
  immutable label-free deployment export. Missing/nonfinite GT blocks this
  complete analysis; no silent valid-only ADE denominator is substituted.

All component means and strata have explicit scene/log counts. A positive PDMS
difference with adverse safety, progress or high-score shifts is reported as a
tradeoff, not unqualified safe improvement. Absence of a significant difference
does not prove noninferiority. Any stronger noninferiority margin or confirmatory
claim needs a separate preregistered design. Scene inspection can explain a flag
but cannot remove it from the registered population.

Spatial reconstruction/probe quality, future prediction, consequence prediction,
gradient direction and actual planning benefit are separate outcomes. This CLI
measures planning and its diagnostics only. The spatial small-target and
road-boundary/depth probe labels, coverage and shared readout capacity must be
frozen by their owners before those outcomes are reported. PCA error or a lower
auxiliary MSE does not select an arm here.

## Invalid data, coverage and identity

`evaluate_campaign.py` is a read-only consumer of `export_world` and
`score_world`, with immutable analysis outputs. It never invokes the scorer or
loads a model. It requires matching full population, current input/history
contract, scene seed, solver, precision, inference environment, canonical
protocol, per-scene metric-cache receipt and executed reference/context hashes.
It verifies the CSV hash, rejects duplicate/missing/extra scenes and foreign
logs, verifies each small prediction payload/receipt and its scored trajectory
hash, and checks every CSV value against the saved per-scene feedback ledger.
No metric, transition or missing row is reconstructed from aggregate means.

Original valid prediction and scoring are separate validity states. Missing
prediction, malformed shape, NaN/Inf, illegal output, missing/incompatible cache,
scorer failure and valid PDMS zero must remain distinguishable in export
attempts and feedback records. The existing scoring ledger owns bounded retry
rules. Analysis records invalid/interrupted earlier scoring attempts even if an
approved retry succeeded. Any unresolved final scene, nonfinite metric, invalid
metric flag or changed context returns `BLOCKED_ANALYSIS`, exit 2, and no complete
comparison. Repair the artifact/technical problem and use the same immutable
policy/prediction and registered retry policy; do not resample until a good
trajectory appears. Systematic prediction failure is FAILED and is not repairable
by dropping scenes or replacing failures with zeros. The fixed N remains 1696.

Verification boundary: this reader reuses the scorer's already-verified native
checkpoint and metric-cache byte receipts. It does not rehash large weights,
images or metric-cache payloads. It verifies metadata, per-scene NPZ/GT payloads
and saved feedback files. The original exporter/scorer and independent reviewers
remain responsible for generating trustworthy native/cache receipts. Local
hash-verified ledger `.pt` files are trusted campaign products, not upload input.

## Remaining bindings and approval owners

Pre-registered now: the seven primary contrasts, fixed checkpoints, common LR
clocks and proposed optimizer settings, full train/dev population identities,
original S0 lineage, deployment/scoring contract, paired log estimator,
bootstrap RNG/replicates, transition/stratum/progress definitions and invalid-data
rules. The actual train/test separation was checked, not inferred from counts.

Still pending: final integrated training/export SHA; the full three teacher
cache identities; shared train-only PCA fit/scope and transform; final data and
target manifests; gradient/init/numeric-execution audit; effective batch,
candidate/BC exposures and resource lease; exact run configs and immutable
checkpoint receipts; all independent approvals. The existing 64-scene PROFILE
and two-update SMOKE configs/weights supply no formal population or approval.
Full train flow-context coverage must be verified before its 101592-scene
manifest can be bound; the existing real32 manifest cannot be promoted by
renaming its scope.

The lead adds one `formal_runs[arm]` entry per real run, each containing pinned
`registration`, `approval` and `run_identity_file` `{path, sha256}` records.
`_formal_binding` checks the config cell, completed step, common schedule,
training seed, run/checkpoint identity and independent review receipts. The
existing `check_formal_approval` remains the training launch gate. Evidence,
Correctness and Systems must each sign their own artifacts; A8 issues no PASS
for those roles. Formal config changes require a versioned preregistration
amendment before examining the affected formal comparison. Later analysis-only
fixes must identify their impact and reproduce earlier outputs.

## Existing real-data engineering validation

Source: `/mnt/project/st-world-policy-20261016/smoke/flow_C1_export_dev16_v1/`.
The existing `official_score` and its sibling `predictions` contain the same
two-update SMOKE policy, one selected dev scene per log. A/A means that exact
same immutable export is supplied as both sides. This is not a second model,
not a C1−C0 result, and not formal dev coverage.

The reader verified 16/16 real scene records from 16 logs. Direct CSV means:
PDMS `.9376117025253765`, EP `.8502680860609033`, DDC `.9375`, and NC/DAC/TTC/
Comfort `1`. Offline GT alignment gave ADE `.7097112681m` and FDE `1.4644756724m`.
There were no zero scores or stopping-proxy flags; only 15 of these selected
scenes have reference progress at least 5m. Every A/A delta and its interval is
exactly zero. The control-zero denominator is zero and is reported as undefined.
None of these observations supports a scientific arm ranking.

Artifacts are `smoke_dev16_aa_v2/ANALYSIS.json` and `PAIRED_SCENES.csv`; the latter
contains the actual scene joins, component values and differences. No image,
model weight, metric cache or raw trajectory is committed.

Reproduce the local CPU analysis (no GPU or scoring calls):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.evaluate_campaign \
  --plan reports/st_world_policy/20261016/evaluation/PREREGISTRATION.json \
  --population /mnt/project/st-world-policy-20261016/audits/history_data/smoke_dev16_pinned_v2/SELECTED_SCENES.json \
  --control-score /mnt/project/st-world-policy-20261016/smoke/flow_C1_export_dev16_v1/official_score \
  --control-export /mnt/project/st-world-policy-20261016/smoke/flow_C1_export_dev16_v1/predictions \
  --treatment-score /mnt/project/st-world-policy-20261016/smoke/flow_C1_export_dev16_v1/official_score \
  --treatment-export /mnt/project/st-world-policy-20261016/smoke/flow_C1_export_dev16_v1/predictions \
  --scope SMOKE --comparison SMOKE_AA --stage SMOKE \
  --output reports/st_world_policy/20261016/evaluation/smoke_dev16_aa_v2

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/st_world/test_evaluate_campaign.py tests/st_world/test_score_world.py
```

For a later approved comparison use `--scope FORMAL`, one of the seven registered
comparison names, its registered stage, the **full** dev selected manifest, and
the real immutable control/treatment directories. No executable formal example
is fabricated while those runs do not exist. Navtest remains unread and unrun.
