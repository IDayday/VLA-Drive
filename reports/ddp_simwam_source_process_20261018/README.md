# SimWAM source-data and published-command process replication

实际启动状态、资源和首100步成本：[中文进展](launch_status_zh.md)。
20k开发评价、当前训练进度及Navtest队列恢复：[20k进展](progress_20k_20261018.md)。

This is a new, registered experiment requested on 2026-10-18. It keeps the
qualified B source ActionDiT and copies the locked SimWAM RL process as closely
as the retained DDP observation/action interfaces permit. It does not add a
new safety objective, curriculum, search, learned noise, replay, or seed matrix.
Training and evaluation completion must be established from actual receipts;
configuration arithmetic is not a result.

2026-10-18 14:47 UTC: both fixed **20k full Navtest observations are complete**,
including native/shift5 ODE and official v1.0/v1.1. See the
[full results and version explanation](navtest_20k_results.md). The main
source-data run gains +0.3087 points in v1.1 native ODE, with a zero-crossing
log-bootstrap interval and NC/TTC regressions. The 170k training endpoints
remain in progress; the earlier launch/progress pages are historical snapshots.

## Source and version boundary

Upstream: [H-EmbodVis/SimWAM](https://github.com/H-EmbodVis/SimWAM/tree/1433ee7e06a62bd8b89da74455df0d82bda7b726),
commit `1433ee7e06a62bd8b89da74455df0d82bda7b726`.
The local upstream checkout was clean. The source file hashes are in
[source_data_audit.json](source_data_audit.json).

The default training config is **20,000 optimizer updates**. The README labels
the released RL checkpoint as step 20,000, but its explicit training command
sets **170,000**. We register the latter command budget and retain a fixed 20k
observation. We do not claim that 170k produced the published 91.9 model, or
that the unavailable original training log has been reconstructed.

## Data and matched runs

The source list is
`navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain_pdm_score_below0p9.yaml`.
It contains 17,389 unique tokens. Of these, 17,037 are in our locked RL_TRAIN,
352 are in locked RL_DEV, and none are in Navtest. All 352 DEV tokens are
excluded from updates. This is an explicit data adaptation, not an exact copy
of the author's full population.

The unchanged source `filter_scenes` function establishes scene order from the
local raw trainval logs, using the original frame/filter settings. That order is
frozen before applying the source epoch sampler and padding. The author's
filesystem enumeration is unknown; cross-machine identical sample order is not
claimed. All retained conditions and v1.1 cache entries were checked. Preparation
used existing authenticated single-candidate B SFT scores, with no new inference
or scoring. Private tokens and logs remain outside git.

| Run | Fixed population | Optimizer updates | Purpose |
|---|---:|---:|---|
| UPSTREAM_DATA | Source tokens, TRAIN-only: 17,037 | 170,000 | Main source-process replication |
| OWN_DATA_CONTROL | B SFT canonical score < 0.9: 21,836 | 170,000 | Same longer budget; isolates data selection |

The populations overlap on 15,904 scenes. On the source population, B SFT has
15,690 safe low-score scenes, 214 unsafe scenes, and 1,133 safe high-score
scenes. These are descriptions of a fixed source list, not additional filters.
The two runs start fresh from the **same B SFT update 100000**, with identical
seed-42 LoRA initialization. Old RL and qualification weights are not starting
weights. The SFT model SHA256 is
`4e71b1fcf963d38456174de1f8d17762a9a4ccf1597c22b7cec262026d9201b7`.

## Unchanged training semantics

Each run uses eight GPUs, one scene per GPU, eight candidates per scene,
ten generation steps, shift 5, and stochastic indices `[7,8,9]`. It retains
the source score-corrected SDE, noise level 0.1, min std 0.001, innovation
clamp ±5, per-element relative logprob clamp width 7, then dimension mean.
PPO clip is 0.005. Advantages use scene centering and the unscreened global
reward population std. Tied groups mask PG but retain the reference anchor.

It retains attention LoRA rank 16 / alpha 32 / dropout 0, permanently frozen
SFT reference, transition-mean MSE anchor beta 50, and DDP native anchor
weights `[0.2,1,0.5,0.5]`. The heading-pair adaptation is not angular MSE
equivalence. Training reward is source EP:TTC:Comfort = **10:5:2** in both
runs; formal PDMS is canonical **5:5:2** in both runs.

AdamW uses LR 5e-5, betas (0.9,0.95), weight decay 0.01, 100-update warmup
and constant LR thereafter. Effective source gradient clipping remains **0**.
One fresh rollout supplies four actual optimizer updates with fixed old chain,
probabilities, reward and advantage. No old constrained C/N objective remains.
Partial-step gradients and clamped probabilities retain their source empirical
proxy semantics; they are not an unbiased full mixed-chain policy gradient.

The actor, sampling, scoring, advantage, anchor, optimizer and checkpoint
implementation is reused unchanged from the qualified B runtime. A separate
runtime admits the new fixed population and registered longer budget; the old
20k and own-hard assertions are not removed. The old controller's default
behavior is unchanged; an optional runtime supplies this finite evaluation queue.

## Explicit adaptations and limits

- Retain frozen DDP/Qwen observation conditioning and the complete B100k SFT;
  this is not the pretrained Wan/video backbone or the author's SFT checkpoint.
- Retain eight-by-four DDP action normalization, decoding and heading mapping.
- Retain the qualified FP32 actor/latent/optimizer and distributed layout as a
  numerical platform adaptation. Do not silently switch dtype for throughput.
- Exclude 352 locked DEV scenes, freeze local source-loader order, retain
  independent candidate scoring and explicit infrastructure-error failure.

The locked source and our canonical v1.1 files are byte-identical for scoring
entry, scorer, simulator and geometry: [scorer_equivalence.json](scorer_equivalence.json).
This supports reuse of scoring components; it does not prove equality of every
asset or the model's input information. Source and canonical reward readouts use
the same simulation, with separate weights.

## Budget, qualifications and evaluation

Per run, 170k updates imply **42,500 fresh global rollouts, 340,000 scene
exposures and 2,720,000 new candidates**. Actual logs must verify these numbers;
failed work, qualification, scoring retries and evaluations are separate costs.
The 20k observation corresponds to 320,000 candidates. Four inner updates never
count as four fresh rollouts.

Before formal training, each new runtime must pass real eight-rank continuous4
versus split2/resume4 comparison, including adapter/Adam/scheduler/RNG/chain/
old probability/advantage equality, frozen-weight checks, merged ODE equality,
and original-observation deployment. Qualification data/weights are isolated.
CPU contract and queue evidence: [cpu_validation.json](cpu_validation.json).

Registered observations: 0, 5k, 20k, 50k, 100k, 170k. Full DEV and fixed TRAIN
diagnostic use seed 2101 and separate native/shift5 ODE. Initial evaluations
reuse the authenticated B100k baseline. Full Navtest is registered at **20k and
170k**, with fixed endpoints, never a best-of-two-grid or best-seed selection.
Canonical v1.1 and official v1.0 rescore remain separate. Historical Navtest use
is disclosed; it is not a new blind test.

Private registration and run root:
`/mnt/project/simwam-source-process-20261018`.
Public preregistration: [registration.json](registration.json).
Actual host admission always rereads the current policy, checks two GPU capacity
snapshots and visible owners, and preserves real/unknown jobs. The fixed global
batch is not increased to consume spare GPUs; spare capacity evaluates independent
fixed milestones. No withdrawn or temporary host is used.

## Executed entry points

Run from the immutable archived source, with the project venv Python:

```bash
python -m scripts.analysis.fm_transfer.dispatch_source_process --mode UPSTREAM_DATA --phase qualify --output /mnt/project/simwam-source-process-20261018/qualification/UPSTREAM_DATA_v1
python -m scripts.analysis.fm_transfer.dispatch_source_process --mode UPSTREAM_DATA --phase train --qualification-root /mnt/project/simwam-source-process-20261018/qualification/UPSTREAM_DATA_v1 --output /mnt/project/simwam-source-process-20261018/runs/UPSTREAM_DATA_seed42_v1
```

The control uses `OWN_DATA_CONTROL` in both arguments and its matching paths.
Explicit same-run recovery adds `--resume <native COMPLETE checkpoint>` to the
formal dispatcher. No implicit failure retry or qualification-weight resume is
allowed. Controller and launch receipts, actual training state and measured
throughput are recorded separately from qualification results.

## Actual qualification and scheduling implementation

Both real eight-rank gates passed; evidence is in
[qualification.json](qualification.json). Continuous/resumed native states and
chains matched exactly. Maximum merged/unmerged ODE error was 2.3842e-7 in
normalized actions. Each gate scored 128 training candidates plus four
original-observation ODE requests, and took approximately five minutes. This
is implementation evidence, not a positive driving result.

Training is locked to immutable commit
`339e9609d9a051908bb5030e5333aebcc8b38bcd`. Later evaluation-controller changes
are separate execution code; they do not modify the running training snapshot.
The new fixed-stage Navtest lock distinguishes completed **20k observation
budget** from incomplete **170k formal budget**, instead of calling 20k a
completed 170k experiment. The existing exact trajectory worker, complete
population analysis and official-release CPU rescorer are reused.

[evaluation_queue_validation.json](evaluation_queue_validation.json) records
91 passing targeted tests. Dispatcher/controller CLI smokes and the official
v1.0 runtime/cache identity check also passed. Fixed-stage lock checks currently
refuse inference until both registered checkpoints and full matching DEV/TRAIN
milestones exist. No new Navtest prediction has been generated by these checks.

The finite DEV queue and finite Navtest queue have separate owners and disjoint
candidate host pools to prevent conflicting GPU reservations before model
loading. Navtest derives shard count from actual qualified individual GPUs,
including authorized sharing; it does not require an idle eight-card host.
Both queues stop on ambiguous failures and preserve records rather than silently
duplicating workers. Formal trainer recovery remains the native same-run path.
