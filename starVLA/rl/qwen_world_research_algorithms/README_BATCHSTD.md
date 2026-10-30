# PPO with centered batch standard deviation

This is a separate normalization-only experiment. The first 1,000 updates
of PPO_FULL, CISPO_FULL and PPO_ADAPTIVE already use a global standard
deviation of **raw rewards**, not a per-group standard deviation. Every
saved advantage in the 6,000 TRAIN groups matched that formula exactly.

For 8 fresh scenes, each with 8 candidates, write

    c[g,i] = reward[g,i] - mean_i(reward[g,i])
    source_A[g,i] = c[g,i] / (population_std(all 64 raw rewards) + 1e-8)
    new_A[g,i] = c[g,i] / (population_std(all 64 c values) + 1e-8)

The new denominator follows [REINFORCE++-Baseline Eqs. 11–12](https://arxiv.org/html/2501.03262v8#S3.SS1)
and the order in [OpenRLHF's official experience maker](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/ppo_utils/experience_maker.py).
The source epsilon is retained to isolate the denominator's population.
This is not full REINFORCE++: source PPO ratio, beta50 mean-reference
anchor, optimizer, Memory scope, data order and four inner epochs stay.
[DAPO Eq. 9](https://arxiv.org/html/2503.14476v1#S3) itself uses group std;
do not call this new normalization a DAPO implementation.

CPU analysis of PPO_FULL found raw/centered std ratio mean 1.21835,
median 1.17367, p90 1.40621 and max 2.93496. Group-mean variation accounts
for about 29.5% of raw reward variance on average. This arm tests whether
that variation unnecessarily suppresses within-scene PG relative to the
fixed reference anchor. It is not a demonstrated performance improvement.
There were no active groups with reward range below .001. Actual PPO
blocking averaged 1.691% of effective items, and gradnorm never reached
clip1 (maximum .04697); neither small-difference amplification nor global
gradient clipping was an observed failure in this training segment.

Create a new config from PPO_FULL, changing only the run identity and
adding under `research`:

    arm: PPO_BATCHSTD
    policy_objective: ppo
    advantage_normalization: group_centered_global_advantage_population_std_v1

Use the same formal arguments with this distinct entry:

    python -m torch.distributed.run --standalone --nproc-per-node=8 \
      -m starVLA.rl.qwen_world_research_algorithms.batchstd_entry \
      --config CONFIG --manifest FULL103K --output NEW_RUN \
      --il-checkpoint IL_PARENT --parent-selection PARENT_SELECTION \
      --data NAVTRAIN --stop-after-update 1000

The new runtime injects its finalizer and validators explicitly. The
generic engine accepts runtime hooks and passes validators as keyword
arguments to checkpoint save/restore. Existing entries use their old
defaults; no module globals are patched, and old run identities are not
reinterpreted. The source boundary shape/digest/optimizer/RNG checks remain.
The new validator verifies saved advantages against all ranks' saved
rewards at collection commit and native checkpoint save/restore.

Finalization performs one global gather before four updates and stores
the resulting detached FP32 advantages. All inner updates and a pending
native resume consume that same tensor. Tied groups remain in the 64-way
variance and retain the source zero-PG mask. `global_std` keeps its raw
reward meaning; `normalization_std` records the new denominator. Policy
metrics include both and their ratio. Uniform full data, seed42, eval2101,
all learning rates and the 50k scheduler identity are inherited unchanged.

CPU check: `python -m pytest -q tests/test_qwen_world_research_policy.py`.
It checks the complete 64-candidate scale, population variance, tied
groups, one gather/four inner reuse, exact loss/gradient/anchor agreement
with manually supplied advantages, and rejection of old normalization
semantics at the new recovery boundary. No GPU qualification is required.
