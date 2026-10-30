# PPO_SAFE_COMFORT candidate

This changes only the scalar training reward on the same stochastic rollout:

    safe = (NC == 1) and (DAC == 1) and (TTC == 1)
    reward = (5*EP + 5*TTC + 2)/12 if safe else canonical_PDMS
    canonical_PDMS = NC*DAC*(5*EP + 5*TTC + 2*official_Comfort)/12

Official feedback records, scorer, canonical scores and evaluation remain
unchanged. `canonical_rewards` is stored separately from transformed
`rewards` in behavior and pending native checkpoints. Existing journals
already distinguish `canonical_random_mean` from `train_reward_mean`.
These transformed training rewards must never be displayed as PDMS.

Only fully safe candidates receive a comfort refund. TTC failures,
NC partial credit (.5), NC failures and DAC failures keep their original
reward exactly. This is a scalar gated transform, with no component-wise
standardization, advantage shrinkage, learned dual, filtering or new
rollout. The existing global RAW-reward std is used on transformed rewards;
the centered-batchstd experiment must not be combined with this arm.

In the first1k PPO_FULL TRAIN banks, 3,956/11,975 fully safe stochastic
candidates failed Comfort. All 5,443 observed EP/score rank inversions
between safe candidates involved different Comfort; constant-Comfort
pairs had zero inversions. Comfort accounts for 78.72% of the within-group
reward variance alignment among the 602 groups with all eight safe.
The existing ODE TRAIN1024 panel had Comfort1 for both IL70 and PPO1k;
20 overlapping training scenes also show ODE Comfort1 versus SDE .65625.
These observations support a sampling/deployment mismatch hypothesis,
not a causal proof that every comfort violation is artificial.

Offline on these same 2,000 groups, the gate displaces 516 strict reward
winners (25.8%; argmax-index changes including ties770), increases selected
EP on average .01966, and gives no new unsafe-positive advantages
(282 ->252). By comparison, fixing Comfort1 for every candidate yields
64 new unsafe-positive advantages. These are counterfactual reward
statistics, not measured policy-performance improvements.

To prepare a new config, copy PPO_FULL and change only its arm/output
identity plus the following research field:

    reward_transform: canonical_all3safe_constant_comfort1_train_only_v1

Keep `policy_objective: ppo`, `data_policy: uniform_full`, no
`advantage_normalization` field, beta50, clip.005, full103k, seed42,
evaluation2101, current learning rates and the same50k scheduler.
The distinct training entry is:

    -m starVLA.rl.qwen_world_research_algorithms.safe_comfort_entry

`SafeComfortRuntime` overrides finalization, not collection. It transforms
once after successful scoring, preserves raw rewards/records, then calls
the original raw-batchstd finalizer. All four inner updates reuse the same
saved advantages. Native boundary checks verify the explicit identity,
transform protocol and both reward tensors against original feedback;
rank validation retains the original source normalization checks. The
checkpoint/export schema and all model/RNG/sampler operations stay.

The principal risk is ODE comfort regression if removing this training
pressure transfers to deployed trajectories. Only official canonical
evaluation can establish a gain. This candidate has not been registered
or dispatched by this change. Three directed CPU tests cover mapping,
safety-gate cases, unchanged records, all64 normalization and native
boundary semantic validation; no GPU qualification was run.
