# Initialization, warm start and training RNG

`module_init_seed`, `training_rng_seed`, `sampler_seed` are distinct config fields; bootstrap seed remains explicitly under execution. Stable SHA256-derived module seeds bind base seed, module kind and workflow/expansion identity. They never use Python hash(). A context saves/restores Python, NumPy, CPU and initialized CUDA RNG around network construction; only the CPU constructor generator is reseeded. DDP ranks check initial module identity before wrapping. Rank-specific training streams remain supported.

Scorer round1 starts from this deterministic state. Later Scorers load the previous completed selector's best state_dict strictly, with parent checksum recorded; a training parent can differ from the last accepted serving bundle. Router pool changes retain existing stable-ID output rows, initialize new rows under the recorded seed and rebuild all soft targets. Rejected rows are not relabeled.

Fresh processes with different preceding random draws produced identical real Scorer hashes, matching the workflow's initialization identity: [REAL_SELECTOR_INIT.json](evidence/REAL_SELECTOR_INIT.json). Changing the init seed changes fresh weights. Automated tests cover whole-workflow/standalone constructor agreement, warm parent, Router new rows, preservation of caller RNG and resume restore order. Real round2/3 Scorers warm-started and executed actual backward with frozen generators; see round_*_scorer_identity/result.json.

Resume reconstructs the module, strictly loads saved state, then restores saved RNG. No subsequent seed reset occurs. Legacy runs lacking init provenance are labeled UNKNOWN_HISTORICAL_INIT_SEED; the repair never claims their initial randomness was reproducible retroactively.
