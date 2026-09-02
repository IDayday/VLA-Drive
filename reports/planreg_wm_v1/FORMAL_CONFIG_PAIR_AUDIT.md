# Formal config-pair audit

Both formal launchers compose and fully resolve their Hydra configurations
under a file lock, then run `scripts/audit_formal_config_pair.py` before either
training process starts. The comparison permits only these differences:

- `agent.initialization.variant`
- `agent.initialization.vlm_checkpoint_sha256`
- `agent.initialization.vlm_config_sha256`
- `agent.vlm_config.vlm_path`
- `experiment_name`
- `output_dir`

Derived output paths are normalized before comparison. Any change to data,
world model, optimizer, schedule, seed, random trainable initialization,
generator, scorer, camera, precision, step budget, or GPU layout is fatal.

Both launchers require the same layout lock, the same seed-matched shared
trainable artifact, the same 103,288-record input-only cache manifest, and the
same paired VLM audit. Automatic resume is disabled. An explicit
`RESUME_CHECKPOINT` is accepted only from the same run directory when all
identity hashes match.

The resulting JSON report lists the exact allowed differences and asserts
`unexpected_difference_count=0`. It also records that both versions have the
world model enabled with `future_mode=correct`, K=1 GT trajectory conditioning,
and real future images from the first optimizer step.

Neither configuration implements multi-trajectory consequence modeling.
