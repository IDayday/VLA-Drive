# Action-only checkpoint training code

This branch preserves the training code associated with the checkpoints in
the [`action-only-checkpoints-v1`](https://github.com/IDayday/VLA-Drive/releases/tag/action-only-checkpoints-v1)
release.

## Trainable visual tower (step 120,000)

The branch root is the exact code state saved by the training run for
`action-only-unfrozen-visual-step120000-pdms89.575.pt`.

It was reconstructed from:

- base commit: `ad90c9c24c13022ea6f29682003ad3c1fd4e1de4`
  (`feature/add-VGGT`);
- the run-local source snapshot saved alongside the checkpoint.

Primary entrypoints and configuration:

- `8-train_action-only-qwen-visual.sh`
- `8-continue_action-only-qwen-visual-200k.sh`
- `run_qwen_visual_200k_dlc.sh`
- `starVLA/config/training/qwen_visual_action_only.yaml`
- `checkpoint_code/action-only-checkpoints-v1/configs/unfrozen-visual-step120000-run-config.yaml`

The saved run configuration contains historical absolute paths. Treat those
paths as provenance only; configure machine-local paths through `env.local.sh`
or one-shot environment variables.

## Frozen visual tower (step 100,000)

The frozen checkpoint was trained from base commit
`9dd6b71b324a58de005dc669394295b9354d189c`
(`feature/add-agent-query`) plus the exact files under:

```text
checkpoint_code/action-only-checkpoints-v1/frozen-visual-overlay/
```

The archived launcher explicitly freezes
`qwen_vl_interface.model.visual` and `qwen_vl_interface.model.lm_head`.
Its merged run configurations are stored under:

```text
checkpoint_code/action-only-checkpoints-v1/configs/
```

To reconstruct that source state in a separate checkout:

```bash
git clone --branch release/action-only-checkpoints-v1 \
  https://github.com/IDayday/VLA-Drive.git VLA-Drive-action-only
cd VLA-Drive-action-only

git switch --detach 9dd6b71b324a58de005dc669394295b9354d189c
git checkout release/action-only-checkpoints-v1 -- \
  checkpoint_code/action-only-checkpoints-v1
cp -a checkpoint_code/action-only-checkpoints-v1/frozen-visual-overlay/. .
```

Use `checkpoint_code/action-only-checkpoints-v1/frozen-visual-overlay/8-train.sh`
as the provenance copy or the reconstructed root `8-train.sh` after applying
the overlay.

## Checkpoint mapping

| Training route | Step | Metric | Release asset prefix |
|---|---:|---:|---|
| Frozen visual tower | 100,000 | PDMS 0.8879028 | `action-only-frozen-visual-step100000-pdms88.79.pt` |
| Trainable visual tower | 120,000 | PDMS 0.8957495, EPDMS 0.8925638 | `action-only-unfrozen-visual-step120000-pdms89.575.pt` |

Both routes require the token-extended `Qwen3-VL-2B-WorldAction` base model.
