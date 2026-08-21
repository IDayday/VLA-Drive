# Gate 2 version-control snapshot

Captured before the Gate 2 snapshot commit on 2026-08-21 UTC.

## Current branch and commit

```text
feature/action-effect-world-model
ad90c9c24c13022ea6f29682003ad3c1fd4e1de4
```

## `git status --short`

```text
 M .gitignore
 M 4-infer.sh
 M 8-continue_action-only-qwen-visual-200k.sh
 M 8-train_action-only-qwen-visual.sh
 M docs/experiments/QWEN_VISUAL_ACTION_ONLY.md
 M env.local.example.sh
 M env.sh
 M infer.py
 M starVLA/model/framework/QwenOFT.py
 M starVLA/training/train_starvla.py
 M starVLA/training/trainer_utils/trainer_tools.py
 M tests/action_only/test_qwen_visual_action_only.py
 M tests/vggt_dense_bottleneck/test_framework_conditioning.py
 M tools/precompute_vggt_dense_cache.py
?? 13-eval_action_only_best_of_n.sh
?? configs/
?? docs/research/
?? reports/
?? research/
?? run_qwen_visual_200k_dlc.sh
?? scripts/
?? tests/action_effect/
?? tests/action_only/test_best_of_n_pdms.py
?? tools/summarize_pdms_best_of_n.py
```

## `git diff --name-only`

```text
.gitignore
4-infer.sh
8-continue_action-only-qwen-visual-200k.sh
8-train_action-only-qwen-visual.sh
docs/experiments/QWEN_VISUAL_ACTION_ONLY.md
env.local.example.sh
env.sh
infer.py
starVLA/model/framework/QwenOFT.py
starVLA/training/train_starvla.py
starVLA/training/trainer_utils/trainer_tools.py
tests/action_only/test_qwen_visual_action_only.py
tests/vggt_dense_bottleneck/test_framework_conditioning.py
tools/precompute_vggt_dense_cache.py
```

The unrelated action-only/VGGT files above predate this research branch work.
They are intentionally excluded from the Gate 2 snapshot. The
`env.local.example.sh` snapshot includes only the action-effect path stanza;
the pre-existing best-of-N stanza remains unstaged.

## Published Gate 2 snapshot

```text
commit: a41f081077d457bc5fd2d6c0550cba8c3c8dc880
message: research: add action-effect Gate1 and Gate2 pilot
tag: action-effect-gate2-v0 (lightweight)
```

No `git add -A`, reset, checkout-overwrite, stash, or checkpoint conversion was
used. Later frozen-feature extraction uses a detached worktree at this exact
tag so the unrelated dirty inference files cannot affect Phase 6.
