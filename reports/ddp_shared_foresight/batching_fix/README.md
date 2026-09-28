# Batched W injection correction

Four-GPU batch32/micro4 startup from05422d2 failed before any optimizer update. Preserved run:student_batch32_profile_v1. Minimal actual CUDA counterexample reproduces the failure only with deterministic algorithms and batch>1: the indexed assignment broadcasts a [1,Q,H] query value incorrectly in installed PyTorch2.5.1. Batch1 and nondeterministic batch4 pass; BEFORE.json records the observed error.

Use shape-checked scatter at unique query positions with an explicitly expanded batch. Native token sequence, mRoPE and attention are unchanged. Both shared W and per-scene ego-history embedding use the same function. CUDA FP32/BF16 batch1/4 output values and exact W gradients pass AFTER_CUDA.json. Full real Qwen padding and full batch training are verified separately; this counterexample alone is not VLM validation.

The earlier micro1 continuous4 vs2+2 proof remains valid for its actual tested scope. No formal student was started with the broken multi-sample injection. The full GT-MAE teacher uses a separate frozen worktree and is unaffected.
