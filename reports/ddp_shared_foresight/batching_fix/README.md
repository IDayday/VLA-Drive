# Batched W injection correction

Four-GPU batch32/micro4 startup from05422d2 failed before any optimizer update. Preserved run:student_batch32_profile_v1. Minimal actual CUDA counterexample reproduces the failure only with deterministic algorithms and batch>1: the indexed assignment broadcasts a [1,Q,H] query value incorrectly in installed PyTorch2.5.1. Batch1 and nondeterministic batch4 pass; BEFORE.json records the observed error.

Use shape-checked scatter at unique query positions with an explicitly expanded batch. Native token sequence, mRoPE and attention are unchanged. Both shared W and per-scene ego-history embedding use the same function. CUDA FP32/BF16 batch1/4 output values and exact W gradients pass AFTER_CUDA.json. Full real Qwen padding and full batch training are verified separately; this counterexample alone is not VLM validation.

The earlier micro1 continuous4 vs2+2 proof remains valid for its actual tested scope. No formal student was started with the broken multi-sample injection. The full GT-MAE teacher uses a separate frozen worktree and is unaffected.

Full real validation at frozen56bf61e: three-front image/DeepStack/mRoPE padded batch versus separate FP32 inference passed predeclared atol1e-3/rtol1e-4. Ego maximum absolute difference4.768e-6; REAL_QWEN.json records per-sample hidden differences. Four-GPU batch32 micro4 and micro8 each completed4actual optimizer updates, all with measured FP32-master changes. Micro8 steady median3.56s/update; rank0peak allocated19.35GB. THROUGHPUT.json reports actual measurements and a compute-only100k-update estimate, excluding auxiliary heads/loading/saves/evaluation.

An independent micro8 run stopped after2updates and resumed for2on the SAME frozen source/devices. Final model, all4FP32 optimizer partitions, scheduler/data progress and4rank RNG states are exactly equal to the continuous4run (REAL_BATCH32_RESUME.json). This is4more real updates/128presentations, not formal learning evidence. Total new student startup updates so far20,416scene presentations across the five completed logical runs. The earlier failed batch32run had0updates. All test cards returned to pressure scripts.
