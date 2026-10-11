# Query backend alignment

P0 runs the current Query model with FP32 trainable parameters/gradients/Adam moments, BF16 language autocast, frozen FP32 visual, eight-rank DDP. The installed environment is PyTorch2.5.1+cu124. Eligible CUDA AdamW defaults to foreach; it is inaccurate to call the baseline a Python per-parameter optimizer. P1 fused explicitly changes the implementation flags after loading the old Adam state, preserving group hyperparameters and the100k scheduler.

P6 retains P0 visual precision and the same Query model, but uses BF16 execution weights and gradients plus separate FP32 master weights and FP32 exp_avg/exp_avg_sq. Stable trainable names bind masters, original FP32 weights initialize masters before BF16 rounding, and each update copies the FP32 master back to BF16 execution weights. Checkpoints store all masters, moments, step, scheduler and names. CPU tests verify an update smaller than BF16 resolution really changes the master and exact state resumes.

This implemented backend is **unsharded master AdamW with DDP**, not ZeRO2. The S-series DeepSpeed configuration is a historical reference, not an IQE measurement. A ZeRO2 partition/merge migration has not been qualified in this bounded round and is not advertised as available. No dependency upgrade, CPU optimizer/offload, vocabulary head, DiT head, or task-frequency change was introduced.

P6's numerical screen and timing are retained even if unsuccessful; low precision direct Adam updates without FP32 masters are prohibited. Kernel/communication arithmetic and BF16 residuals can change gradients, so this is T1. Cross-backend continuation is not bitwise. A GPU same-backend resume result is necessary before claiming this backend ready for a long run.
