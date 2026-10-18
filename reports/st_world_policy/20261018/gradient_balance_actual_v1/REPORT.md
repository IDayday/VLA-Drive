# First actual gradient diagnostic: execution failure retained

The real checkpoint and native action state loaded successfully. The diagnostic then rejected the first frozen encoder linear operation because deterministic algorithms were enabled without the explicit CuBLAS workspace setting. No BC/FPO gradient result was produced; optimizer updates and new official scores are both zero. This is an execution-environment omission, not evidence that the policy gradient is zero or that the mathematical objective works.

The wrapper reported an owned descendant after leader failure and cleaned the owned process tree; live-owned count is zero. Wall362.422s, peak observed RSS26.14GB, conservative nominal read charge73.83GB and observed physical reads29.19GB. Part of the time was spent waiting on shared-storage IO. These costs are retained.

The original C1 launcher declares :4096:8 and its trainer requires an explicit valid CUBLAS_WORKSPACE_CONFIG. The repair pins that environment before importing PyTorch, rather than disabling determinism. It creates a separately registered v2 output; source83679937, checkpoint3584, original four rank batches, objectives, RNG protocol and budget remain unchanged. Its independent delta approvals and actual execution are separate evidence.
