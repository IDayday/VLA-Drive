# Independent gradient v3 resource reassignment review

APPROVE_GRADIENT_BALANCE_EXECUTION, correctness only. Exact v3 registration `6a48f942e8b57f568d74be0bd92cab389daa6140d0a0307896acd40da3358770`; wrapper `763c753ef28f8959d0f99b0e2693a50dfe25f9f421c469519e74f682a4912e11`; mathematical runtime `83679937fe9702be39f0184154fdf9ebdc4eb98a`.

The committed0792bc28 wrapper and actual source file match. OnlyREGISTRATION,REG_SHA,GPU_LOCK constants change; all other AST is identical to the independently approved environment-repaired wrapper. The registration changes only host/GPU and output. It preserves checkpoint3584,next32 scenes,original four-rank RNG/order,CFM math and budgets. Metadata-only CLI resolves v3 and the exact `CUBLAS_WORKSPACE_CONFIG=:4096:8` requirement. Direct source/control reads43644bytes; check elapsed0.072seconds. No PyTorch/GPU/model/image/GT/cache/scoring or dispatch occurred.

New resource identity is training-vlawm-tsinghua-zt2-worker-0, GPU-c0a3d651-8691-b507-5df6-6c7f0cc6c897 (Lead-designated index1), two CPUs. Live remote readiness belongs to the separate independent resource review. Runtime/package/hardware checks and strict tensor/condition/CFM replay must still pass; source equality alone does not prove cross-host bitwise continuation.

This report contains fresh v3-only execution conditions. Prior v2 mathematical/environment/cleanup evidence is inherited, not its host/output authorization. v2 output and attempt directories are absent at review; v1 failure and costs remain preserved. Only one v3 diagnostic with zero updates and zero new scores is approved, subject to Lead admission and resource approval. No automatic retry or formal training is authorized.
