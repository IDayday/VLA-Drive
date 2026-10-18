# Independent gradient v2 environment repair review

APPROVE_GRADIENT_BALANCE_EXECUTION, correctness only, subject to fresh independent v2 resources approval and Lead admission. Mathematical runtime remains `83679937`; wrapper SHA256 `29556ded05f532cabc7f4fa8255668f7ab232341f7ce34a6f5fa5fa5650f76ae`; registration SHA256 `fb55ca585688cd2ca19ea55af55628c60c91554c00f68ef4a6ae1821d22b06ec`.

The narrow fix pins `CUBLAS_WORKSPACE_CONFIG=:4096:8` in parent, passed child environment, child-before-torch preflight, admission and review bindings. Missing,wrong or late setup is refused. Determinism is not disabled. Original launcher source explicitly set this value beforePopen; the original trainer requires explicit deterministic workspace. Original LAUNCH does not dump the environment, and no such history is invented.

Seven independent CPU tests passed in0.26seconds: three wrong/missing parent environments refuse before control reads; child environment preserves unrelated numeric flags without mutating parent; late torch import refuses; actual child entry is intercepted at its first torch import only after a valid immutable environment receipt; unchanged cleanup/accounting AST and output-only registration delta verified. No actual torch/GPU/model/GT/image/cache/scoring execution occurred. Eight original source/small receipt hashes were independently verified.

The first actual attempt remains DIAGNOSTIC_REJECTED with zero updates/scores and zero live owned children. Its362.422seconds and73,834,874,003 admitted bytes remain recorded, alongside its original failure reason. This is execution setup failure, not an observed numerical mismatch. No history is overwritten. The new namespace is v2 only; checkpoint3584,samples,BC/FPO math,RNG replay and budgets remain fixed.

Lead must start the Python process with the exact environment value, provide freshly bound reviews/admission and run at most this one repaired attempt. The unchanged observer limits/owned cleanup and separate BF16-gradient interpretation remain applicable. Actual v2 GPU replay is NOT_RUN; this review does not certify the yet-unseen gradients, training gain or jointZeRO2 update.
