# Numerical and resume evidence

The original normal-CUDA4-update versus2+2 real-cached-sample audit did not pass
bitwise loss/gradient/model/Adam equality. Its sample IDs, two sampler cursors,
Python/NumPy/CPU/CUDA RNG and scheduler matched. The first tiny loss difference
(about4.17e-8) appeared at step2 before interruption. Step4 parameter max
difference was2.78e-8, Adam max3.73e-9, and update cosine approximately
0.99999999994. Preserve REAL_RESUME_AUDIT.json (FAIL) and
NONDETERMINISTIC_RESUME_DIFFERENCES.json; do not relabel them PASS.

A separate deterministic audit used CUBLAS_WORKSPACE_CONFIG=:4096:8,
deterministic algorithms, TF32 off and cuDNN benchmark off. It kept ordinary
training dropout, real candidate labels, the two sampling streams and original
2,000-step learning-rate timeline. Continuous4 and interrupted2+2 updates
matched exactly for module, Adam, scheduler, RNG, samplers, consumed IDs,
losses and gradient norms. REAL_RESUME_DETERMINISTIC.json is PASS. This setting
was not applied retrospectively to production weights or the main run.

The full main workflow actually recovered expert250→1000 and scorer334→342→500
→2000. Best snapshots remained the historical stage_val winners. C1/C2/C3,
terminal resume and uneven-rank-denominator tests remain in the full regression.
The actual NCCL optimizer-boundary and terminal tests ran before those GPUs
were reassigned to the user's restored Base-cache production.

The host reader migration checks every retained tensor byte/dtype/shape and
does not consume RNG. HOST_READER_EQUIVALENCE.json passes on64 real scenes.
The32-image original-Base/K1 and raw W/A rebatch audit reports zero differences.
The final scorer bundle strictly reloads to bitwise equal output on identical
cached context. The separate online batch1 versus cached batch4 scorer comparison
has max predicted-score difference1.0377e-4; it is not a proof of identical
online choices across an entire dev bank. No tolerance was retroactively widened
and no formal serving bundle was activated.
