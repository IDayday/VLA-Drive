# DDP vehicle-only from-scratch campaign — CORRECTED LEARNING RUNNING

Branch: feature/ddpolicy-vehicle-joint-from-scratch-20260928
Reference base:632cf74c4c7269228d66d569e0c54846b07c95f4
Development worktree:/mnt/project/VLA-Drive-ddpolicy-vehicle-joint-20260928
Artifacts:/mnt/project/ddpolicy-vehicle-joint-artifacts/20260928
Immutable corrected training source:4843e4ddf8340ecc8b44a47fc5fc9b688a58e3a8
Training worktree:/mnt/project/VLA-Drive-ddpolicy-optimizerfix2-20260928
Remote verified through4843e4d. Subsequent report/controller commits are separate from training provenance.

Authorization: full public-generic/random-driving A/DDP-Base, B/vehicle Joint, C/Joint+role auxiliary; original Qwen3-VL-2B/Wan/PPD framework. No released driving weights, historical foundation/Reader/graph/scorer/hidden caches, old controllers, new BEV, nonvehicle prediction, RL or second executed ego. Old V3 evidence remains sealed. Ego is joint slot0; only its standard decoded trajectory executes.

Current runs (read actual status before restarting):
- small_fit_fixed_A_seed42_001: local GPU0/1, active.
- small_fit_fixed_B_seed42_001: local GPU4/5, active.
- small_fit_fixed_C_seed42_001: vla-zt2 GPU1/2, starting/active.
Each is a NEW generic/random model, fixed64 training scenes, globalbatch16/microbatch2,512updates,lr1e-5,warmup32/horizon512,final64all-hidden. Main FM every update; B extra all-hidden/C role every4 at0.1. Head xy scale1; the provisional scale20 hypothesis is NOT selected. All formal models will restart independently from generic/random initialization.

CRITICAL historical correction: small_fit_A/B/C_seed42_001 paused at179calls each, scale-B paused at34. startup_B_2gpu_001 also had4no-op calls. All are INVALID_OPTIMIZER_STASIS and must not be resumed or counted as learning. The installed FusedAdam metadata uses signed32-bit tensor lengths, while one two-GPU ZeRO partition had2.35billion elements. GPU counterexample:2147483904elements do not update; splitting into1073741952-element tensors updates correctly. Raw ledgers/cost/exposure remain unchanged; validity overlay is optimizer_stasis_correction_v1/validity_overlay.json and the public reports/optimizer_stasis directory.

Correction: identical-hyperparameter optimizer groups bounded at500M elements, with exact integer sampling and an actual FP32-master change check on EVERY step before progress increments. No shared environment was patched. Full two-GPU C4 startup now completed4verified updates; same-device2+2 also completed4verified updates. Full FP32-master comparison completed: all RNG states equal, max difference4.95e-6, predeclared tolerance FAILED. Do not claim exact resume. startup_fixed_C_sameparent_001 repeats steps3/4 from the exact same saved step2 to diagnose independent-prefix variation; its inherited2updates are not charged as new updates. The first corrected startup attempt was safely paused at0before updates to fix probe-index rounding. Tests:25CPU passed,1CUDA-marked test skipped in that run; prior explicit CUDA RNG test passed. The true oversized FusedAdam counterexample also ran on GPU.

Data:103288 raw navtrain scenes/1192logs; fixed whole-log split train101592/1176logs,dev1696/16logs, zero intersections. Full vehicle-only GT identity59a0f36f22f33cc0e10bc82328191d126a9acfdc89b8c59b3931873ec212a423;0failures. Raw source population is available; vehicle filter precedes capacity32. Current-only dev1696,Navtest12146/136logs,train64 caches complete. Generic depth covers all103288 tokens. No Navtest model prediction has occurred. Two-scene dev scorer smoke PDMS0 was a near-random startup, never a baseline result.

Resources: existing local/vla-zt2 GPUs only; vla-zt2GPU0/3 remain other tasks. New full campaign cap8000GPUh registered autonomously after optional cap preference had no reply; user changes override it. No expansion/rental. Bounded diagnostic allowance40GPUh includes failed/no-op runs, startup, extraction, tests and evaluation. Old48GPUh quota does not apply. Initial20GPUh diagnostic registration is preserved as history.

Formal plan: A/B/Cseed42 and B/Cseed43,100000updates/globalbatch32,AdamW1e-5,warmup5000,cosine minimum5e-7,final10000all-hidden for B/C. Configs exist at formal_configs_v1, generated from immutable4843e4d. One pass3175updates with24scene tail;100000updates=3199752scene presentations/run. Prior8GPU valid-size startup measured6.6–7.0s/update; five runs extrapolate about7500GPUh, not a completion-time promise. Formal training is NOT_RUN until the corrected real learning diagnostic is checked. Full dev and Navtest results are NOT_RUN. Small fits are not the full experiment.

Next: finish corrected resume comparison; inspect actual learning/graph/role counts and train64 predictions; continue matched512-step diagnostics; start fresh formal models with the frozen source/config and bounded controller; complete fixed dev checkpoint grid, freeze selection, then full v1 Navtest with seeds42–46. Full-source controller and offline evaluation helpers are committed separately; do not duplicate existing jobs. Fixed step64 train64 inference completed: A egoADE7.317m, B10.037m, no failures; B currently0/333 GT vehicles within the fixed2m evaluation gate. Training later begins selecting predicted vehicles and has nonzero joint vehicle supervision. These are learning diagnostics, not full-training or PDMS results.

Read-only progress:
python -c 'import pathlib,json; r=pathlib.Path("/mnt/project/ddpolicy-vehicle-joint-artifacts/20260928/training"); print([(p.parent.name,json.loads(p.read_text())["status"],json.loads(p.read_text())["real_optimizer_updates"]) for p in sorted(r.glob("*/status.json"))])'

Commands:docs/DDPOLICY_VEHICLE_FROM_SCRATCH.md. Never resume invalid/no-op diagnostic IDs. Preserve scene images, weights, caches and raw data outside git.
