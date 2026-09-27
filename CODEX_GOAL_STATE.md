# Structured World V1.1 signal rehabilitation — ACTIVE

Objective: reports/structured_world_v1p1/OBJECTIVE.md (new user attachment). Read it before continuation.
Base393c53b; isolated branch feature/structured-world-v1p1-signal-rehab-20260927.
Development worktree /mnt/project/VLA-Drive-structured-world-v1p1-20260927.
Immutable run-source worktree /mnt/project/VLA-Drive-v1p1-runs-d537401 at d537401c5029d80a84d0a9d3f77e275fe8c7e669. Running small-set experiments use this pinned source, not moving development HEAD.
New artifacts /mnt/project/structured-world-v1p1-artifacts/20260927; old artifacts /mnt/project/structured-world-v1-artifacts/20260926 remain immutable.

## Completed

P0/P1: complete12-bank20352-row read-only metric re-audit; independent maximum-cardinality geometry/class matching, legacy preserved; actual ego supervision/timing audit; original1696 PDMS untouched. append_tail preserves native prefix/mRoPE/action and trajectories EXACT on64 real scenes with actual gate0 adapter; legacy untrained insertion shifts up to6.88m. Reports METRIC_REAUDIT/INJECTION_SHIFT.

P2 implementation: reference residual heads, fixed-current-reference eligibility, separated GT displacement supervision, pre/post-Qwen head choice, bounded graph accumulation, locked wall-time/step budget, strict resume.14 targeted CPU tests passed. Real300-step checkpoint/target-independence/newgate0 action/serialization test passes. Actual optimizer+sampler resumption has run. Original VLM/vision/DiT all frozen; Reader/queries/project/current/motion heads update.

Matched16 diagnostic final checkpoint300: W_PRE57.6779% recall/35.5658% precision/55.4307% class recall; W_POST61.4232%/41.1028%/58.8015%. Each2400 presentations150 epochs. W_PRE consumed304, with4 interrupted extra updates charged; model300 used for pairing. W_POST phase ended300. Do not resume either16 phase; approved_end_step in ledger. First64 launches stopped at0 updates for conservative budget reconciliation, histories retained. New64 runs each696, combined family W_PRE1000/W_POST996 updates. **Both64 experiments still running, gate80%/50% NOT passed yet.**

Full capacity999 target cache targets_v6_full128 generated128/128,0 failures; all64 training target fields exactly match originalv6 (zero truncation). Holdout64 selected BEFORE evaluation by complete logs excluding57 overfit logs;59 holdout logs,zero overlap. train_log_holdout64_tokens.json is authoritative. Earlier token-only candidate holdout was never used for adaptation/validation.

P3 one real external prior: official Depth Anything V2 Large weight hash a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345 verified against HuggingFace LFS. Frozen pretrained visual backbone + NEW calibrated geometric BEV, NOT pretrained BEV.192 cache extractions total (first128 includes unused candidate holdout; second64 proper log holdout). Independent probe trains Reader/current heads600 updates, batch8,64 scenes,75 epochs; backbone frozen. Train recall52.7325% precision27.7497%, log-holdout recall23.6793% precision16.6085%. TESTED_INCONCLUSIVE. Real online prior→Reader→Qwen→adapter→DiT connectivity passed; gate0 native exact, blanking changes world hidden, nonzero gate changes actions. This is connectivity only, no planning benefit. Source, model-specific CC-BY-NC-4.0 weights license, pretraining data and cost in PROVIDER_AUDIT; upstream training compute not treated as zero.

Full-GT16 inference exports completed both variants; per-object failures/predicted NPZ;32 rendered PNGs (same16 scenes×2 variants) with geometric matches. Indexes in artifacts, no private images/data in Git.

## Live jobs — verify before waiting or resuming

- W_POST_64_bounded696_d537401: PID677139, accounted step322, GPU-hours0.3077 (live value will advance).
- W_PRE_64_bounded696_d537401: PID677138, accounted step627, GPU-hours0.3078 (live value will advance).

W_PRE64 tool session66235, PID677138, localGPU2. W_POST64 session38157, PID677139, localGPU3. Both terminal target696 steps; eval_every100 and final696. Do not restart while handles/processes live. Other experiment jobs completed:16 training, provider probe, source extraction, trained checkpoint checks, pretrained injection,16 eval, visualisation.
GPU0/1 placeholder restore launched in session46824, log idle_gpu01_restore.log; verify actual PID before pausing. Original local4–7 workers3516985..3516988 untouched. vla-zt2 resources available but unchanged this round. No unrelated tasks stopped.

## Immediate next work

1. Poll actual64 jobs and inspect fixed-step eval files. Do not redraw data, thresholds, tokens or initialization to pass the gate. Peak memory observed13.3GB PRE/17.6GB POST; batch8. Failures/aborted costs retained.
2. Once terminal, export full64 predictions, full-GT object CSV and stationary-motion comparison via tools/structured_world_v1p1/evaluate_world.py; use targets_v6_full128. Holdout evaluation only after the learnability gate, as objective requests. Need per-object/full-class failure analysis, dynamic/static ADE/FDE, motion coverage. No unmatched ADE=0.
3. If learning still insufficient, at most two documented repair rounds remain. Shared-query cls gradients in original64 traces exceed box gradients ~4–10x; do not blindly scale by hundreds. Actual300-step16 residual diversity is nontrivial, so do not simply assert identical-slot collapse. Per-task cosine/assignment churn and actual postclip norm were not fully recorded; add targeted diagnosis before claiming a specific failure cause. No LoRA unless W_PRE has learned and sufficiently trained W_POST fails.
4. P4 remains NOT_RUN and not yet implemented. Must only proceed after world learnability: independent current-only/current+motion pretraining each>=4×8192 exposures, then only P_CAPACITY/P_CURRENT/P_FUTURE with matched budget/provider/append-tail, frozen original driving model, gate0 step0 fidelity. Do not reuse small-set1000-step cap for long pretraining. Track real scene exposures, complete1696 dev PDMS fixed protocol, compare A0 and control, fixed intermediate/final results, optional shared velocity-field preservation only once if all regress. No navtest tuning, no RL/scorer/fullDiT updates.
5. Finish reports FOUR separate statuses, run ledger, commands/configs/checkpoint/resume/viz and final requirement-by-requirement audit. Stage commits exist, new branch NOT YET PUSHED; final push only task branch and verify remote SHA.

## Budget and recovery

New cap24000 optimizer updates and48 GPU-hours. Authoritative live budget_ledger.json with locks in NEW artifacts; snapshot REPORT RUN_LEDGER is not live. Initial isolation family aggregate cap1000 is explicitly enforced by the planned16+64 counts. Provider600 updates separate, all unsuccessful starts and GPU checks charged. Both convergence repair rounds unused. Old11639 steps sealed.

cd /mnt/project/VLA-Drive-structured-world-v1p1-20260927
export PYTHONPATH=.:/mnt/project/DriveVLA-M0/nuplan-devkit
export WORLD_PYTHON=/root/miniconda3/envs/ddp/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export DEPTH_MODEL_CKPTS=/mnt/project/DriveDreamer-Policy/depth_model_ckpts
export WORLD_PRETRAINED_VISUAL_WEIGHTS=/mnt/project/DriveDreamer-Policy/depth_model_ckpts/depth_anything_v2_vitl.pth

Only after verified terminal/accounted checkpoint (never while running):
CUDA_VISIBLE_DEVICES=2 "$WORLD_PYTHON" tools/structured_world_v1p1/resume_world.py --checkpoint /mnt/project/structured-world-v1p1-artifacts/20260927/W_PRE_64_bounded696_d537401/checkpoint.pt --worktree /mnt/project/VLA-Drive-v1p1-runs-d537401

Wrapper rejects behind-accounting checkpoints and spent approved16 phases. Full executed commands: docs/STRUCTURED_WORLD_V1P1_QUICKSTART.md. Base checkpoint and VLM are /mnt/project/DriveDreamer-Policy/models/{DriveDreamer-Policy,Qwen3-VL-2B-WorldAction}; raw logs /mnt/project/onevl_navsim_data/navsim_logs/trainval. No applicable AGENTS.md found in worktree/parents; optional detailed V1P1 specification not found in attachments/worktree.
