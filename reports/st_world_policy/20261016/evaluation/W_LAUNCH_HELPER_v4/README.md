The v4 candidates repair the demonstrated missing W launch binding in the fixed full-dev helper. Original SP0 and SP2 intents omit training_binding.launch, while v3 accesses that key unconditionally. SP1/SP3 already carry exact launch pins. All six original C0/C1 intents have launch pins; the existing C2/C3 adapters are unchanged.

W_LAUNCH_BINDINGS.json is a new immutable supplement containing only actual SP0–SP3 registration/config/approval/launch references, approved fingerprints and original runtime identities. It does not rewrite the original ten-job files or bind invented checkpoints. Each helper adds the missing launch field to an in-memory copy of the W intent; SP1/SP3 retain their existing value. The endpoint registration records the supplement pin, and scoring requires that same normalized W intent and pin.

Training stays on fdf25f99b702754787904215bf1c2a2b7515ea75 at the existing immutable official runtime. Evaluation stays on 6714b1e9fb2e0c8059b90b8b4d48cd40796fe345. Historical W LAUNCH.json has no source_sha field; its exact pinned bytes, cwd, registration source, approved fingerprint and current immutable training-runtime HEAD establish that binding. The adapter never substitutes evaluation6714 for training fdf25f99. Actual launch argv, output run, arm and approval gate must match. The original large W identity is retained by reference and is not read by the supplemental adapter.

C jobs return unchanged from the new adapter before any W control read. Existing route, native, population, source, score, call-budget and lock functions remain unchanged. Full dev1696/16 logs, seed42, world1/rank0, FP32 master/TF32-off/SDPA, one Euler10 trajectory, total1728 scoring calls, two workers, existing invocation bounds and campaign concurrency all remain fixed. No original command/config/source/runtime file or deployed v3 helper was modified.

CANDIDATE_VALIDATION.json records 54 passing CPU checks in 1.573 seconds: original SP0/SP2 KeyError reproduction; all four actual W launch/approval/source bindings; missing/wrong launch, arm, source, fingerprint and input-config rejection; unchanged native/route/budget/C-adapter definitions; and all twelve existing C jobs unaffected. Both --help calls exited zero. Validation extracted metadata functions and permitted only read-only git subprocesses. It did not execute launcher main, GPU queries, model/native payload reads, exports, scoring or training.

Independent review is required before Lead deployment. The author does not sign an efficiency/correctness role for this implementation. Exact candidate/baseline/diff/test hashes are in HANDOFF.json. A reviewer can preserve the owner test receipt by selecting a separate output:
```sh
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python /mnt/project/VLA-Drive-st-evaluation-agent-20261016/reports/st_world_policy/20261016/evaluation/W_LAUNCH_HELPER_v4/VALIDATE_CANDIDATES.py --output /absolute/new/reviewer_validation.json
```
That path must not already exist. No main command is run by this validation.

SP0_READY_ENDPOINT.json independently records the now-real tag step_005000_1792217229597178910. COMPLETE receipt SHA256 is 1583c98960120ad2cde5859919507386b27259c4b3524a82b40de6ce0f1299fe. Exact native inventory and file sizes match, totaling 37,366,035,746 bytes; no payload checksum was recomputed. Training reports PAUSED / BUDGET_PAUSED, completed5000 and exposure159992 on its original20k clock. This later observation preserves the prior pre-completion snapshot in the separate C2 READY commit.

After independent review/deployment and an actual immutable route, Lead may use the unchanged argument interface with this exact endpoint:
```sh
: "${GPU_INDEX:?Actual reviewed route GPU required}"
: "${ROUTE_SHA256:?Actual immutable route hash required}"
/root/miniconda3/envs/ddp/bin/python /mnt/project/st-world-policy-20261016/resources/launch_formal_dev_export_v4.py --job-id SP0_step005000_dev1696_seed42_v1 --checkpoint-tag step_005000_1792217229597178910 --gpu "$GPU_INDEX" --route-sha256 "$ROUTE_SHA256"
```
Scoring uses launch_completed_dev_score_v4.py with the same job ID/route only after full1696 completion and exporter exit. These are documentation templates, not dispatch. No route or launch approval is created here. GPU3's reported reservation is not a fresh resource qualification.

Learned-W extraction, tensor storage claims, probe fits and their distinct qualification remain outside this patch. Actual export success, throughput and scientific results remain unmeasured by this work. Lead-reported C0/C1/C2 dispatches are not repeated.

Parent commit: f4e28993e619f0f137719150c197a81e131e087e. Owned path: reports/st_world_policy/20261016/evaluation/W_LAUNCH_HELPER_v4 only. Existing untracked smoke_dev16_aa remains untouched.
