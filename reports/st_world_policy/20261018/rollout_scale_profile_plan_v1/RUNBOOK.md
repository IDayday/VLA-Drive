# Conditional handoff — NOT_APPROVED

No launch is executed here. Existing reusable implementation:

- Runner: `tools/st_world_policy/train_flow_policy.py` in the final clean runtime.
- Legacy diagnostic launcher template:
  `/mnt/project/st-world-policy-20261016/resources/profile_flow_tsinghua_2x4_v1.py`.
  It is pinned to old6714/64-scene A/B runs: do not invoke `--replica A/B` for this
  study. Lead only adapts its single-job registration, locks, timeout and observer
  portions to the exact new source/config/namespace. No topology search.
- Legacy live CPU sampler:
  `/mnt/project/st-world-policy-20261016/resources/sample_profile_process_tree_v2.py`
  (`--profile-root`); reuse only after binding its bounded lifetime/output and
  fresh owned launch. The FORMAL4096 observer/launcher is not a drop-in substitute.
- Original recipe:
  `/mnt/project/st-world-policy-20261016/formal/consequence/registrations/C1_fulltrain_seed42_official_4096_v1/CONFIG.json`.
- Full original population:
  `/mnt/project/st-world-policy-20261016/audits/flow_policy/full_train_contexts_v2/TRAINING_MANIFEST.json`.
- Original S0 binding:
  `/mnt/project/st-world-policy-20261016/bindings/BASE_BINDING.json`.

Draft namespaces (do not create until admission):
`/mnt/project/st-world-policy-20261016/smoke/scene_center_rollout_std_v1_4update/`
and `/mnt/project/st-world-policy-20261016/profiles/scene_center_rollout_std_v1_fulltrain240/`.
Each owns CONFIG/REGISTRATION/LEASE and a distinct runs directory. Both initialize
original S0; the PROFILE must not initialize from the4-update SMOKE.

Configuration deltas from C1 are enumerated in PLAN.json. New method syntax must
match the final validator; there is deliberately no speculative runnable YAML.
No flags bypass review, missing artifacts or scope checks. The following actual
runner CLI is verified via `train --help`. Final paths must be assigned by Lead:

```bash
PY=/root/miniconda3/envs/ddp/bin/python
: "${RT:?Final clean integrated runtime required}"
: "${REG:?Exact admitted SMOKE or PROFILE REGISTRATION.json required}"
: "${CPUSET:?Fresh owned sixteen-CPU allocation required}"
: "${OUTER_SECONDS:?1200 for SMOKE or 7500 for PROFILE}"
cd "$RT"
export CUDA_VISIBLE_DEVICES=0,1,2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
# Use the same registered attention/determinism environment as C1; no new override.
"$PY" -m tools.st_world_policy.check_formal_approval --registration "$REG" --repo "$RT"
# Only under the reused, reviewed owned launcher holding dispatch/GPU/run locks:
timeout --signal=TERM --kill-after=180s "${OUTER_SECONDS}s" \
  taskset -c "$CPUSET" "$PY" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=4 \
  -m tools.st_world_policy.train_flow_policy train --registration "$REG"
```

The gate accepts bounded diagnostic registrations without manufacturing formal
approval. New independent implementation/resource review remains required by
campaign policy. Environment/hardware/parent/manifest hashes and launch receipt
must already exist before the command. A shell command alone is not admission.

After one complete PROFILE, the existing reducer uses all four rank files:

```bash
: "${RUN:?Actual completed PROFILE run directory required}"
"$PY" -m tools.st_world_policy.profile_pipeline \
  --output "$RUN/TIMING_SUMMARY.json" summarize --ranks \
  "$RUN/profile/rank0.json" "$RUN/profile/rank1.json" \
  "$RUN/profile/rank2.json" "$RUN/profile/rank3.json"
```

Only an explicitly reviewed **SMOKE** interruption may use the existing flags,
same REG/RUN and remaining total4-update/call/native budget:

```bash
# Same timeout/locks/affinity/environment as above; never a fresh init on resume.
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  -m tools.st_world_policy.train_flow_policy train --registration "$REG" \
  --resume --acknowledge-stop
```

Omit `--acknowledge-stop` when no operator stop marker exists; exact launcher
checks decide that from evidence. PROFILE timing explicitly rejects resume;
retain incomplete evidence and request a bounded decision. No run is launched,
no profile weight promoted, and no formal training command is issued here.
