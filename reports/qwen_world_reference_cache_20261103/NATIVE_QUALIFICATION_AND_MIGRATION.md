# World8 native qualification and an explicit execution migration

This v2 adds CPU planning/comparison tools only. It does not change the v1 cache,
the formal trainer, scientific config, owners, resources or any live job.

## Qualification: three operator-dispatched jobs

After freezing `P/runtime/reference_cache_probe_v2`, prepare a new output:

```sh
python -m tools.qwen_world_performance.reference_cache_native_qualification prepare \
  --source /mnt/project/qwen-world-performance-rl-20261103/runtime/reference_cache_probe_v2 \
  --root /mnt/project/qwen-world-performance-rl-20261103/qualification/REFERENCE_CACHE_WORLD8_NATIVE
```

`PLAN.json` contains exact torchrun argv, dependencies and expected transport
RESULT paths. The tool does not dispatch, select hosts, allocate a GPU or contact
SSH. The operator admits the same real eight GPUs for each sequential job:

1. `original4`: existing `tools.qwen_world_performance.train --qualification`,
   exact B128/G16/M4, same complete IL70 and fresh Adam/RNG42, stop4/save2,
   cache **off**. It runs through2 without restarting.
2. `cached2`: same cfg/seed/IL70 and fresh independent Adam0, cache **on**,
   stop2/save2. Actual native `update000002_inner2` must be fully sealed.
3. `cached_resume4`: same cached run/output/source/flag, exact native
   `--resume .../update000002_inner2`, stop4. Built-in restore rechecks COMPLETE,
   files, transaction, every rank, pending ancestry/cache, Adam/scheduler and RNG.

Both paths consume the same128 genuine TRAIN scenes and2048 primary candidates.
They are qualification artifacts, never formal initialization or NAVTEST scores.
Three exact transport RESULT returncodes0 are required. If an operator chooses
other job directories, set `--job-root` at prepare so PLAN retains exact paths.

Run the PLAN `compare_command` after all jobs exit. It independently verifies
all files and the real native validators at original/cached inner2 and final4;
checks LoRA+memory parameters, full Adam/scheduler tensors, all8 saved RNG/data
streams and cursors; compares pending mathematical material without own-run
labels/cache payloads; checks every actual loss/gradnorm/LR journal row; and
requires each rank's actual invocation to prove inner2→4 restore. Cached pending
leaves must all contain valid frozen-reference constants; uncached leaves must
not. Timing fields and own-run labels are not claimed bit-exact.

## Minimum production migration: root/operator only

1. Keep the actual single-GPU full-IL70 PASS and actual world8 native PASS sealed.
   No CPU mock or missing result qualifies. Keep formal_v1 immutable.
2. Cooperatively save each existing formal run at an exact native boundary;
   verify actual exit0/complete8rank pending checkpoint. Keep all Adam moments,
   scheduler exposure clock, rank RNG/generator, scene cursor, bank and pending
   behavior/logp/A. Never restart from IL70 or import qualification weights.
3. From the new frozen source, construct that run's **actual formal** new
   identity through the same full actor/config/metadata path. Preserve every
   `scientific_identity` field. Record cache enablement only under
   `execution_optimizations`; do not copy a qualification identity or invent
   parameter closures. Use the source exactly bound by world8 evidence.
4. `migration-draft --old-identity ... --new-identity ... --checkpoint ...
   --evidence SINGLE_GPU_COMPLETE --evidence WORLD8_COMPLETE --output DRAFT.json`
   seals exact identities, checkpoint inventory and evidence hashes, and invokes
   the existing `verify_migration`. It only writes a **DRAFT_NOT_APPLIED** receipt.
   It never changes registry/argv, signals a process or launches training.
5. After explicit root review, resume the same formal output with the full old
   checkpoint, `--migration-receipt DRAFT.json --frozen-reference-cache` from that
   exact tested source. Built-in validation runs before binding: old pending
   cache absence may then be filled with same-run frozen constants; present
   invalid cache is rejected. Old source density/advantage and fresh-ratio1e-4
   checks stay untouched. Observe all8rank advancement and seal a new boundary.

The full-model singleGPU v1 probe has actually passed; the v2 world8 probe is
prepared only until the operator dispatches it and `compare` produces PASS.
