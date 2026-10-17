# Independent review of bounded provider PROFILE admission

**PASS_CODE_FOR_BOUNDED_PROFILE**, tested exact integrated commit
`2d5e9483a8c8bc80b71348211756f3cab6aa21b1`.

This review covers implementation `3b65594193a5b7ec336420c4592d903ead71ec17`
plus Lead's two-line registered-manifest check in `train_world.run`. It does not
dispatch a PROFILE, reproduce the completed real exact32 composition or student
smoke, approve resources, or grant FORMAL training. The reviewer changed only
this new report/test directory in an isolated worktree.

## Actual tests

| Independently executed scope | Result | Seconds |
|---|---:|---:|
| Implementer's new `test_provider_profile_upgrade.py` | 21 passed | 6.06 |
| New independent behavior and negative cases, first run | 17 passed | 13.25 |
| Additional PROFILE source-drift test plus two relevant old paths | 3 passed | 4.88 |

These are **41 distinct cases**, including 18 reviewer-owned cases and two old
path regressions. No failures or skips occurred. Complete logs are preserved;
the existing pynvml deprecation warning is not a test failure. The implementer's
previous 108-test report is not claimed as this review's execution.

All commands used the existing `ddp` Python, CPU affinity 0,1, two CPU library
threads, empty `CUDA_VISIBLE_DEVICES`, no bytecode writes and a 120/180-second
external timeout. Only tiny temporary synthetic metadata, SQLite and tensor
fixtures were used. No real database, image, raw log, teacher weight, large
tensor, production descriptor/PLAN, ledger or GPU was opened or changed. Test
durations above are measured pytest durations, not a memory or throughput
qualification. This bounded review completed within its ten-minute window.

## Findings and acceptance boundaries

1. **Old SMOKE remains strict.** The default composition plan validator still
   requires the original exact <=32 population, source pins, all cases and all
   numerical evidence. A historical control-source plan fails by default; no
   `skip_source_check` API exists. An explicit PROFILE admission cannot be
   relabeled as SMOKE. The old current-code composition/factory fixture passes.

2. **Source migration is explicit, not a resume escape.**
   `validate_control_migration` permits exactly three named control files and
   requires all 15 numerical/reader/PCA/sampler files to match the archived
   commit. It compares the archived bytes to actual `git show <commit>:<path>`,
   checks the original parent descriptor/PLAN/result and inventories current
   bytes. The independent negative cases reject an omitted math entry, wrong
   old/new control hash, and a modified archived mathematical file even after
   its metadata is re-pinned. The true `python -m ... prepare-profile-migration`
   CLI produces only the same explicit proposal, with both qualification and
   formal flags false. It neither runs qualification nor creates approval.

3. **The numerical implementation is unchanged.** `SOURCE_SCOPE.json` records
   all 15 protected source files byte-identical to real qualified runtime
   `31c3dca2fbc697b419bf343ce05d10be0a4fe13f`. AST comparison also confirms no
   changes to `make_plan`, `qualify_layouts`, `run`, `Budget`, `prior_evidence`,
   `_full_difference` or `_checked_summary`. The consumer's only new control
   argument selects the reviewed explicit migration validator; full comparison
   and accounting checks remain intact. The trainer's two added lines are
   reviewed separately and remain covered by the registered whole-source
   fingerprint, not hidden inside that three-file migration list.

4. **PROFILE population and recipe are genuinely bounded and bound.** A new
   descriptor carries its own source migration, original exact32 descriptor,
   selected manifest, request index, reference recipe and full proposed recipe.
   It retains the qualified current source, fixed-eight FP32 execution,
   original target times/views, raw contract, LRU limits and PCA basis. Scene/log
   order must equal the selected data manifest, whose current/ego hash fields
   must be present. A synthetic real indexed population of 256 is accepted;
   257 is rejected before publishing the provider descriptor. The CPU fixture
   does not claim real 256-scene target qualification.

5. **Actual recipe changes are checked, not just the YAML pin.** Original model,
   losses, optimizer, PCA, data roots, seed, precision and mathematical recipe
   remain fixed. The 20,000-update schedule and 1,000-step warmup are retained;
   finite invocation stop/sample counts are <=256. Explicit allowed diagnostic
   topology/logging/budget settings are fully recipe-pinned and still pass the
   installed official execution resolver. Independent actual `validate_config`
   calls reject mutations of LR, input root, patch count, precision, seed,
   workers and FORMAL scope. The PROFILE path requires synchronous workers=0.

6. **Lead's manifest check is on the real trainer path.** The new call occurs
   after the exact registered teacher-descriptor check and before provider
   preview/dataset construction. An independent CPU test invokes the actual
   `train_world.run` twice, once fresh and once with `resume=True`; a wrong
   registered manifest pin is rejected before either provider/dataset is
   reached. For this narrow test only, outer resource/GPU/distributed setup is
   stubbed. This is a call-chain test, not real DDP or training evidence. A
   different path containing identical manifest bytes is also rejected: both
   path and SHA must match the admitted manifest. Actual current/ego file-byte
   checks remain in the unchanged dataset, and their real execution is not
   repeated here.

7. **Resume source checks are preserved.** Every run, including resume, first
   calls the unchanged registration/source gate. An independent temporary-git
   PROFILE test passes its original registration, rejects uncommitted source
   drift, then rejects the same drift after commit. In the real trainer source,
   run identity still includes registration SHA, source SHA, config, dataset,
   precision and Base binding. Resume compares that complete identity and
   selects checkpoints with the same signature before strict native loading.
   This extension does not authorize loading an old SMOKE checkpoint into a
   new PROFILE identity. The AST/source checks are stated separately from
   actual native resume evidence, which was not rerun.

8. **New population evidence is not overstated.** PROFILE descriptors and
   configuration lineage explicitly retain
   `new_population_tensor_qualified=false` and `formal_approved=false`. The old
   receipt validates only its original exact population. The code does not
   fabricate a new per-tensor receipt for 65/256 scenes. Actual PROFILE targets,
   resource costs, data waiting, multi-rank behavior and interruption/resume
   need measurement on the newly registered run.

9. **FORMAL remains rejected for this provider.** PROFILE cannot be promoted by
   relabeling its scope, and diagnostic weights remain excluded as formal
   initialization. The old independent formal-review and source/config/data/
   environment/resource registration mechanisms are unchanged. No resource or
   launch approval is created by this review.

## Source hashes

| File | SHA256 |
|---|---|
| `teacher_provider.py` | `bca3975e6d9e22b83746e6a366a2e99b5bb49a3b76a0704353d373e7da5aa8df` |
| `qualify_indexed_provider.py` | `6fd0a7b9d3317f78aa139906f20ff81b97a7bb7bee3a397a038a3553aaa5fc08` |
| `training_config.py` | `87cb75dc7c948a8c37911fd2037794a48a50a5aae6d8153b867eaf0dc32ced97` |
| `train_world.py` | `83c9bdbfeac76ec44ad9e13164eb83ed0d1aebd67bc62f02f6b78eac44a5b689` |

All paths are under `tools/st_world_policy/`. Full source scope is recorded in
`SOURCE_SCOPE.json`.

## Reproduction

From an isolated checkout of the tested commit with these report tests present:

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  timeout --signal=TERM --kill-after=5s 180s taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python -c '
import os,sys
sys.path[:0]=[os.getcwd(),os.getcwd()+"/tests/st_world"]
import pytest
raise SystemExit(pytest.main(["-vv","-p","no:cacheprovider",
"tests/st_world/test_provider_profile_upgrade.py",
"reports/st_world_policy/20261017/correctness/provider_profile_upgrade_v1/test_independent.py",
"tests/st_world/test_indexed_composition.py::test_actual_adapter_layouts_masks_rng_and_factory_admission",
"tests/st_world/test_approval.py::test_diagnostic_never_approves_or_exports_formal_initialization"]))'
```

The three actual separate runs and their durations are the evidence above; this
combined command is the equivalent reproduction, not a claimed fourth run.

## Decision for Lead

The reviewed code is suitable to proceed to a **new registered bounded PROFILE**
using either the same exact32 population or an independently admitted ordered
population of at most 256 scenes, with at most 256 updates. A 32-scene timing
run does not establish broader-population throughput. For 256 scenes, real
future metadata/index completion, storage admission and fixed selection must
exist first; this review does not create them. Each run still needs its actual
source/config/data/runtime/resource registration and available authorized
resources. The existing 4-update SMOKE path is not silently expanded.

No FORMAL/full-population/long-throughput/scientific PASS is granted. The
remaining eager student dataset metadata path is unchanged and its per-rank
memory cost must be measured for the proposed population. Prior real exact32,
two-update smoke and original failure evidence remain untouched.
