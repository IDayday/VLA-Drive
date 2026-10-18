# Native SP2/SP3 20k registration and approval handoff

Preparation only. No registration, approval, lease, process, or checkpoint is created by this review. Runtime remains `/mnt/project/VLA-Drive-st-official-runtime-v2-20261016`, commit `fdf25f99b702754787904215bf1c2a2b7515ea75`. The two fixed config proposals and parent pins in `PREPARATION.json` are unchanged.

The existing tools do not expose a `create_registration` or `approve` CLI. Actual registration was built by `resources/register_spatial_formal_v1.py` and `register_spatial_tsinghua_formal_v1.py`, using `atomic_json`, `source_fingerprint`, and `verify_registration`. Those scripts bind specific historical hosts and stop5000. Do not execute them for this continuation. The following commands reuse their original APIs; they do not create a second trainer or change the native gate.

## Registration construction, after the actual lease exists

Run separately for `SP2` and `SP3`. `SPATIAL20K_LEASE` and its SHA must identify the actual new lease prepared by systems; no historical placement is substituted. Keep the old implementer list, including root/cache_budget. Any additional author of the new registration or runtime binding must be disclosed, not removed to pass review independence.

The command hashes the registered metadata through the original gate, including the approximately30MB scene manifest. It does not read native `.pt` or teacher payloads. It creates a new immutable registration directory and will fail if that directory already exists; preserve partial evidence and investigate rather than overwriting it.

```bash
export SPATIAL20K_ARM=SP2
: "${SPATIAL20K_LEASE:?actual new four-GPU lease path}"
: "${SPATIAL20K_LEASE_SHA:?actual lease SHA256}"
cd /mnt/project/VLA-Drive-st-official-runtime-v2-20261016
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python -B - <<'PY'
from copy import deepcopy
import json, os
from pathlib import Path
from tools.st_world_policy.check_formal_approval import file_hash, source_fingerprint, verify_registration
from tools.st_world_policy.training_config import validate_config
from tools.st_world_policy.budget_extension import _recipe
from starVLA.dataloader.st_world_targets import atomic_json

repo = Path('/mnt/project/VLA-Drive-st-official-runtime-v2-20261016')
prep = Path('/mnt/project/VLA-Drive-st-temporal-closeout-independent-20261017/reports/st_world_policy/20261018/spatial20k_native_continuation_preparation_v1')
assert file_hash(prep/'PREPARATION.json') == 'f288662eb29913c688733dea30ba4cfe83c5a5ee6bb707cca7c14e97d5cc7599'
arm = os.environ['SPATIAL20K_ARM']
assert arm in ('SP2', 'SP3')
entry = json.loads((prep/'PREPARATION.json').read_text())['arms'][arm]
def checked(ref):
    p = Path(ref['path'])
    assert p.is_absolute() and file_hash(p) == ref['sha256'], str(p)
    return json.loads(p.read_text())
parent = checked(entry['parent_registration'])
cfg = checked(entry['config'])
receipt = checked(entry['parent_receipt'])
parent_cfg = checked(parent['artifacts']['config'])
validate_config(cfg)
assert _recipe(cfg) == _recipe(parent_cfg)
assert cfg['execution']['stop_after'] == cfg['execution']['max_updates'] == 20000
assert cfg['execution']['world_size'] == 4
assert receipt['state'] == entry['parent_state'] and receipt['state']['completed'] == 5000
lease_path = Path(os.environ['SPATIAL20K_LEASE']).resolve()
lease = checked({'path': str(lease_path), 'sha256': os.environ['SPATIAL20K_LEASE_SHA']})
assert lease['schema'] == 'st_training_resource_lease_v1' and lease['world_size'] == 4
assert lease['max_invocation_seconds'] >= cfg['runtime']['max_seconds']
assert source_fingerprint(repo) == parent['source_fingerprint']
out = Path('/mnt/project/st-world-policy-20261016/formal/spatial/registrations')/entry['proposed_run_id']
assert not Path(entry['new_output_if_registered']).exists()
out.mkdir(parents=True, exist_ok=False)
atomic_json(out/'CONFIG.json', cfg, immutable=True)
reg = deepcopy(parent)
reg.update(run_id=entry['proposed_run_id'], execution=cfg['execution'])
reg['artifacts']['config'] = {'path': str(out/'CONFIG.json'), 'sha256': file_hash(out/'CONFIG.json')}
reg['artifacts']['resource_lease'] = {'path': str(lease_path), 'sha256': file_hash(lease_path)}
atomic_json(out/'REGISTRATION.json', reg, immutable=True)
fingerprint = verify_registration(out/'REGISTRATION.json', repo=repo)
atomic_json(out/'FINGERPRINT.json', fingerprint, immutable=True)
print(json.dumps({'status': 'REGISTERED_PENDING_THREE_REVIEWS', 'registration': str(out/'REGISTRATION.json'), 'fingerprint': fingerprint}))
PY
```

This preserves the parent's environment artifact. Actual host execution must satisfy the original `environment_record()` equality. A genuinely different installed environment is a changed dependency, not something to hide by writing the old JSON again. The new lease must cover the complete bounded invocation and explicitly account for the 15000 additional updates, checkpoint storage, current dispatch policy, UUIDs, and authorization. `maximum_optimizer_updates_this_stage` is descriptive in the old lease; the actual hard mathematical limit remains registered `stop_after=20000` and original max20000.

## Three independent review receipts, then assembly

Each reviewer binds the **new** fingerprint and cites the old immutable role receipt plus the narrow changed-scope evidence. Old PASS documents are inherited evidence, not valid new-fingerprint approvals. The unchanged implementation evidence can be reused without another model qualification. Evidence/correctness/efficiency identities must be mutually distinct and absent from `implementer_agents`. In particular, `/root/st_cache_budget` is listed as an original implementer; its resource analysis can support another independent efficiency reviewer, but it cannot sign the native efficiency role for this registration.

The original native checker validates all referenced artifact hashes. This command assembles existing genuinely signed receipts; it never manufactures reviewer PASS fields. Only the Lead executes it after those files exist. A failed check leaves the immutable proposal for diagnosis and produces no final approval.

```bash
: "${SPATIAL20K_REGISTRATION:?actual new registration path}"
: "${SPATIAL20K_EVIDENCE_REVIEW:?actual independent evidence receipt}"
: "${SPATIAL20K_CORRECTNESS_REVIEW:?actual independent correctness receipt}"
: "${SPATIAL20K_EFFICIENCY_REVIEW:?actual independent efficiency receipt}"
cd /mnt/project/VLA-Drive-st-official-runtime-v2-20261016
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python -B - <<'PY'
import json, os
from pathlib import Path
from tools.st_world_policy.check_formal_approval import file_hash, verify_registration, check_formal_approval
from starVLA.dataloader.st_world_targets import atomic_json
repo = Path('/mnt/project/VLA-Drive-st-official-runtime-v2-20261016')
reg = Path(os.environ['SPATIAL20K_REGISTRATION']).resolve()
reviews = {}
for role in ('evidence', 'correctness', 'efficiency'):
    p = Path(os.environ['SPATIAL20K_'+role.upper()+'_REVIEW']).resolve()
    reviews[role] = {'path': str(p), 'sha256': file_hash(p)}
body = {'schema': 'st_world_approval_v1', 'fingerprint': verify_registration(reg, repo=repo),
        'reviews': reviews, 'formal_launch_allowed': True}
proposal = reg.parent/'RUN_APPROVAL_PROPOSAL.json'
atomic_json(proposal, body, immutable=True)
checked = check_formal_approval(reg, proposal, repo=repo)
assert checked['status'] == 'VERIFIED'
approval = reg.parent/'RUN_APPROVAL.json'
atomic_json(approval, body, immutable=True)
assert check_formal_approval(reg, approval, repo=repo) == checked
print(json.dumps({'status': 'APPROVAL_VERIFIED_NOT_LAUNCHED', 'approval': str(approval), 'sha256': file_hash(approval)}))
PY
```

Actual standalone check:

```bash
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python -B \
  -m tools.st_world_policy.check_formal_approval \
  --registration "$SPATIAL20K_REGISTRATION" \
  --approval "$SPATIAL20K_APPROVAL" \
  --repo /mnt/project/VLA-Drive-st-official-runtime-v2-20261016
```

## Exact native argv and existing lock ownership

The existing `launch_reviewed_spatial_v1.py` hardcodes stop5000, no prior output/LAUNCH, and a first-wave `4*240GiB` reserve. It is not an approved20k launcher. Lead's versioned local binding should change only these stage/explicit-approval/resume/resource inputs, preserving the actual current-policy, host, environment, GPU, lease, duplicate-run, source-clean, timeout, and lock checks.

After those checks the actual first invocation argv is:

```python
argv = ['timeout', '--signal=TERM', '--kill-after=180s', '29400s',
        '/root/miniconda3/envs/ddp/bin/python', '-B', '-m', 'torch.distributed.run',
        '--standalone', '--nnodes=1', '--nproc_per_node=4',
        '-m', 'tools.st_world_policy.train_world',
        '--registration', str(registration_path), '--approval', str(approval_path)]
```

First20k invocation has **no `--resume`**: `config.continuation` explicitly resolves the parent5000 snapshot. Subsequent invocation appends `--resume` and selects the newest complete checkpoint in the **new20k run**, never the parent5000 directory. Only an explicit acknowledged operator STOP request justifies the separate `--acknowledge-stop` flag.

Preserve the old per-GPU lock mechanism: open `/tmp/foresight_gpu_<physical_index>.lock` for the four exact leased IDs, `fcntl.flock(fd, LOCK_EX|LOCK_NB)`, retain every FD through launch and pass them in `Popen(pass_fds=...)`. The child timeout/torchrun lifetime retains the lock descriptors; close only the parent's copies after successful launch. On failure close all acquired descriptors and do not dispatch. Native `RUN.lock` still protects the output identity. A new timestamp must not bypass the same-run lock.

Use the existing launch environment: exact leased `CUDA_VISIBLE_DEVICES`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONDONTWRITEBYTECODE=1`, `TOKENIZERS_PARALLELISM=false`; keep the old launcher thread policy and production FA2 flags. Do not turn on a new deterministic FA2 backend for this continuation. Existing28800s trainer wall bound and29400s+180s outer exit bound remain. Fresh lease duration must encompass the complete external interval, not merely be unexpired at startup.

The updated binding must validate a current checkpoint-commitment receipt rather than reuse the old static first-wave free-space expression. No cleanup or old checkpoint deletion is authorized. Preserve per-invocation launch/log receipts on resume; do not overwrite the first LAUNCH or TRAIN.log. Actual post-load strict checksum/optimizer checks happen once in the native runtime, not in this metadata review.

## Inherited protection and minimum changed-scope checks

| Protection | Existing evidence | Scope |
|---|---|---|
| Exact parent registration/tag/receipt; only stop/output may change | `test_budget_extension.py` | Real tiny native inventories, negative recipe/receipt/source/data/partial-boundary cases; mocked registration verifier, not a large-model engine test |
| Parent model/master/Adam/options/RNG restore | `OFFICIAL_WORLD_LIVE_NATIVE_RESTORE_REVIEW.json`, SHA39b59283… | Actual sourcefdf25f99 four-rank exact live values, zero new updates; nonpersistent buffers rebuilt by unchanged source |
| Actual resumed sample IDs, LR, counts/cursor and task RNG | `OFFICIAL_WORLD_RESUME_EXECUTION_REVIEW.json`, SHA92feea11… | Actual world4/global32/micro4/original20k horizon resumed run; live values covered separately above |
| No fresh-master overwrite on continuation | `test_fp32_master_init.py::test_native_resume_and_budget_continuation_are_unconditional_noops` | CPU installed native-layout/linker + actual AdamW state; not a new CUDA engine execution |
| Complete optimizer/rank RNG inventory, corrupt/partial/foreign rejection; terminal handling | `test_training_state.py`, `test_review_training_contracts.py` | Existing native receipt/reader behavior |
| Original LR timeline | `test_student_state.py::test_schedule_continues_across_pause_and_respects_declared_horizon` and strict `_recipe` | Restored completed step feeds original warmup/cosine; no20000-step restart |

Production FA2 A/A backward variability is recorded in `OFFICIAL_FA2_AA_VARIABILITY.json` (dfb692ab…). Exact state restore does **not** promise bitwise equality of all subsequent cross-process gradients/updates. No numerical tolerance or backend is changed to disguise this.

Only these affected checks remain: (1) materialized new fingerprint/config equal the exact proposal and parent; (2) new three role receipts bind that fingerprint while inheriting the existing evidence; (3) versioned launcher stop20k/first-versus-resume/output/locks/explicit approval routes reject stale5k or other runs; (4) actual current host/UUID/environment/CP/lease/floor and no duplicate active output. No new model smoke, algorithm gate, teacher rebuild, test population change, or full checkpoint rescanning by reviewers is required. The native runtime remains responsible for its genuine strict payload verification immediately before loading.

This handoff is not a launch approval and does not claim20k training or scientific gain has occurred.
