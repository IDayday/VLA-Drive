# Independent Systems review: exact indexed-composition qualification

**PASS_RESOURCE_PLAN_WITH_EXPLICIT_REPORT_BUDGET**, limited to one execution of the exact 32-scene plan. This is not teacher-cache construction, training, formal temporal admission or a performance result. Independent Correctness approval remains a separate requirement. Reviewer did not execute the qualifier, open images/tensors/weights, modify the ledger or dispatch a job.

Runtime: `/mnt/project/VLA-Drive-st-indexed-qualified-runtime-v1-20261017`, clean commit `31c3dca2fbc697b419bf343ce05d10be0a4fe13f`.

Plan: `/mnt/project/st-world-policy-20261016/audits/indexed_provider_composition_plan_v1/PLAN.json`, 211,856 B; SHA256 `f96e5612c068aa4000172f7076ea37031b8fe09d03b2f60706d57300e540f0eb`; semantic identity `df38f4997f888c868a7de3991768be065646907d3d30480a720147131683688a`.

Exact fresh result directory: `/mnt/project/st-world-policy-20261016/audits/indexed_provider_composition_run_v1`.

## What was checked

The standard-library `check_plan.py` read 3,892,011 B of exact controls/source and small shard receipts, plus file stats. It did not rebuild the plan or hash the large full-current index or payloads. All checks passed. Its source/receipt pins, source-clean status, GPU snapshot and conservative serialization fixture are preserved in `RESOURCE_REVIEW.json`.

The request index and original full-current source index match their exact COMPLETE hashes and identities. All nine executable source pins match. The previous fixed-8 qualification has no errors, the same FP32 execution identity, no tensor writes and unchanged construction RNG. The same native teacher/processor/three-view contract is retained. All 188 referenced source shard receipt hashes and existing payload byte sizes match the plan; payload hashes are inherited from those receipts, not newly claimed verified by this reviewer.

Actual population: 32 scenes, 294 unique images, 189 existing-cache hits and 105 compute misses. The registered cases are cold, partial LRU, cold restored and reordered. The call bound reconstructed from the actual scene keys is exactly **367 calls × 8 compute slots = 2,936 maximum image forwards**, including padding. No extra worker or training step is launched by this qualifier. Its embedded `training_admission.max_updates=4` describes a possible later consumer ceiling, not permission to perform it now.

Resource limits remain: one GPU, two CPU threads, cooperative 8 GiB peak process RSS, 16 GiB peak CUDA **allocated** memory, 900-second cooperative deadline, 64 GiB logical input accounting, zero new teacher tensors. Reference tensors are RAM-only: 854,863,632 B registered online/composed reference values, under the 2 GiB reference limit; provider LRU is 14,159,232 B. Planned conservative logical input is 21,103,670,755 B. CUDA reserved memory and total device usage are distinct from the 16 GiB allocated counter and must be reported after execution. There is no hard claim that an individual non-preemptible operation cannot briefly overshoot a cooperative guard.

Use the original owning GNU timeout with TERM at 930 seconds and KILL after another 30 seconds. The one finite main process has no worker tree. Missing RESULT, nonzero exit, timeout or interrupted report publication is INCOMPLETE/FAILED, with evidence retained and no automatic retry. This is deliberately not the separate frozen-W lifecycle controller.

## Output accounting and exact decision

A new teacher-cache ledger claim is **not required for this finite qualification-report-only output**. No reusable target cache or tensor is created. However, the report bytes are not free: Lead must retain an explicit **64 MiB administrative reservation**, 32 MiB for each exact plan/run directory, including controls, logs, root/block allocation and partial atomic files. This reservation must be carried into any subsequent storage-policy/typed-W admission and cannot silently disappear when calculating the 4 GiB package. It is not represented as an already admitted cache ledger claim.

The existing v7 metadata reserve is 1,136 MiB; adding this report reserve yields **1,200 MiB (1,258,291,200 B), below the unchanged 4 GiB package**. Cache cap remains 768 GiB and free floor 1 TiB. The observed ledger SHA is `7fcfc0c18fa69fc65a125ee36acc1204f9976e62926bede016269136f79cebbe`; the policy SHA is `b7062c4da878519c435db2e88b899bd476e5cd6a0e9f16dca191fe61522bb831`. Both recent 32 MiB index claims and all preceding claims remain protected. At review, 7,646,259,310,592 B was available, exceeding the conservative full cache cap + 1 TiB floor + 1,983 GiB checkpoint obligation + this 64 MiB reservation.

`bounded_json` checks the *directory's other logical bytes + new serialization + old destination bytes*, so its 32 MiB bound already includes old/new atomic coexistence. It does **not** mean a free additional 32 MiB staging allowance. It does omit filesystem block rounding, root allocation and unrelated external logs. Those omissions are covered here by a much tighter exact output-file set and file-size limit:

- Fixed plan directory: existing PLAN, one one-shot LAUNCH control and one RUN log; no other outputs or retries.
- Fixed run directory: RESOLVED_PLAN, PROGRESS, RESULT and at most one atomic temporary file.
- Set `RLIMIT_FSIZE` to **4 MiB in bytes**, `RLIMIT_CORE=0`. Do not use an ambiguous shell block count. A one-MiB limit was explicitly rejected.
- The actual-layout CPU serialization fixture, including long finite float representations and a 64 KiB overhead allowance, is **2,589,495 B**, below 4 MiB. It is a size fixture, not fabricated numerical evidence.
- With each file capped at 4 MiB, the fixed set permits at most roughly 12 MiB in the plan root and 16 MiB in the run root, leaving ample root/block/log headroom below each 32 MiB ceiling. Only these files are allowed. Recheck root-inclusive stat before/after; retain every partial file. Do not add new attempt directories under this reservation.

This exception must not be generalized to an unbounded producer, cache, report population, second concurrent qualifier or new tensor kind. Those require explicit admission/reservation through the shared storage mechanism.

## One concrete command

Lead owns execution, must have the separate Correctness approval, and must recheck GPU0 immediately before launch. During this review GPU0 had 0 MiB and no compute apps. GPU1 is now reserved for the separately reported SP1 evaluation; GPU3 export and GPUs4–7 training remain protected. Do not select a substitute GPU by killing a task.

The following is one shot, keeps the log within the plan allocation, and leaves the run output absent until the qualifier creates it. It applies only to the already pinned plan/runtime above. The LAUNCH file is intentionally created with exclusive mode; any existing LAUNCH/log/run directory requires stopping, not retrying automatically.

```bash
cd /mnt/project/VLA-Drive-st-indexed-qualified-runtime-v1-20261017
/root/miniconda3/envs/ddp/bin/python - <<'PY'
import hashlib,json,os,resource,shutil,subprocess,time
from pathlib import Path
root=Path('/mnt/project/st-world-policy-20261016')
plan=root/'audits/indexed_provider_composition_plan_v1'
output=root/'audits/indexed_provider_composition_run_v1'
assert not output.exists()
assert hashlib.sha256((plan/'PLAN.json').read_bytes()).hexdigest() == 'f96e5612c068aa4000172f7076ea37031b8fe09d03b2f60706d57300e540f0eb'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip() == '31c3dca2fbc697b419bf343ce05d10be0a4fe13f'
assert not subprocess.check_output(['git','status','--porcelain'])
assert shutil.disk_usage(root).free > ((768+1024+1983)<<30)+(64<<20)
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
gpu=next(r.split(',') for r in rows.splitlines() if r.startswith('0,'))
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
assert int(gpu[2].strip()) == 0 and gpu[1].strip() not in apps
assert sum(max(p.stat().st_size,p.stat().st_blocks*512) for p in [plan,*plan.iterdir()]) < (16<<20)
resource.setrlimit(resource.RLIMIT_FSIZE,(4<<20,4<<20))
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
os.environ.update(CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',
                  PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
command=['timeout','--signal=TERM','--kill-after=30s','930s','taskset','-c','0,1',
 '/root/miniconda3/envs/ddp/bin/python','-m','tools.st_world_policy.qualify_indexed_provider','run',
 '--plan',str(plan/'PLAN.json'),'--output',str(output)]
with (plan/'QUALIFICATION_LAUNCH.json').open('x') as f:
 json.dump(dict(pid=os.getpid(),started_unix=time.time(),command=command,
               report_reservation_bytes=64<<20,cache_ledger_claim=False,automatic_retry=False),f)
fd=os.open(plan/'QUALIFICATION_RUN.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
os.dup2(fd,1);os.dup2(fd,2);os.close(fd)
os.execvp(command[0],command)
PY
```

The command changes no scientific/runtime source. Lead should retain its return code and the timeout PID/start identity. One concurrent CPU scoring workload does not justify increasing the two-thread budget. After execution inspect RESULT status, actual budgets, frozen/RNG checks and root-inclusive usage; no numerical PASS is implied by this resource review.
