# Full causal-history metadata: fixed two-role binding

Prepared only; no migration, builder or GPU execution by this author.
Use the frozen runtime `/mnt/project/VLA-Drive-st-full-history-v3-runtime-20261018`,
commit `eae8922d72b3881c0c162e96b8dd67e7c15a41c7`. The original packed builder,
4b2f monitor, read counters, callback/resource guards, ledger and 40 s cleanup
remain unchanged. The new controller binds only full train (101,592) and full
dev (1,696), same original observations, offsets −1.0/−0.5 s, tolerance 0.05 s,
current reference time 0, three views and `packed_log_v1`.

`FULL_ROLE_PLAN.json` binds the original COMPLETE and selected-list hashes,
two actual native registrations, fixed builder budgets and the append-only v3
policy proposal. The new role policy adds only two typed declarations; admission
precharges their full amounts through the original API. It does not release,
shrink, retarget or overwrite any existing declaration. The envelope migration
must finish first. Neither this policy file nor the templates are approvals.

The reviewed history32 W compatibility installer is reused unchanged with the
current runtime paths and the new metadata source pin. It executes in both
parent and child before importing the metadata guard. The original SP0/SP1
source identity and proof remain intact; no old runtime is hot-modified.

Limits, including failed partials:

| Role | Typed artifacts | Separate control | Work + cleanup | Paced source budget | Outer logical-read stop | RSS |
|---|---:|---:|---:|---:|---:|---:|
| train | 2,047 MiB | 1 MiB | 14,400 + 40 s | 448 GiB | 512 GiB | 4 GiB |
| dev | 127 MiB | 1 MiB | 900 + 40 s | 16 GiB | 24 GiB | 4 GiB |

Each role uses one worker and two CPU threads, currently registered CPU 10/11,
32 MiB/s source throttle. GPU is hidden. The common `train_controls/FULL_POPULATION.lock`
is nonblocking and held for each whole parent invocation, so the two roles cannot
run concurrently. This small lock is inside the train control's 1 MiB. The Lead
must create both plain control directories before either role starts.

Train's estimated source read floor is 12,500 s before additional CPU/filesystem
cost. The 4 h cap is a stop condition, not a promise to finish. The original builder
retains immutable shards and supports its existing explicit resume mechanism;
this controller deliberately provides no automatic retry or resume. A failed role
retains its receipts and full allocation. Any later retry needs a separate reviewed
attempt binding and charged residual read budget, not deletion of INVOCATION.

Conservative metadata accounting is 10,097/10,240 MiB: 7,904 retained, 16 held for
the corrected diagnostic, 1 separate migration control, and 2,176 for both full
history roles including controls. Remaining 143 MiB. All model checkpoint
commitments and the 1 TiB free floor remain separately reserved.

## Actual executable entries

```sh
python /mnt/project/VLA-Drive-st-full-history-controller-20261018/reports/st_world_policy/20261018/full_causal_history_build_preparation_v1/migrate_v3.py --help
python /mnt/project/VLA-Drive-st-full-history-controller-20261018/reports/st_world_policy/20261018/full_causal_history_build_preparation_v1/run_full_history_once.py --help
```

After independent review and the Lead's fresh exact migration receipts:

```sh
cd /mnt/project/VLA-Drive-st-full-history-v3-runtime-20261018
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
timeout --signal=TERM --kill-after=10s 60s taskset -c 10,11 \
  /root/miniconda3/envs/ddp/bin/python \
  /mnt/project/VLA-Drive-st-full-history-controller-20261018/reports/st_world_policy/20261018/full_causal_history_build_preparation_v1/migrate_v3.py \
  --execute --old-policy "$OLD_POLICY" --old-policy-sha256 "$OLD_POLICY_SHA" \
  --new-policy /mnt/project/VLA-Drive-st-full-history-v3-runtime-20261018/reports/st_world_policy/20261018/full_causal_history_build_preparation_v1/metadata_v3/PACKAGE_ONLY_POLICY_PROPOSAL.json \
  --new-policy-sha256 744666144432007438c1030ca8144369de38df521bf34fbe2d4a092913d97b0e \
  --expected-old-ledger-sha256 "$LEDGER_SHA" \
  --quiescence-receipt "$QUIESCENCE" --quiescence-sha256 "$QUIESCENCE_SHA" \
  --pilot-inventory "$EMPTY_PILOTS" --pilot-inventory-sha256 "$EMPTY_PILOTS_SHA" \
  --checkpoint-receipt "$CP" --checkpoint-sha256 "$CP_SHA"
```

Those variables are actual Lead-owned inputs, not values inferred from expired
receipts. The two `*_ADMISSION_TEMPLATE.json` files likewise require the actual
independent reviewer, host, start/expiry interval and CP pin. CPU 10/11 availability
must be verified, and the train CP/admission must cover at least 14,440 s from
actual start. The wrapper's source and plan pins must be preserved.

After that, run exactly one role at a time:

```sh
cd /mnt/project/VLA-Drive-st-full-history-v3-runtime-20261018
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
timeout --signal=TERM --kill-after=40s 14440s taskset -c 10,11 \
  /root/miniconda3/envs/ddp/bin/python \
  /mnt/project/VLA-Drive-st-full-history-controller-20261018/reports/st_world_policy/20261018/full_causal_history_build_preparation_v1/run_full_history_once.py \
  --role train --admission "$TRAIN_ADMISSION" --admission-sha256 "$TRAIN_ADMISSION_SHA"

CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
timeout --signal=TERM --kill-after=40s 940s taskset -c 10,11 \
  /root/miniconda3/envs/ddp/bin/python \
  /mnt/project/VLA-Drive-st-full-history-controller-20261018/reports/st_world_policy/20261018/full_causal_history_build_preparation_v1/run_full_history_once.py \
  --role dev --admission "$DEV_ADMISSION" --admission-sha256 "$DEV_ADMISSION_SHA"
```

Stdout/stderr capture must be bounded outside the 1 MiB control directories,
which accept only the original control receipt names plus the shared lock.
Do not use shell redirection that creates unregistered files in those roots.
The internal controller still enforces its earlier absolute work deadline and
original owned-child cleanup; outer GNU timeout covers bootstrap failures.

## Local verification scope

18 focused fixtures cover both role bindings, strict changed-resource/source/
lease rejection, selection streaming with wrong hash/size/symlink/FIFO failures,
the real nonblocking common lock, no-retry INVOCATION preservation, and unchanged
v3 core/old claims. Separately, the real native parser, `_resources` and
`prepare_causal_history.validate` pass both actual registration files with a
fixture-only active guard. That check read no population rows/images/logs/tensors
and did not admit a writer: 4.64 s, peak RSS 579,264 KiB, process rchar 133,011,973 B
including imports. These are implementation checks, not actual build results.
The old packed-history and monitor reviews are inherited; review only new role,
source, policy, read/lease and lock bindings.
