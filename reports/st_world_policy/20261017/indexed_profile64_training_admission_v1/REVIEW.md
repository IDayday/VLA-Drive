# PROFILE32 / 64 updates: independent resource and narrow control admission

**PASS_ONCE_PROFILE64_RESOURCE_CONTROL.** This admits one preparation and one initial PROFILE invocation for the exact run below. It does not claim that training has started or completed, authorize retry/resume, or qualify formal/full-population training.

Reviewed controller commit `238db78dd3ed3018a63592f82a1b50617a03e142`; Lead's copy integrated at `b5f7dc92` has the same verified controller and adjacent INPUTS bytes. Runtime stays at clean `2d5e9483a8c8bc80b71348211756f3cab6aa21b1`. Existing independent source/numerical review is inherited, not repeated.

| Binding | SHA256 |
|---|---|
| controller.py | `2d57d72b6ca704d8dcadf7d31fe2402fb023e8f92a334f631ea9950ac757cecc` |
| INPUTS.json | `d6152dc17f5adddad324876bae55d5b50d5ddc813e14edbd951128a1caf87bc8` |
| Actual factory RESULT | `6407b37cd7f457c57f60717c71ef47bcedc9e2d40c090e51239cdc2fcf8be6d1` |
| Actual config | `e72f767f4dfdc349508befb64665bb11f4e407b8e670fc3a31dd983f8c0c7676` |
| Independent ADMISSION.json | `416654825c1645770301dbe0b60157e3aba6248cbd0918030959b9801e74fe80` |
| Fresh CHECKPOINT_COMMITMENTS_2183G_V1.json | `d5bce1fec28d6ce3dab496287c9be41852ca154495ecf4cf686c7e780ebf4d52` |

## Control checks

- Default `preflight` is read-only. `prepare` and `run` require the explicit admission hash and exact schema, source, run, output, controller/input hashes, byte budgets and finite validity interval.
- The launcher uses physical GPU 0 and `/tmp/foresight_gpu_0.lock`; it does not call the old GPU1/two-update `execute`. Preparation captures the idle GPU UUID under that lock; execution reacquires it, checks idle state and the same UUID, and protects other jobs. Current GPU0 UUID `GPU-f679bff0-a4a2-ae4f-0b85-74c519b45a9c` was observed at 0 MiB / 0%. Launch must recheck.
- Source and completed factory/config/provider/population pins are rechecked. The new registration has its own environment, lease, immutable config and source fingerprint. Old SMOKE registration is not repurposed.
- Run/output/launch guards preserve incomplete attempts and reject automatic reuse. Only one initial invocation is approved. The GPU lock descriptor is inherited by the newly created `timeout` session. No unknown process is signalled and no old checkpoint is removed.
- Child command uses existing `torch.distributed.run` and `train_world`, one rank, microbatch 1/global batch 2, 32 fixed scenes, stop_after 64, unchanged 20,000-update/warmup1,000 schedule and objectives. A zero exit is insufficient: the launcher requires `completed == 64` and `status == PAUSED`; earlier boundaries cannot masquerade as 64 completed.
- CPU controls are two-thread OMP/MKL/OpenBLAS settings, not hard process affinity or a cgroup CPU limit. Data workers remain zero. This reviewer did not invent a new training I/O cap; metadata-factory 4 GiB is not a valid cap for loading native training state.

Six owner CPU tests were independently rerun: **6 passed in 0.009 s**. Four additional negative cases reject altered controller hash, altered INPUTS hash, omitted 120 GiB prior commitment, and validity shorter than 1,800 seconds. The actual issued admission passes the real controller reader. Actual metadata `common()` and `storage()` checks pass without creating output; no prepare/run/model/GPU computation was called.

## Runtime and storage boundaries

The trainer's 1,500-second setting is checked at optimizer-loop boundaries. An in-progress update or subsequent boundary save can exceed it; it is not a guaranteed complete-save deadline. The actual outer `timeout` is **1,620 seconds**, with **180 seconds** before forced cleanup: the admission and lease cover at least **1,800 seconds** after every prepare/run check. This permits a bounded attempt; it does not guarantee storage or kernels finish successfully. Failed/partial evidence must remain intact.

`save_every=64` yields one normal checkpoint; the immediate stop boundary uses the existing verified `can_reuse_just_saved_boundary` path rather than duplicating it. The 80 GiB allowance covers one observed 37,365,928,655-byte normal native directory plus one partial slot, leaving 11,167,488,610 bytes for overhead. No additional periodic snapshots are authorized.

Fresh filesystem available bytes: **7,339,375,292,416**. Conservative requirements are:

- Existing full commitments: **2,103 GiB**, retaining original 1,983 GiB plus previous 120 GiB smoke budget without subtracting written files.
- This PROFILE: **80 GiB** additional.
- Combined checkpoint commitments: **2,183 GiB**, plus an independent **1 TiB** free floor, totaling **3,443,490,029,568 bytes** required.

The new receipt is a separate file; the old 2,103 GiB receipt was not changed. The controller rechecks live space at prepare and dispatch, accounting this run's existing owned file bytes against its remaining 80 GiB. This is advisory accounting, not filesystem quota enforcement or an in-training disk monitor. Shared quota remains unverified. No live cache ledger/policy is modified; metadata remains at its prior effective 1,264 MiB reservation and the 768 GiB cache cap stays separate. Later resource admissions must carry forward the new checkpoint commitment.

## Validity and exact Lead commands

Not before: Unix **1792225552.2251627**. Latest start: **1792230952.2251627**. Deadline: **1792232752.2251627** (two-hour approval; a full 1,800-second run/cleanup must fit). Both prepare and run check the remaining window.

```bash
CONTROL=/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261017/indexed_profile64_control_v1/controller.py
ADMISSION=/mnt/project/VLA-Drive-st-sp0-dev-plan-resource-review-20261017/reports/st_world_policy/20261017/indexed_profile64_training_admission_v1/ADMISSION.json
ADMISSION_SHA256=416654825c1645770301dbe0b60157e3aba6248cbd0918030959b9801e74fe80
env PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python "$CONTROL" --action prepare \
  --resource-admission "$ADMISSION" --resource-admission-sha256 "$ADMISSION_SHA256"
env PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python "$CONTROL" --action run \
  --resource-admission "$ADMISSION" --resource-admission-sha256 "$ADMISSION_SHA256"
```

Lead is the only dispatcher. The source copy used to issue the token and the Lead command copy are byte-identical, so the hash-based reader accepts both. Adjacent INPUTS must stay identical. The review script is an issuance record, not permission to regenerate expired approvals. No training output/registration/claim/root was created by the reviewer, and no GPU lock or resource was acquired.
