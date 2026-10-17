# Fixed TRAIN64 artifacts to the original fit300 stage

This is a narrow consumer adapter, not another extraction, monitor or trainer.
It is based on the existing DEV16 reconciliation helper and storage-bridge
loading pattern. The original DEV16 helper, `supervise_once.py`, observer,
spatial-probe computation, and registration remain unchanged.

The TRAIN extraction result remains `STOPPED_NO_AUTO_RETRY`, reason
`OWNED_DESCENDANTS_AFTER_LEADER_EXIT`, exit code 0, six verified owned cleanup
processes and zero remaining. That original result is pinned by SHA256
`4d24e4ed9a97744fe0d977c21284cb5fd64ec14c1348d71f6edadeff61245b27`.
No success status is substituted for it.

The independent original-validator review is pinned by SHA256
`725d3427d042f47882e89a5814937b10e3a7c49b5ea58da690e374c7b37dbfad`.
`train_artifact_receipt.py` accepts only its exact TRAIN64 population,
preparation, cache root and failed attempt. Its receipt records
`ARTIFACTS_ACCEPTED_SUPERVISOR_FAILED`, consumer `fit`, and a single extraction
attempt. It checks small metadata and payload file versions without re-reading
tensor contents. The original fit still performs its own full `read_world`
hash validation when separately authorized.

`supervise_fit.py` loads the exact original supervisor through the existing
pinned loader. It wraps only validation and stage-input selection. The new
admission field `train_artifact_reconciliation` cannot be used for any other
stage, retry or DEV transition. Two additional source hashes are mandatory in
`train_reconciliation_sources`. All original admission, helper, runtime,
registration, lease, CPU, shared-lock, observer and monitor checks remain.
The original fit read allowance stays 2 GiB and wall/RSS limits stay
1260 seconds / 2 GiB. There is no GPU or storage-ledger operation.

The original preparation and fit registration are pinned to
`34c174a6aa0c859a1b0a5f8b7dd800bbe189d51c770211b49c35d061e92b6431`
and `7984d1ddedb9c769caa2649fd285af7782fe464f0c6a8b6c10f546fe64940d1e`.
The original fit protocol retains 300 cycles. These controls do not approve a
launch; an independent fresh fit resource admission remains necessary.

Author regression command:

```bash
taskset -c 10,11 timeout 120s /root/miniconda3/envs/ddp/bin/python -B \
  reports/st_world_policy/20261017/w_train_reconciliation_v1/test_consumer.py \
  > reports/st_world_policy/20261017/w_train_reconciliation_v1/TEST.log 2>&1
```

16 targeted tests passed. Total script time 0.344863 seconds, peak RSS
29,429,760 bytes, once-counted logical reads 2,289,089 bytes; kernel reads
638,976 bytes are reported separately. Inputs were small pinned JSON controls
and source files; all tensor file versions were in-memory fixtures. No tensor,
checkpoint, ledger or actual fit was accessed. Fixtures wrote no disk data.
Negative tests cover wrong failed identity/path/population, arbitrary failure,
live cleanup, changed payload version, wrong consumer, source hashes,
registration and unsupported stage/retry combinations.

Source SHA256:

- `train_artifact_receipt.py`: `782f29a8aad0518c2e688e030ded9994f9feab54660d5d5773462c6e8a04edd5`
- `supervise_fit.py`: `2e630f3b1db30cdade58d960982d9f4aa5d8344518ce688e8fb1c2b96a643c06`

`RECEIPT_MECHANICAL.diff` records the changes from the preserved DEV16 helper.
This implementation and its fixture tests are not independent launch approval.
