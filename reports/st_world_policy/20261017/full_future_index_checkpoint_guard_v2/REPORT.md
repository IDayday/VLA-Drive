# Index checkpoint-space guard update

The existing index controller required exactly 2183 GiB of checkpoint obligations. The separately admitted T1 two-invocation qualification adds 120 GiB. Reusing the old constant in a later index admission would undercount that commitment.

The new guard requires an independently pinned CURRENT receipt with integer remaining_bytes at least 2303 GiB, retains the 1 TiB floor and full lease-expiry checks, and passes the actual (possibly larger) receipt obligation to the existing storage guard. No active future builder, its old receipt, runtime, storage ledger, scientific input or index implementation is changed. New receipts remain subject to the existing exact SHA validation.

75 CPU tests passed; ten new cases reject the old smaller obligation, one-byte under-reservation, non-integer/bool values, wrong floor, stale/nonfinite expiry and historical status. A larger valid commitment is accepted. Full output and exact command are retained. This is an implementation result, not self-issued independent approval.

The previous index RUNBOOK and admission templates refer to the historical 2183 GiB bound. For all subsequent index launches, replace that reference with the current pinned full commitment receipt (currently controls/CHECKPOINT_COMMITMENTS_2303G_V1.json under the T1 qualification root), and pin this updated controller. Do not edit or reuse a frozen old admission. The live TRAIN metadata writer must finish and release the shared writer lock before index claims or launches. Complete future metadata and successful actual build receipts remain mandatory.

No index launch and no new teacher tensor bank occurred in this change. History/future target mathematics and official training precision are unaffected.
