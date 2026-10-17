# Independent history64 actual admission

Reviewer: `/root/st_calibration_independent_correctness_v2`. Exact runtime:
`b847f5a99dd44ee5256362d380d9d96efb47d35b`, with prior independent static wiring
review `8319e30531a19b12bccf6b0cbc62fb122829aebc`.

**PASS_NARROW_ACTUAL_ADMISSION_ONLY.** This authorizes one already-reviewed
fixed64 CPU metadata pilot. It is not a completed experiment, formal training,
full-population build, automatic retry, GPU admission, or scientific result.
No task was launched and no ledger or runtime was changed by this reviewer.
The earlier pending `CONDITIONS.json` is preserved as the preceding conditional
precheck; it is not itself an admission.

The actual admission is `ACTUAL_ADMISSION.json`, SHA256
`abe66a14cbfc9e617f876b145088b4c66b9df4b1f934e6d7d8dce37db4e1e9e8`.
Its issue/latest-start/expiry times are respectively
1792243194.164604 / 1792243494.164604 / 1792244694.164604 UTC Unix seconds.
Expired admission must not be reused. Lead must do a final fresh check before
launch; this receipt does not reserve host resources against other operators.

## Exact interface and window

The controller requires these 16 fields: `schema`, `status`, `reviewer`,
`hostname`, `issued_at_unix`, `latest_start_unix`, `valid_until_unix`, `limits`,
`checkpoint_receipt`, `output_root`, `runtime`, `control_root`,
`control_reserved_bytes`, `controller_sources`, `policy`, `registration`.
All fixed fields match the unsigned proposal without modification.
The status is `APPROVED_HISTORY64_PILOT_ONLY`; the reviewer is the independent
identity above and the host is `training-vla-zt-worker-0`.

The reviewed existing controller's pure `validate` function was executed from
its AST only, without importing controller/module top-level code. It accepted
this actual receipt. Finite issue <= launch <= latest-start is required;
latest-start + 940 <= expiry <= pinned checkpoint commitment expiry.
The current CP expiry is 1792250452.226822. The 900 second work plus 40 second
cleanup budget, CPUs 6/7, no GPU, 16 GiB logical read and 4 GiB RSS caps are
unchanged. Artifact budget is 31 MiB plus a separately adopted 1 MiB control
budget. The existing 4 MiB monitor addendum and 128 MiB administration scope
remain intact. No earlier 44/14/15 test suites were rerun.

## Actual evidence and retained failure

The fresh Lead quiescence, process observation and control adoption receipts
were verified against the exact supplied hashes. Actual same-ledger migration
is `1a729d88423b0f6643c33304aed715bff49d2b4f4d57c8fbfc1f13d4b438d6b7`;
its complete receipt and current ledger hash
`d81af1a046ce83765964935ec6cf9d9d876a21476e6f4e575bbfa85e95b6f526`
match. Old claim records, observed bytes and charged bytes are exactly equal
to the saved pre-migration ledger. The new core matches the proposed policy;
768 GiB cache cap and 1 TiB free floor remain unchanged.

A failed writer can be quiescent: `metadata_storage.migrate_policy` requires
fresh Lead-owned QUIESCENT evidence for every covered root, no active/unknown
writers, disabled queues, exact input hashes and lock/accounting checks. It
does not require the previous build to have succeeded. Therefore the stopped
TRAIN's **FAILED_NO_AUTO_RETRY** result is preserved and checked. Independent
exit review `2e8240ee34b9ee9a03c408af8ef146db6a4cb5cc` confirms controlled exit;
its earlier observer RSS failure is also retained. Original parent/child PIDs
were independently absent here. Descendant coverage and disabled queues rely
on the pinned fresh Lead census plus that independent exit review, not on
absence of two PIDs alone. No new all-host census or signal was performed.

The pinned CP is CURRENT and covers 2303 GiB remaining commitments plus the
1 TiB free floor. Current statvfs free space exceeds this plus 32 MiB. This is
an advisory reservation, **not verified filesystem quota**. Both new roots
were absent and their parent chains plain. Five controller source hashes and
eight builder source hashes match; the frozen runtime HEAD matches. The
original fixed64 population, target time 0 and packed history mathematics
inherit the earlier exact source/config review.

## Measured administrative cost and boundary

`ACTUAL_REVIEW.json` and `.log` contain the independent review and full output.
The main review used CPU8/9 for 0.07443 seconds, 481290 explicitly read small-file
bytes, 535108 measured process logical-read bytes, and peak RSS 19357696 bytes.
`SOURCE_RECHECK.json` separately records eight-source recheck cost. Preliminary
interactive small-source/schema reads are not included in that process counter;
they were also bounded small JSON/source reads, with no payload, image, model,
checkpoint tensor or teacher data read. No whole-repository test or scan ran.

Lead still owns immediate pre-launch checks: lease validity, unchanged pins,
CP/free accounting, output/control namespace, no competing writer, CPU6/7
availability, CUDA hidden, and exactly one invocation. The signed receipt does
not convert prior failures to success or authorize a retry if this pilot fails.
