# Final exact T1 adapter lease clamp: PASS

**PASS_BOUNDED_T1_ADAPTER_ONLY**, exact owner `59c256ec7c387d4b822a45787074db55386e2116`, adapter SHA `fe7b418dadb683c143f61d51508df32dae0c9d92f2726a4aade1950f9d112f93`.

This adds only the actual admission-expiry clamp to the independently passed resource review `64ffae2be2bdedf84d1df124648a2da4a01ab914`. The earlier failed review `14831196b440081e104c3ec2ab9a9ab4255d6a06` and both test records remain intact. No scientific/model/recipe or new resource scope was reviewed or changed.

`parent_main` passes the real admission `valid_until_unix` to `monitor`. Before Popen, the monitor converts this expiry to a monotonic deadline and refuses a setup-delayed invocation lacking3780s. Its hard deadline is the earlier of that lease boundary and original start+3780. First TERM cleanup, finally cleanup and failed-spawn cleanup all retain this bound. The previous deadline assertion remains unchanged; only the required expiry fixture argument was added.

Final affected tests: **7 passed in0.83s** on CPUs0/1 with CUDA hidden: late setup/no-spawn, nonfinite actual-expiry rejection, original cleanup deadline, and normal/terminated tiny owned children. Prior27PASS remains inherited for unchanged controls/storage/accounting. No real GPU, model, image, teacher payload, registration, ledger change or training launch occurred.

`REVIEW.json` is directly consumable by the adapter. It pins the final SHA, actual Lead CP13f3 control, separate120GiB qualification reservation, original2183GiB obligations,1TiB free floor and expiry **1792250452.226822**. It acknowledges the existing Lead-created advisory commitment; this reviewer allocated nothing. Actual quota remains unverified. Fresh space/mount/GPU/lease checks still occur at launch, and phase4 requires actual successful phase2 lineage plus remaining512GiB cumulative reads. Only two invocations are in scope. No third reload, automatic retry, formal initialization or scientific PASS is implied.

Use the same owner-path command from v2 with an actual immutable Lead admission pin referencing this final review. Source/lease/control changes invalidate the exact binding.
