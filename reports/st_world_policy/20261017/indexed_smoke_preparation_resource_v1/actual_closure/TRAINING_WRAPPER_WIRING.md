# Narrow training-wrapper wiring observation

At Lead request, inspected only private `run_indexed_student_smoke2_v1.py` SHA `97a5e2294a8973ec2a99727bb47b5a98266397d92c1d1ec25a16dae32f4bdcbb` and the old controller's called `execute` function. No registration, GPU query, launch or signal was executed by this reviewer. This is not another model/formal audit.

A concrete adaptation defect was found before launch: new wrapper changes `m.GPU` to0 and preparation locks `/tmp/foresight_gpu_0.lock`, but old `execute` line241 hardcodes `/tmp/foresight_gpu_1.lock`. It would run GPU0 while locking GPU1. The minimal correction belongs only in the new wrapper: map that exact old lock path to the new GPU0 path, retaining all other lock paths and original execute implementation. This protects GPU0 and avoids competing for an unrelated GPU1 lock.

Other direct call arguments are consistent: `args.case='single'`, `args.action='run'` produce the original one-invocation, nonresume path; overridden locations point to the separate training2 root. New storage check reserves the full1983GiB other-run commitments plus remaining120GiB task allocation plus1TiB floor. This is accounted advisory space, not quota. No rewrite of the completed factory or inherited training source is required.
