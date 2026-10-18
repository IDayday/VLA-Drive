DEV only is admitted for one metadata build under the unchanged native controller. The actual package migration completed, and the current ledger matches its pinned receipt. Independent correctness review passed 18 focused fixtures; this resource review validates the actual admission without running the builder.

DEV uses CPU 10/11, one worker, 4 GiB RSS, 900 seconds plus 40 seconds cleanup, a 16 GiB paced source budget and a 24 GiB sampled logical-read threshold. Output and staging remain inside 127 MiB, with a separate 1 MiB control root. The command sends outer stdout/stderr to /dev/null; native terminal and cleanup receipts retain the evidence. The redundant outer timeout is 940 seconds plus 40 seconds kill grace and is covered by the admission lease.

All old commitments remain. Conservative metadata accounting is 10,097/10,240 MiB, leaving 143 MiB. Checkpoints retain all 2,703 GiB, separately from the 1 TiB free-space floor. Both history roles are reserved, but only DEV is admitted. The shared lock is charged to train’s reserved control space.

No builder, GPU, SSH, live ledger write or payload hash was run by the reviewer. The existing selection hash is inherited; only its size was observed here. After DEV completes, TRAIN needs its own fresh actual admission and completion evidence, with the same source review inherited.
