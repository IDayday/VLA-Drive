# Independent narrow plan-wrapper review

This review covers only the one-attempt metadata/stat PLAN wrapper, not the
future builder, index execution, teacher extraction or training.

Owner source:
`/mnt/project/VLA-Drive-st-full-future-plan-control-20261017/reports/st_world_policy/20261017/full_future_plan_control_v1/plan_once.py`.
Final reviewed SHA256:
`bd3c1dadaed8c79ea5242bcf40e95d991b11851edfd6b61d47f1073f6f41159f`.

Initial source SHA `ad52763cc483718d9e5921b6c189d210c273ce1e54342cdecab7a74938454c50`
had two independently reproduced failures. Train's otherwise valid exact argv
was rejected because the allowlist reused dev's smaller future-build limits.
Its directory accounting returned one byte for a one-byte fixture whose root
plus allocated file occupied 4,116 bytes on the fixture filesystem. The original
counterexamples remain in `WRAPPER_REVIEW_ORIGINAL_FINDINGS.json`.

The owner repaired only those points. Train now requires its exact 14,400 s /
512 GiB source-read / 2 GiB metadata / 192 MiB raw-log proposal; dev retains its
original values. These proposal limits do not change the outer PLAN execution
budget. The root census now includes the root itself, subdirectories and every
file, charging `max(st_size, st_blocks*512)`, and retains symlink, hardlink,
nonregular file and 4 MiB file-bound rejection.

Independent CPU fixtures: **7 passed in 0.24 s**. They cover both exact role
argv, rejection of dev limits in train, directory/allocated accounting,
allocated-cap rejection, symlink rejection, and insufficient cleanup window.
The owner's CPU fixtures were rerun independently using the actual `ddp`
Python: **33 passed in 0.70 s**. Full output is retained. Neither test suite
invokes a real PLAN, supervisor child, raw-log load, image, teacher tensor,
model or GPU. They are resource/control tests, not empirical full-population
throughput or memory evidence.

Code inspection confirms the actual flow is only `prepare_future_metadata.plan`
once. It validates exact wrapper/source/helper/argv hashes, fresh output root,
the original ledger hash and the complete checkpoint reserve before spawning.
The sole mutable COUNTERS read takes the writer's original control lock. Other
controls retain strict reads. The original supervisor owns only its new child
session, bounds logging, and preserves failures as spent attempts.

The outer bounds are CPU 0/1, 2 GiB parent-plus-child sampled RSS, 4 GiB combined
logical rchar including imports, 180 s running time, and 32 MiB root-inclusive
output per role. The inherited supervisor has 30 s TERM grace followed by up to
10 s final cleanup: expiry must cover at least **220 s** from start. RSS/time/
read monitors are sampled/cooperative, not cgroup hard ceilings. Their finite
overshoot/cleanup semantics are retained rather than described as instantaneous
hard limits. Child RLIMIT_FSIZE is 4 MiB; core files are disabled. Its file-open
audit rejects raw-log/image/tensor suffixes. CUDA visibility is empty; no model
or optimizer is instantiated. Existing auxiliary helper imports are pinned.

**PASS_BOUNDED_PLAN_CONTROL** for the exact final source and argv. This does not
approve ledger migration or any metadata build. The final independently signed
admissions name the actual execution path, clean runtime and deadline. Earlier
preliminary navsim-argv admission is superseded and must not be dispatched.
