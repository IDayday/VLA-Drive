# Bounded full-population provider preparation

This entry operationalizes the existing handoff at lead `f4764a94` without
changing the frozen runtime `a8b0036f33554fd2a00ea1af8c9b5f20fb6b965a`.
It executes exactly two existing metadata CLIs, sequentially inside one owned
CPU child: `qualify_indexed_provider prepare-profile-migration`, followed by
`teacher_provider`. The factory retains its original full-population checks,
DB/dependency hashes, ordered comparisons and frozen PCA validation. It does
not instantiate an image/teacher/model training executor or create tensors.

Limits are CPU0/1, 4 GiB aggregate RSS, 16 GiB aggregate logical reads,
1500 seconds plus 40 seconds cleanup, and 32 MiB total new output including
the one-MiB terminal/control reserve. These are stopping bounds, not measured
runtime predictions. The resource reviewer must independently admit the
invocation and current storage budget.

The 32 MiB is a separate control reservation from the original 491 MiB
unassigned part of the 8 GiB metadata envelope: 7701+32=7733 MiB reserved,
459 MiB remaining. The original 128 MiB administrative commitment is preserved.
No typed claim, ledger or checkpoint reservation is changed.

`launch_once.sh` applies GNU timeout before Python bootstrap. The wrapper uses
the previously reviewed bounded source reader and absolute alarm, original
global one-loader lock, fixed `4b2f0f…` monitor, owned cleanup and read accounting
after `wait`. Both CLIs share the absolute deadline and accounting envelope.
The original CLI JSON and logs go to stderr; stdout contains only the final
accounting JSON. Existing monitor pipe limits remain one MiB each. A log-limit
failure is retained; there is no automatic retry or parsing relaxation.

The output namespace is fixed to
`/mnt/project/st-world-policy-20261016/audits/full_population_provider_preparation_v1`.
Any existing output rejects a second invocation. Outputs include source
migration, provider descriptor, configured training config/PCA scope/lineage,
resolved recipe/consumption and immutable invocation/result controls. A failed
second CLI leaves all first-stage evidence intact. No PROFILE training starts.

After separate independent resource review, Lead invocation is:

```bash
bash reports/st_world_policy/20261017/full_population_provider_preparation_v1/launch_once.sh \
  --admission "$PROVIDER_PREPARATION_ADMISSION" \
  --admission-sha256 "$PROVIDER_PREPARATION_ADMISSION_SHA256"
```

These variables must refer to the actual newly issued admission; none has been
created by this implementation. Its schema is
`st_full_population_provider_preparation_admission_v1`, status
`APPROVED_PROVIDER_METADATA_PREPARATION_ONLY`. It binds the exact PLAN pin,
both controller-source pins, host, lease, original limits, CPU0/1, fixed output,
and PLAN's `control_reservation`. GPU, retry, ledger writes and checkpoint
reservation creation must all be false.

Author fixture command:

```bash
taskset -c 10,11 timeout 120s /root/miniconda3/envs/ddp/bin/python -B \
  reports/st_world_policy/20261017/full_population_provider_preparation_v1/test_prepare.py \
  > reports/st_world_policy/20261017/full_population_provider_preparation_v1/TEST.log 2>&1
```

Nine targeted tests passed, including real printed synthetic CLI JSON/stderr,
SystemExit(0)/failure propagation, fixed ordering/arguments, source and admission
rejection, and original monitor/lock/deadline wiring. `TEST.log` records measured
time/RSS/logical reads. Preparation inspected 2,360,454 bytes of source/small
controls while creating PLAN; no real DB, large request/selected manifest,
PCA tensor, model, image or ledger was opened. No actual preparation was run.
This author evidence is not resource approval or a successful provider build.
