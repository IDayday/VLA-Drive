# Qwen-World/PostCoT IL37k — user-requested Navtest

This distinct clean-x World/PostCoT IL is still training toward100k; its latest
saved checkpoint at this request is37k. The other original native Qwen-Flow IL
was stopped by the user at98486 and is not resumed by this request.

There is **no full Navtest score yet** for this World/PostCoT run. The last
completed DEV checkpoints are30k90.568472 and35k89.471151 on1696 scenes;
these are DEV-only observations and are not Test evidence of method improvement.
The previous native IL98k/RL50k/temporal-DINO24k evaluation did not include this
separate model family. The standing user request says to start Navtest when no
new result exists, so a native37k checkpoint is pinned for one full interim Test.

The original World evaluator waits for100k IL and DEV-only parent selection.
An independent, explicitly named user-interim registration is added for this
37k request. It binds the exact native/export weights, full data/population,
canonical protocol, output, seed2101 and source. It grants no RL-parent selection,
training/tuning use, extra seed or qualification/subset exception. The original
final-selection gates remain in place by default; running formal sources and
the original training/controller are untouched.

46 model/codec/schedule/loader/deployment/scoring source files match the frozen
formal_main_v2 snapshot exactly. Only evaluation authorization/identity, optional
native/export real prediction parity, and a finite dispatcher are added. The
existing native worker gains two exact World export/evaluate module names and
an opt-in existing shared GPU lease; it cannot start World training. Actual
native/export parity must pass on rank0 before the remaining shards are admitted.

One candidate, clean-x K10 shift5, canonicalv1.1, seed2101, full12146 scenes/136
physical logs. Current legal camera/navigation/ego inputs; no teacher, search,
candidate bank or scorer selects the deployment output. The existing World
model.predict/export/RewardService/scene validator/summarizer are reused. The
canonical protocol matches the already completed native Navtest protocol exactly.

Only fixed authorized hosts, positively identified cohosts and measured headroom
are admitted:48GiB for the two-model native/export parity step,32GiB for other
workers, one new evaluator per GPU, bounded CPU loading and existing shared leases.
No real/unknown job is stopped and no temporary host is reactivated.

Eight targeted CPU tests passed (interim role rejection, complete population,
deployment/official-feedback joins), and executable compilation/help passed.
CPU scope tests, unchanged-source hashes and executable compilation are recorded
alongside the pinned manifests. Completion requires all eight native workers to
exit0, actual native/export parity, exact full scene membership and official
metric joins. Incomplete results are not assigned a PDMS. Worker wall and
forward/scoring time remain separate; native parity/export are additional costs.

```bash
/usr/bin/python3 -u -m tools.qwen_world_postcot.user_interim_owner \
  --root /mnt/project/qwen-world-postcot-20261025/interim_NAVTEST_IL037000_20261027T0520Z
```

Do not duplicate a live owner. This dispatcher schedules compute; it has no
separate result-watching loop. A completion callback validates and pushes the
full report. The next assistant read window is recorded from the historical
DEV per-scene rate, eight shards, export/loading and tail buffers; do not query
status repeatedly before that window except on user request or actual event.
