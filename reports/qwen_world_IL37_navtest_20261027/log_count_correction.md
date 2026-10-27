# Report-only log-count correction

The frozen Navtest has12146 distinct tokens and136 manifest log fragments.
The existing World `physical_log` definition removes final start/end frame
suffixes and correctly groups these into44 physical recording sessions. The
original World summarizer already produced44; the new finite request owner
incorrectly required136 in that different field and stopped after successful
scoring, before publishing a COMPLETE report.

All eight evaluators and the exporter exited0, and native/export actual
predictions matched exactly. The existing full-population summarizer was reused
to revalidate every scene seal, token/log/score/protocol join and recompute all
means; every value is exactly unchanged. There are0 invalid/missing scenes.
The original SUMMARY.json, failure.json and per-scene results remain unmodified.

The correction validates scene completeness against the full locked population,
then checks fragment and physical recording counts separately. The new regression
accepts136/44 for the complete12146-scene population and rejects a missing scene,
wrong physical count or invalid scene. Three interim/summary tests pass.

No actor, export, sampling, decoder, precision, reward or scoring code changed.
Only the finite owner's report validation/display and this regression changed;
the original training/controller snapshot stays untouched. The completed request
is finalized read-only from the existing outputs, not restarted. Original
execution source55e5ae94cad3cd3f96c480676cb943861df9dc10 remains recorded.

The136 fragments must not be described as136 independent physical recordings
for a recording-cluster confidence interval. This correction does not change
PDMS or replace scene weights; this single-model report does not claim an RL
gain or a paired interval against another model family.
