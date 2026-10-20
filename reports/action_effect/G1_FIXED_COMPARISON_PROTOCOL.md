# G1 fixed comparison before physical results

Registered in source on 2026-10-20, before the first G0 simulation is launched.
All six controls use the same original S0 snapshot, fixed hash TRAIN cohort,
log-separated fit/validation, model allocation, initialization seed42, sampled
scenes/pairs, optimizer recipe and1000-update budget. Fit scenes have been seen
by the original S0; validation means post-training held-out logs, not Base-unseen.

For all G1 arms the four `joint` longitudinal+lateral combinations are excluded
from fitting. They remain in G0, including harmful outcomes, and are evaluated
separately on held-out logs as unseen intervention combinations. This definition
uses intervention metadata, not simulated outcomes. A/B still use the identical
verified-repair dataset; this G1 holdout is not a change to their targets.

G1 primary comparison is real held-out top1 intervention delta (including an
identity fallback), with direction/event/time errors, uncertainty coverage,
independent-scene counts and non-tie denominators. A single predictive RMSE or
training loss is insufficient. G0 outcome coverage and class imbalance must be
reported. Privileged-context control is BLOCKED until a verified asset exists.

A0/A1 and B0/B1 use the same verified repairs and original high-quality keeps.
B uses fixed1:1 repair/keep exposure; all arms report actual unique support and
replacement. Initial finite budget1000 updates/arm; early engineering smoke uses
stop_after against that unchanged schedule. Only supported finalists receive
additional seeds43/44. No development/navtest feedback chooses these rules.
