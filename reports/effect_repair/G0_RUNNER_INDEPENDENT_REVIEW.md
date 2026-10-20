# Bounded G0 reader/simulation review

Reviewer: lead, independent of runner author. Source target f67ff2b4. Inspected
prepare/CallLedger/process_scene/score_candidate/read_population/build and existing
native ConsequenceRequest/Engine contracts. This approves only64 prelocked TRAIN
observations,4 CPU workers,2048 admissions ceiling,3600seconds; no model training.

The original S0 replay and immutable64-feature completion already exist. Engine
receives one official reference and one requested trajectory each time. Local
engineering controls retain source identity; resampled original-policy outputs
retain S0 provenance. Same reference INPUT/context/protocol/numeric execution
is required; execution output hashes are not incorrectly required identical
across alternate native batch histories. No failed simulation becomes valid zero.

Call admission is reserved under flock before execution, with conservative
charged crash windows. A COMPLETE is followed to canonical checksum and request,
not just path existence. Cross-population reuse uses exact request identity and
revalidates canonical bytes. Unexpected errors block rather than silently retry.
Fixed-grid longitudinal/lateral/joint and larger-support/extra policy samples
are separated. Geometric screen is not safety; controller tracking is also
required for verified repair. Minimal local repair is selected by mean pointwise
XY distance under declared safety/progress/comfort constraints. Old failed markers
cannot force a new invocation COMPLETE.

Root rerun initially omitted CUDA_VISIBLE_DEVICES='', so one environment-guard
test rejected the launch (13 passed/1 environment failure). Correct CPU launch
must hide CUDA and set OMP/OPENBLAS/MKL=1, as enforced by the runner. The corrected
rerun is recorded separately. This is not real physics or scientific success;
actual simulation/replay costs and outcomes are the next bounded step.
