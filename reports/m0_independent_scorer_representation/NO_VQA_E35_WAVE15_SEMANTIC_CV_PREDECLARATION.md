# No-VQA E35 Wave-15 semantic-BEV plus current-actor CV predeclaration

This protocol was fixed before any Wave-15 training result or Navtest result
existed. It was written after observing Wave-14 epochs 0--4 on Navtrain
held-out logs: semantic-BEV supervision was non-degenerate and ranking gains
were positive, but the selected candidates still traded some DAC/TTC for
progress. No Wave-14 Navtest result was available.

## Question

Can a scorer-private representation preserve the static-road benefit of the
semantic BEV while adding an explicit, current-observation-only dynamic safety
path?

Wave-15 makes exactly one model change relative to Wave-14. The already
supervised current-actor slots are extrapolated once with a differentiable
constant-velocity prior in the current ego frame. The same shared actor rollout
is geometrically relabeled against each candidate to obtain clearance, soft
collision, TTC, corridor occupancy and nearest relative actor state. A
zero-initialized gate injects this consequence token into the candidate scorer.

This is a predicted current-actor consequence under a constant-velocity prior,
not a logged future, reactive response, or counterfactual future. It does not
use future supervision in this experiment.

## Locked comparison

The following remain identical to Wave-14:

- No-VQA epoch-35 checkpoint and immutable 64 proposals;
- all 103,288 legal Navtrain scenes and 162 physical-log groups;
- the same five disjoint risk-stratified folds;
- pool-4 current F0/L0/R0/B0 M0 visual tokens;
- current actor and 16 x 32 semantic-BEV training targets;
- candidate encoder, conservative-reference objective and 192-policy gate;
- seed 2, risk multiplier 4, optimizer, scheduler, eight epochs and fixed
  epoch 7;
- all loss weights, including current-actor 0.5 and semantic-BEV 0.5.

The sole added switch is `--current-actor-cv-relabeling`. Candidate ordering
cannot alter either the semantic BEV or current-actor state. Permuting
candidates must permute all path/consequence/scoring outputs consistently.

## Information boundary

Deployment inputs are limited to the frozen M0 current-camera features,
current ego/navigation status, frozen M0 proposals and released M0 deployable
scorer features. Current annotations, semantic maps, PDM factors and all future
data are training-only or offline-evaluation data. DrivOR and every external
model representation/checkpoint remain forbidden.

## Promotion rule

The final epoch-7 model is evaluated on all five held-out-log folds. One common
policy must have a positive point estimate and log-bootstrap lower bound on
every fold while satisfying the predeclared NOC/DAC/TTC tolerance. Only then
may the architecture be refit on all 162 training logs and evaluated on the
complete FP32 Navtest protocol: 12,146 scenes, 136 logs, 64 immutable
candidates, zero invalid scenes and same-device online/cache parity at 1e-6.

Validation can promote or reject Wave-15, but Navtest cannot select its epoch,
gate, scale, shortlist or architecture. A score above 0.93 is accepted only
from that strict complete Navtest result.

## Pre-training implementation evidence

- The combined semantic-BEV/current-actor-CV forward path has a regression test
  showing that both shared scene predictions are computed once per forward and
  remain unchanged under candidate permutation.
- Candidate-relative consequence, semantic path tokens, factor logits and
  utilities follow the same candidate permutation.
- Both fusion gates are zero initialized; legacy selection is unchanged before
  learning.
- The dedicated combination test passes under the locked Python 3.9 runtime.

## Launch audit

Folds 0--2 were first assigned to idle vla-zt GPUs 5--7. The third concurrent
loader was killed by the host memory limit before it created an output
directory, training batch, checkpoint or validation result. Its empty training
log was moved into a `failed_attempts` audit directory and retained. After one
Wave-14 fold naturally released approximately 105 GiB, fold 2 was relaunched
on the same GPU with the identical command and canonical output path.

When vla-zt2 recovered, its excluded Python-3.10 Wave-13 diagnostic processes
were stopped after exact PID/command validation. That released otherwise idle
resources, and Wave-15 folds 3--4 were launched there on GPUs 5--6 with the
same exact Python 3.9.25 runtime and shared immutable caches. No unrelated job
was stopped, no partial result was overwritten, and host assignment is not a
model or data variable.

After launch, the complete repository test suite was rerun with the same
locked Python and compatibility paths: `277 passed`, with warnings only. This
includes the combined representation test as well as checkpoint compatibility,
no-future input boundaries, candidate permutation, cached evaluation and
Navtest-audit regressions.

## Resource migration audit

At 2026-09-02 14:57 UTC, training use of `vla-zt` and `vla-zt2` was disabled
by operator instruction. Wave-15 folds 0--2 on `vla-zt` and folds 3--4 on
`vla-zt2` were terminated only after exact PID and command-line validation.
All five partial fold directories and logs were retained without modification
under the suffix `partial_vla_migration_20260902T1457Z`. At migration time,
folds 0--1 had completed two epochs and folds 2--4 had completed one epoch.

The trainer does not save a complete optimizer/scheduler resume state, so the
canonical five-fold run was restarted from epoch 0 rather than combining
partial and restarted histories. Folds 0--1 were assigned to `rl-zt4` GPUs
4--5, folds 2--3 to `rl-zt3` GPUs 2 and 4, and fold 4 was queued on `rl-zt4`
GPU 6 after the existing Wave-13 fold 4 releases host memory. The Wave-15
post-processing watcher was also moved to `rl-zt4`; its sweep GPUs are locked
to 4--6 and its refit/Navtest GPU to 7. Existing Wave-13 post-processing keeps
GPUs 0--3, so the two campaigns do not share a GPU.

No new training launcher remains on `vla-zt` or `vla-zt2`. A completed
Wave-14 statistical summary was allowed to finish, but its auto-refit launcher
was disabled before it could start training. Host migration changes neither
the immutable caches nor any model, optimizer, split, seed, epoch or promotion
setting.
