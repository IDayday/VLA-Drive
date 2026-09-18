# Continuous training and independent evaluation

The previous controller ran train-to200, CPU export, development evaluation and
fresh distributed model initialization sequentially. On2026-09-18, F update200
was logged at06:15:05 UTC, checkpoint COMPLETE at06:16:05, full1696-scene evaluation
completed at06:29:57, and the next launcher started at06:31:27. GPU availability
between these stages did not mean optimizer updates were being performed.

The new optional `async_evaluation` plan uses the unchanged native trainer in
one continuous session up to the fixed2000-update budget. Native complete
checkpoints are still saved every100; the controller consumes100,200,400,...,2000
without controlling training progress. Checkpoint100 and200 artifacts already
completed by the previous controller are reused through their native validators.

Full distributed checkpoint saving still synchronizes ranks to capture weights,
optimizer, RNG and pending behavior consistently. It cannot simply be delegated
to another GPU reading a changing live model. No asynchronous checkpoint writer
or altered numerical kernel is introduced here. Only export and evaluation are
decoupled: CPU export reads an immutable COMPLETE checkpoint, followed by the
same single-candidate ODE and official NAVSIM-v2 one-stage evaluator.

F and U share one serialized background evaluation allocation: training-rl-zt2
GPU7, logical CPUs16–19/80–83, one-thread BLAS/OMP. This node now has two F training
GPUs5/6 plus one evaluator, below the user's4-GPU cap. rl-zt4 devices6/7 remain
reserved. Allocation checks include evaluation slots and prohibit overlap with
training. Evaluation launches require the assigned GPU to be idle; no external
process is killed or displaced. Final five-seed evaluations can use each group's
training allocation only after that group's native training has fully exited.

The evaluator can lag training. Every prescribed checkpoint remains queued; no
checkpoint is skipped, and best selection still uses full seed42 dev EPDMS at
the predetermined200-update boundaries, with earliest ties. Final outputs wait
for all evaluations; reaching2000 alone does not imply experiment completion.
Evaluation failure cancels this experiment's supervised jobs, retaining complete
checkpoints and failed evaluation attempts. The48-hour continuous-job deadline
replaces the8-hour deadline appropriate to short segmented jobs.

`adoption.wait_existing` allows a replacement controller to attach to the exact
SHA-bound existing train-to400 specs, checking group/config/output/commands,
original deadlines and independent supervisor exits. It never replays active
updates. The current train-to400 sessions can finish normally, then resume their
complete400 checkpoints directly to2000. Actor, optimizer, topology, randomness,
data, reference, parameter contracts and native executable remain unchanged.

Validation:88 affected CPU control/evaluation regressions PASS, exit0. Coverage
includes actual paired CLI execution/restart in both scheduling modes, progress
while evaluation is blocked, latest-checkpoint selection, failure cancellation,
file mutation rejection, adoption identity, and user GPU caps. These are control
tests, not new BF16 mathematical qualification. Existing measured GPU gates and
both historical chunk/layout failures retain their original status. New cluster
sources and this tested plan require a newly source-bound release.

Live deployment receipts and the source-bound release directory are recorded
after handover. Future ordinary restart command from the repository root:

```bash
PYTHONPATH=$PWD/navsim:$PWD /root/miniconda3/envs/ddp/bin/python -m scripts.cluster_flow_grpo.paired --plan configs/cluster_flow_grpo/paired_world16.json --resume
```

`--adopt-running` is only for an intentional controller handover with existing
supervised jobs; it takes a JSON mapping from each variant to `spec` and its
SHA256. Ordinary resume uses native latest-complete checkpoint selection.

Live handover completed at approximately06:51 UTC. Controller PID1341050 replaced
only PID1331113 and its identified idle compiler children; all five existing
native torchrun supervisors and their training process groups were retained.
`async_handover_receipt.json` records exact identities and signals' targets.
Both training logs continued increasing after replacement, with reward_errors0.
`async_live_handover.json` records the measured updates and unchanged native PIDs.
The first actual sidecar evaluation at400 is pending; CPU concurrency tests must
not be described as a completed real GPU overlap benchmark.

Code commit1668316, source-bound release65c007375ca44e201b55f02f2a26186f9a43f10ca07b9da21008a0b6ff988271.
Both native17-gate records reuse unchanged actual GPU evidence, plus the new
88-test control regression. An additional17 affected tests passed in a clean
detached worktree at1668316, exit0; its JUnit file is `async_clean_checkout.xml`.
No full-data verification scan, model-path change or new GPU numerical claim
was introduced by this deployment.

The new controller log is `runs/asynchronous_evaluation_v1/controller.log`.
Per-variant `continuous_training.json`, `artifact_activity.json`,
`orchestration_progress.json` and `selection.json` separate training, export,
evaluation and best-selection progress. During adoption, the already-running
segment's original `*_to400_*.control/node*.json` remains the live worker status.
