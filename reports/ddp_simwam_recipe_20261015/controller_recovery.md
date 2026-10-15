# Evaluation controller recovery, 2026-10-15

The first update5000 export failed before inference: the controller on
tsinghua-zt2 had no SSH alias configuration, so it could not resolve
`training-vlawm-tsinghua-zt`. The old milestone controller exited with SSH
status255. Its error, dispatch, log and progress records are preserved under
`/mnt/project/simwam-recipe-20261015/controller_recovery_20261015T0530Z/`.
The failure affected evaluation scheduling, not either training process.

Queue control now has an explicit `--administrative-local` mode on the existing
local control host. It requires `CUDA_VISIBLE_DEVICES=''`, checks the current
global host prohibition, and validates SSH destination identities before waiting
for a checkpoint. GPU workers remain restricted to the four authorized Tsinghua
hosts; the original experiment-worker contract is unchanged. Credentials stay on
the local control host. No remote private key or global SSH configuration is
installed. Negative admission checks reject vla-zt2 and local experiment workers.

The idle final controller was stopped by its exact PID/start-time identity
after verifying it had no children and had not opened Navtest. The two training
processes were not signalled. Both queues use their original exclusive locks.
Previously saved 5k/10k/15k/20k checkpoints retain their identities; recovery
does not recreate rollouts or restart training. Inference still uses frozen
source `12b8e0d75691d7bed04ab02072020e130f711a3b`, the original evaluator,
two separately reported ODE grids, seed2101 and complete populations.

Executed route checks covered both milestone hosts and all four final-evaluation
hosts. The final `serve --once` check kept the gate closed until both actual20k
endpoints and complete DEV stages exist. The scientific recipe, model exports,
candidate budgets and endpoint selection rule are unchanged.

The user's withdrawal of vla-zt2 is enforced in the global scheduling policy.
The unrelated G2 run saved its complete native update89574 and exited; its
existing migration owner is recorded in the project AGENTS.md. This task does
not start a duplicate G2 continuation. The current R_SOURCE/R_CANONICAL runs
are on tsinghua-zt4/zt3, respectively, and remain authorized.

Recovery has now completed both fixed5000 populations: DEV1696 and TRAIN512
for both variants, with all32 shard records complete and direct native-path
parity checks passed. [The completion receipt](recovered_evaluation_receipt.json)
binds their identities. Both queues remain alive; the milestone queue has moved
on to the already saved10k checkpoints and the final Navtest gate remains closed.
