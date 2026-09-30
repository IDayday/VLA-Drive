Initial registration for the user's request to evaluate the latest complete C0/C1/C4 checkpoints on full Navtest.

| Candidate | Frozen checkpoint | Actual updates | Scene exposures | DINO target /pool /W |
|---|---|---:|---:|---|
| C0 | periodic_051400 | 51,400 | 1,644,672 | 128×96 /1×1 /144 |
| C1 | periodic_051400 | 51,400 | 1,644,672 | 256×192 /2×2 /144 |
| C4 | periodic_026000 | 26,000 | 831,936 | 512×384 /4×4 /144 |

The snapshot is fixed at the request time, not repeatedly advanced while evaluation runs. Complete checkpoint files and run identities are independently preserved by hardlink before rolling retention removes their original directory entries. No profile, old driving checkpoint or other weights are substituted. All three formal students retain the same four training tasks and unchanged training source `d1d40854299b9599b2accc382bcfc4b676dd7623`.

The explicit latest-checkpoint request supersedes the earlier pending exact40k request. Common40,800 snapshots remain preserved as historical assets; they will not be evaluated or relabeled as this new request. C4 has a shorter training length than C0/C1, so the three raw latest scores are not a matched-endpoint resolution comparison. C0/C1 can be paired at the same51,400 updates; C4 progression can be paired with its own completed18,400 result. No Navtest-driven configuration, training length, loss or model selection follows from this measurement.

Evaluation reuses the validated canonical NAVSIM v1 runtime and original full-precision cache snapshot: all12,146scenes /136logs, FP32 optimizer masters restored strictly, FP32 inference, TF32off, seed42,10FMsteps, one executed ego, no scorer/oracle. Pure current three-front images/navigation/allowed ego inputs; auxiliary heads removed and144W retained. Offline ego-fit labels are separate from inference. Every final row must match the original scene/log/metric-cache hash. Prior complete development evidence at50k for C0/C1 and25k for C4 is explicitly prior evidence, not a fabricated matched-checkpoint development result.

Topology: eight capped independent exporters per model, sharing each model's authorized training host GPUs0–7;24 exporters total. Three canonical16-worker CPU scorers consume predictions asynchronously on the local64-physical-core host. Training and controllers remain running and immutable. Eight pressure parents on rl-zt2 were verified against allocation ownership, exact command, CUDA GPU, process group and log, then released under the user's existing authorization; no trainer/unrelated process was signaled. No teacher retraining or DINO extraction.

Artifacts: `/mnt/project/ddp-full-foresight-study-artifacts/20260929/navtest_latest_three_20260930`.
Branch: `experiment/ddp-full-foresight-navtest-latest-three-20260930`.
Results are pending. Training progress and test performance are distinct; no planning improvement is claimed before full completion.
