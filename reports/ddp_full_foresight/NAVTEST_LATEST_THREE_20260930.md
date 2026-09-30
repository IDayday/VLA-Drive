All three requested latest complete checkpoints finished full Navtest. Each contains12,146unique scenes /136logs with zero inference, scoring or ego-fit failures. The snapshot is fixed at request time; ongoing training weights were not substituted during evaluation.

| Model | Updates | PDMS /100 | NC | DAC | TTC | EP | Comfort | Zero scores | Ego ADE/FDE (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 51,400 | 88.3269 | 98.2505 | 96.3115 | 94.7802 | 82.8330 | 99.9918 | 651 (5.3598%) | 0.8997/1.9253 |
| C1 | 51,400 | 88.1848 | 98.0981 | 96.3280 | 94.4097 | 82.7722 | 99.9918 | 664 (5.4668%) | 0.8975/1.9094 |
| C4 | 26,000 | 86.7090 | 98.3575 | 94.8296 | 94.6978 | 80.8845 | 99.9753 | 803 (6.6112%) | 0.9775/2.0660 |

C1−C0 at the common51,400-update checkpoint is-0.1421PDMS points; paired136-log bootstrap95% interval[-0.4429,+0.1782]. This single-training/inference-seed result does not distinguish an overall winner. C4 has26,000updates and cannot be ranked as a matched-endpoint resolution comparison against these51,400-update models.

| Same-run progression | PDMS change /points | Paired log95% interval /points |
|---|---:|---:|
| C0:31,600→51,400 | +0.7269 | [+0.2027,+1.2898] |
| C1:31,600→51,400 | +0.9961 | [+0.3146,+1.7548] |
| C4:18,400→26,000 | +3.9087 | [+3.0589,+4.7762] |

All three improve over their own prior measured checkpoints under the exact matched protocol. Improvement is not uniform across safety factors: C0 NC/TTC decrease relative to31,600 while its progress increases; all submetrics and zero scores remain visible. This evidence supports continued learning, not auxiliary-task attribution, convergence or training-seed stability. No resolution, loss, scheduler or checkpoint selection changes follow from Navtest. Full100k/five-run endpoints, other candidates, training-seed repeats and task ablations remain unfinished.

[Combined complete36,438-row CSV](navtest_latest_three/ALL_COMPLETE.csv) · [C0 CSV](navtest_latest_three/C0_complete.csv) · [C1 CSV](navtest_latest_three/C1_complete.csv) · [C4 CSV](navtest_latest_three/C4_complete.csv). [Summary](navtest_latest_three/SUMMARY.csv), [RESULTS.json](navtest_latest_three/RESULTS.json) and query/scene/log progression files preserve all official factors, ego errors, trajectory/cache hashes and identities. Actual launches: [COMMANDS.json](navtest_latest_three/COMMANDS.json).

Protocol: strictly restore learned FP32 optimizer masters; all inference parameters/compute FP32; TF32off; training/inference seed42; original10-step FM; one executed ego; no learned scorer, oracle or online teacher. Current three-front images/navigation/allowed ego only;144W retained and auxiliary heads removed. Frozen models are C0/C1 periodic_051400 and C4 periodic_026000. The latest-three request explicitly supersedes the earlier pending exact40k replacement choice; unused40,800 snapshots remain preserved.

Canonical NumPy1.26.4/SciPy1.13.1/Shapely2.0.7 and NAVSIM v1.1 are unchanged. Every row matches the original scene/log/metric-cache SHA256 snapshot. Full-precision maps/full environment/reference+oneprediction scoring are retained, addressing the recorded audit risks. No EPDMS mixing or legacy quantized-map/erroneous fixed-progress path. Label-side ego fit follows original eight successive keyframes, retaining all scenes including the known36 timestamp-deviation cases. Prior complete dev50k(C0/C1) and25k(C4) are explicitly prior evidence from the identical runs, not fabricated51,400/26,000 development results.

Twenty-four capped exporters shared the three authorized training hosts, eight per model; three16-worker canonical CPU scorers ran asynchronously. Summed shared-GPU process-hours23.1911 include startup/loading. They are not additional exclusive physical GPU occupancy or isolated deployment latency. Per-model wall times and memory are recorded below; CPU attempts, including any safe completion-merge resume, are preserved separately.

| Model | Export wall minutes | Shared-GPU process-hours | Peak allocated/reserved GiB per process |
|---|---:|---:|---:|
| C0 | 58.52 | 7.7492 | 11.956/12.373 |
| C1 | 58.46 | 7.7310 | 11.956/12.373 |
| C4 | 58.78 | 7.7108 | 11.956/12.373 |

Eight rl-zt2 pressure parents were released only after verifying allocation ownership, exact command, GPU environment, process group and log. No trainer or unrelated process was signaled. Training continued on immutable source/controllers. No teacher training or DINO extraction was rerun; evaluation performed zero optimizer updates.

Branch: experiment/ddp-full-foresight-navtest-latest-three-20260930. Training source:d1d40854299b9599b2accc382bcfc4b676dd7623. Inference/scoring/analysis source:46a6a8096ba6a7a90ddb18fae52c31c207f277b1. Source/result commits are distinct. Weights, raw GT/images, teacher latents and caches are not uploaded.

Reproduce same-run progression into a new directory:

```bash
cd /mnt/project/VLA-Drive-navtest-eval-46a6a80
/root/miniconda3/envs/ddp/bin/python -m tools.full_foresight.compare_checkpoint_updates \
  --first /mnt/project/ddp-full-foresight-study-artifacts/20260929/navtest_latest_three_20260930/C0_scores_v1 \
  --baseline /mnt/project/ddp-full-foresight-study-artifacts/20260929/navtest30k_20260930/C0_scores_v1 \
  --output /mnt/project/ddp-full-foresight-study-artifacts/20260929/navtest_latest_three_20260930/C0_progression_reproduced
```
