# Minimum remaining shape checks, without repeating the topology matrix

This checklist separates actual measured throughput from compatibility evidence. It is not a new dispatch, blanket PASS or request to rerun all profiles. Final source is the new master-integration runtime2d3e4f5a; old240-update profiles remain bound to their actual58f85044/ab2184bf sources.

| Cell | DINO target image / pooling | Actual target grid | W tokens per view / total | Sampled positions per view / total | PCA384 output at micro4 | Existing full timing |
|---|---|---|---|---|---|---|
| S-P0 | 192×256 /2×2 | 6×8 | 48 /144 | 48 /144 | [4,144,384] | Not separately measured |
| S-P1 | 288×384 /3×3 | 6×8 | 48 /144 | 48 /144 | [4,144,384] | Not separately measured |
| S-P2 | 288×384 /none | 18×24 | 48 /144 | 96 /288 | [4,288,384] | 240 updates,8GPU and concurrent2×4 |
| S-P3 | 288×384 /none | 18×24 | 96 /288 | 96 /288 | [4,288,384] | 240 updates,8GPU |

These are the actual strict `SPATIAL_CELLS` contracts with the campaign's selected384 PCA width. The DINO target resolution is not a change to the original Qwen input image contract. Measured padded language lengths are1930 for W48 and2074 for W96. Actual masks and valid lengths must still be inspected; padded length is not a valid-token count.

The smallest additional execution is **one to four real optimizer updates for each of S-P0 and S-P1**, under an explicitly approved small qualification. Use the final clean source and actual teacher namespace; keep original microbatch4/global32 via the selected available rank/accumulation topology. This is a real backward/master/shape check, not another8GPU-vs2×4 study. If a final-source qualification already has those exact contracts, reuse it rather than rerun.

Each check needs:

1. Actual teacher manifest, image/grid/pooling, storage dtype, source hashes and complete chosen population coverage. P0 andP1 share6×8 shape but have different image/pooling/source identities. Storage dtype must be read from each actual manifest; shape equality does not authorize cache reuse or imply identical supervision precision.
2. The fixed PCA basis and namespace/channel-scope binding. Same384 width alone does not prove the same transformation. Do not refit PCA during this check or count future/held-out data as fit statistics.
3. Three real views/current observations, finite FP32 post-PCA targets, correct48-position masks and readout shape. Full-valid target denominator at global32 is32×3×48×384=1,769,472; for P2/P3 it is3,538,944. With real missing/invalid positions compute the actual summed mask denominator instead of forcing these full-valid constants. Ego scenes and repeat8 remain unchanged.
4. Finite original ego/current losses, real gradients reaching the language/W/readout/action paths, an actual FP32 master update, and no teacher-model forward inside training. Preserve the original optimizer groups, dropout, DDP global reduction and masks. Existing negative/empty-rank tests support the denominator logic but do not replace these real-data checks.
5. Actual allocated/reserved memory, batch/sequence lengths and per-rank calls. Record short per-update wall plus setup/save separately; mark the throughput as a short qualification, not200 steady observations.
6. Native completed checkpoint/source identity for the small run, with the new master helper's fresh initialization receipt. Reuse the independent final-source native recovery evidence for the unchanged checkpoint/optimizer machinery; do not relabel old rounded-master checkpoints as canonical new initialization.

The shared original encoder, DiT repeat8, backend and full-update loop are unchanged relative to the measured topology. Fewer readout positions plausibly reduce work, but this is not a measured P0/P1 speed and cannot justify copying P2's15.63scene/s into their result table. Eight-GPU current-only topology/resource evidence may be cited with the explicit source migration note and real cell-specific checks. Initial FP32 capture/restore adds startup CPU work and can change the learned trajectory; its new allocation cost and numerical qualification must be reported separately.

No new240-update timing matrix, extra seed, model simplification or future/history experiment is required for this narrow remaining check. Temporal/history and C1 MC50 have additional paths and cannot inherit this current-only qualification. Formal reviews still bind the actual new source, population, teacher, PCA, optimizer and resource identity; this document signs none of them in advance.
