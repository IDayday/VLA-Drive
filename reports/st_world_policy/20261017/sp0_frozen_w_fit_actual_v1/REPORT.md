# S-P0 frozen-W capacity diagnostic: actual CPU fit

The original registered fit completed all 300 cycles/updates on the fixed 64-train/16-dev population, with 600 scene presentations and 57,600 queried patch exposures. All 64 training scenes were consumed. The 495,362-parameter readout used CPU only; the cached learned W stayed unchanged. This is a capacity diagnostic, not policy training or a planning-gain result.

| Split | Small-region positives / valid | PR AUC | ROC AUC | Depth valid | Depth MAE (m) | Depth AbsRel |
|---|---:|---:|---:|---:|---:|---:|
| dev | 291 / 7180 | 0.705765 | 0.983113 | 15922 | 7.533805 | 0.612282 |
| train | 1231 / 19463 | 0.832134 | 0.980306 | 64067 | 6.835687 | 0.571134 |

Wall time was 10.826 s; sampled process-tree RSS peaked at 793,411,584 bytes. No GPU allocation or native checkpoint reload occurred. Hash admission/completion was 173,933,668 bytes; process logical reads were 312,542,045 bytes. Physical reads (including mapped/library files) are recorded separately and are not interchangeable with these counters. Fit artifacts total 12,454,544 bytes. The owner process exited with code 0 and cleanup reported 1 verified / 0 remaining.

Small-vehicle labels are the fixed current raw-box/LiDAR-supported region proxy; unknown patches remain excluded through their original validity masks, not relabeled negative. Development depth has 15,922 valid / 20,736 grid positions and small-region classification has 7,180 valid / 20,736. This 16-scene development diagnostic is not the full 1,696-scene planning evaluation and is not a fresh blind test.

The result shows that this small readout fits useful region information. It cannot determine whether more W, finer targets or higher teacher resolution improve the learned representation without the other pre-registered matched arms. Depth errors remain substantial. No population, threshold or model selection rule was changed.

The original extraction supervisor failure remains immutable. Its independently accepted tensors were consumed through the reviewed TRAIN64-to-fit adapter; the successful downstream fit does not relabel the extraction process as successful. No automatic extraction retry, new Navtest, ACTIVE update, or large feature cache was performed.
