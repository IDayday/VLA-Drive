# DDP shared foresight + GT trajectory MAE — IMPLEMENTING

New authorized objective: current-camera Action-Only DDP with64 shared W queries; train-only deterministic future FLUX VAE regression and masked GT vehicle trajectory teacher latent regression. Deploy original ego-only DDP action head. No joint actor generator, detector or old driving weights.

Branch: feature/ddp-shared-foresight-gtmae-20260928
Worktree: /mnt/project/VLA-Drive-ddp-foresight-20260928
Code reference: a68420722891ba690852907f096f57381a81b1af; historical tools remain legacy and are not student modules.
Artifacts: /mnt/project/ddp-foresight-artifacts/20260928
Old campaign remains STOPPED. This new campaign must never resume its controller or weights.
Resources: local8 A80080GB free; vla-zt2 GPUs4–7 free. User additionally permits terminating verified GPU stress scripts when more capacity is needed. Never stop unrelated real workloads.

Completed: isolated branch; initial source/resource checks; confirmed legacy QwenOFT eagerly imports video/depth and registers unused world tokens — fixing actual Action-Only path.
Current dependency: official FLUX.1-schnell VAE download is gated401; requested an authorized local path or configured HF access. Do not bypass repository access restrictions or silently change teacher.
Next: core MAE and foresight modules, focused tests, real vehicle data, full teacher training, VAE targets, student calibration and complete matched campaign. CPU/data/teacher work continues during VAE dependency resolution.
Real training updates in THIS campaign:0. No results yet. Short tests never count as complete training.
Push this branch after tested code stages; verify remote SHA. Do not upload weights/cache/raw images or private annotations.
