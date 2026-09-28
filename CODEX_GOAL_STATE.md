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

## Live stage1 status

Initial implementation pushed and verified atd3c05d1. Full teacher data prepared0failures,101592train/1696dev, log-disjoint. Teacher512small-fit COMPLETE; metrics inreports/ddp_shared_foresight/teacher_preflight.8real resume-test updates,512small-fit updates. Full teacher from independent random init RUNNING PID1346936 localGPU0, frozen source/worktree d3c05d136949ecc14b1a8312fa12bf966ae03643 /mnt/project/VLA-Drive-foresight-run-d3c05d1. Run: /mnt/project/ddp-foresight-artifacts/20260928/teacher_full30_v1; exact command teacher_full30_launch.json. DO NOT duplicate or edit frozen source.11910plannedupdates,30epochs. Full teacher resume must use same source, original command plus --resume --acknowledge-stop after checking live status.

Student GPU preflight v1 FAILED before model construction (environment float conversion),0updates. Fixed at2069df2; rerun uses newrun ID and frozen source. FLUX download401remains the only known external dependency; teacher and other implementation continue. Old stopped campaign remains sealed.
