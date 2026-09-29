# Current/future DINOv3 + GT-MAE campaign — ACTIVE IMPLEMENTATION

User instruction2026-09-29 supersedes the FLUX campaign. New worktree /mnt/project/VLA-Drive-ddp-dinov3-20260929; branch feature/ddp-current-future-dinov3-gtmae-20260929; base3832ab6c387065b2875f162b4442c1e555733c07. New artifact root /mnt/project/ddp-dinov3-artifacts/20260929. Prior workspace and evidence preserved; see INHERITED_STATE.md. Do not resume old controllers or driving checkpoints.

Current student uses original DDP ego FM,64 shared W, one physical-time DINO readout for0/1/2/4seconds and one interaction readout. DINO and GT-MAE are targets only. Current and future independently normalized; each scene requests one future horizon irrespective of missing labels. No FLUX in new formal path.

Assets to verify/reuse: generic Qwen2B revision89644892e4d85e24eaac8bacfd4f463576704203; downloaded generic timm DINOv3 L/16 revision30c1109559f65dea34316b0d4842d35c5771fe11; full GT-MAE milestone030 hash9730278f9c920de649536dec108226ac9d7f27430e265e8befac19896d58336c,11910updates; old pure current/ego and interaction train/dev caches. No compatible DINO training/cache result found yet; original user DINO recipe UNVERIFIED. Fallback must be explicit.

Authorized existing local and training-vla-zt2 GPUs only. Recheck pressure parent/worker identities before use; stop only verified pressure scripts for allocation, restore after exit. Other tasks untouched. New independent budget registered in execution_budget.yaml, formal plan frozen after measured throughput and train-only gradient calibration. No formal students or Navtest results yet.

Next: implement encoder/cache, physical-time head and separate losses; verify teacher identities; focused CPU and real GPU checks; cache targets; short fits/calibration; complete matched full-data training and dev/Navtest. Real updates in THIS campaign currently0.
