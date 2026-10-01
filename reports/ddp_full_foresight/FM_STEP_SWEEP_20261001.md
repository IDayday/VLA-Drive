# Frozen C1 100k Euler-step sensitivity

Status at launch: RUNNING, not a completed PDMS result. Source is fixed at
`771c802021e24d7ad3c3ff4f77c6de3824669edd`; training source remains
`d1d40854299b9599b2accc382bcfc4b676dd7623`.

C1 checkpoint identity:
`5e2ea076c8404ffb5aa1b2e6d10a2ce9455cbf481a971bdd7148c1a22416558a`,
100,000 training updates, seed42. All inference loads strict FP32 optimizer
masters, TF32 off, auxiliary heads removed, W retained. There is no optimizer.
Only the original Euler count changes to 1,2,3,5,8,10,15,20,30; dt=1/N and
the original 1000 time buckets are unchanged. Single ego/no scorer, common
token-hashed CPU initial noise seed42, complete12146scenes/136logs.

19 targeted CPU tests passed. Four real scenes passed bitwise trajectory parity
against archived original10-step exports. For every tested N, shared current
encoding also matched the original full predict_action path bitwise. This
allows each scene's Qwen conditioning to be reused within the live process.
No driving hidden-state cache is written or reused across checkpoints.

Registration:
`e518076e3da33dde313194bc93835c4c14c15d1b60a33f1e8fd54fef6c43421d`.
ControllerPID2748835; local8 + training-vla-zt2 eight GPUs. Both previously
completed training allocations released only exact-owned pressure parents;
C4 training and its milestone observer are untouched. Each host restores idle
reserves after its children finish. Three concurrent canonical CPU scorer
groups,16workers each, numerical threads1. Incremental cap100GPUh within the
existing6000GPUh campaign cap; every GPU exporter bounded to6hours. Loading,
failed attempts and inference are charged. Initial registration signature
mismatch failed before any GPU work; its evidence is retained externally.

Complete scoring uses the established official NAVSIMv1.1 CPU implementation
and original full precision caches from the audited protocol. No scorer,
oracle, quantized map, alternate progress formula or environment class filter.
Fresh10-step scoring must reproduce historical factor rows within1e-8;
archived C1@100k PDMS is89.2395355750. No new-score claim is made at launch.

External artifact root:
`/mnt/project/ddp-full-foresight-study-artifacts/20260929/fm_step_sweep_C1_100k_20261001`.
Each steps_NN directory contains all scene trajectories and full score CSV.
The completed run writes RESULTS.csv/RESULTS.json, paired log-cluster intervals,
full FP32 batch-one prediction latency on64fixed scenes and complete-scene
solver timings separately. Targets/weights/raw images remain outside Git.

Reproduce/resume from the immutable source after inspecting any stop marker:

```bash
cd /mnt/project/VLA-Drive-fm-step-sweep-run-771c802
/usr/bin/python3 -m tools.full_foresight.fm_step_sweep run \
  --registration /mnt/project/ddp-full-foresight-study-artifacts/20260929/fm_step_sweep_C1_100k_20261001/registration.json \
  --attempt 2
```

Do not launch a duplicate while the controller holds campaign.lock. This is
a user-requested fixed-weight Navtest inference diagnosis. It does not change
locked10-step training/milestone evaluation or automatically promote the best
test-measured N. One inference/training seed cannot establish general stability.
