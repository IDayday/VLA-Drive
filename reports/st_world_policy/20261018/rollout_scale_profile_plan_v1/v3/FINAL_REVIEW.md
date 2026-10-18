# Final bounded PROFILE systems review

**PASS_BOUNDED_240_UPDATE_PROFILE_ONLY.** This supersedes the pending launch status of the preserved pre-review, without changing its historical record. The exact runtime is `751fcfa2b41549ae16f7b3204aef8dd18b912016`, the PROFILE registration is `a67ce92a7b5d366aea21b13cd7ae2e470f4d8f91bb089a9326311133a008cc08`, and the v2 launcher is `1f2828ffed12d1f27bd78b143fb97ee65000c79f90a7742a9e593fb5564c9e30`.

The real four-rank SMOKE exited zero after four updates and 512 valid official scores. All logged losses and gradients were finite, all four steps recorded FP32 master updates, and the frozen encoder was unchanged. Its 61 monitor samples recorded no monitoring error or resource stop. Process-tree RSS peaked at 115.91 GiB, driver GPU use at 17.68 GiB per GPU, and shared free space remained above 3.24 TiB. The complete run took 312.03 seconds including loading and saving. Four update times are functional evidence only, not a steady-state benchmark. The independent correctness audit verifies the native final checksum and exact recorded normalization tensors; this resource review does not duplicate model/rollout reads.

The original SMOKE ended as `PAUSED` with `BUDGET_PAUSED`, as intended by the four-step diagnostic limit. It is not a completed 4096-step formal training run. No new GPU resume was executed.

SMOKE retained three native states at steps 0, 2 and 4. The planned PROFILE adds two states, at 0 and 240. Their aggregate payload estimate, using actual SMOKE receipt sizes, is 41.22 GiB across five directories. This stays within the 64 GiB native reservation inside the 96 GiB total study limit; a sixth directory remains the bounded contingency. No checkpoint is deleted. This estimate is a capacity check, not a measured PROFILE result.

The one approved PROFILE uses zt5 GPUs 0–3 and CPU 0–15: world4, micro4, accumulation2, global32. It starts from the original S0 and uses the original 4096-update scheduler clock with warmup100. It consumes the first 240 updates of the full 101592-scene population in original sampler order, with two windows of 20 warmup plus 100 steady updates. The official-score budget is 30,720 primary calls and 31,744 including retries. The invocation budget remains 7,200 seconds, with its owned TERM at 7,500 seconds and KILL backstop 180 seconds later.

Fresh host checks must pass again before and after allocation locks, and the exact independent correctness review must accompany this receipt. The v2 monitor sums both stage output trees and stops only the owned run for excessive output/native count, unknown selected-GPU PIDs, resource excess or monitoring failure. It never signals another job. The v1 SMOKE controls remain unchanged.

No formal training, automatic retry/resume, additional candidate, topology search, or subsequent experiment is approved. PROFILE weights cannot initialize a formal run. The reviewer has not dispatched this PROFILE.

After the Lead creates the exact independent-review admission, the existing entry is:

```bash
/root/miniconda3/envs/ddp/bin/python \
  /mnt/project/st-world-policy-20261016/resources/launch_rollout_scale_diagnostic_v2.py \
  launch --stage PROFILE \
  --admission /mnt/project/st-world-policy-20261016/profiles/scene_center_rollout_std_v1_fulltrain240/ADMISSION.json
```
