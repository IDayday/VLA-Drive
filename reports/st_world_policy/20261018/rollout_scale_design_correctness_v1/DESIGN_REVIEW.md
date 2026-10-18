# Independent review: scene centering followed by rollout-wide scale

This is a mathematical design review, not an implementation, experiment, resource or training approval. The reviewer is `/root/road_proxy_correctness_review`. No model, image, GT, metric cache, saved reward payload or full TRAIN analysis output was read for this task. No GPU, scoring, training or dispatch occurred. The proposal binding and final disposition are in `DESIGN_REVIEW.json`.

## Existing call path

The audited files are byte-identical to frozen runtime `83679937fe9702be39f0184154fdf9ebdc4eb98a`; exact hashes are in `CURRENT_SOURCE_BINDING.json`.

- `fpo_objective.py::group_centered_advantages` performs the original FP32 sum/count centering, requires complete valid feedback by default, and detaches its output. G=4, finite [0,1] constant reward groups remain exactly zero. This statement is about the actual fixed G=4 contract; it is not a promise about arbitrary group sizes and numerical ranges.
- `train_flow_policy.py` verifies one shared old-policy identity, scores the full optimizer rollout, validates feedback and freshness, then centers rewards at line 850. The new scale belongs after that complete validation and before the loop slicing optimizer microbatches. It must not be applied separately inside `FlowTrainingModule.forward` or the CFM MC chunks.
- `FlowTrainingModule.forward` preserves BC and FPO as separate terms. FPO receives `global_count=4*global_scenes`; the loss is local sum times world size divided by that count. Rank-averaged gradients and accumulation without a second division recover the global candidate mean.
- `validate_rank_batches` rejects empty rank tails and unequal backward-call counts in the existing trainer. A mathematical statistics helper can accept an empty local shard, but that does not make the existing end-to-end trainer support new empty-rank topologies.
- The saved upstream `akanazawa/fpo@418c2554` source normalizes its whole GAE rollout before minibatching with `(A-A.mean())/(A.std()+1e-10)`. This proposal deliberately differs: navigation scene centering stays, numerator is not globally recentered, and scale moments use FP64. It is not exact reproduction of upstream MDP/GAE training.

## Mathematical findings

For detached original centered advantages a, one optimizer rollout contains all complete valid G=4 groups across ranks, including zero groups. N is the number of candidates, not the number of nonzero candidates, scenes, MC50 points, physical trajectory times or Flow steps. The denominator is the correction=1 standard deviation, computed from two-pass FP64 global central moments, plus 1e-10 after the square root. The numerator remains the original a and output returns to FP32.

This definition is internally consistent. All-valid/all-zero rollouts produce zero policy signal while retaining BC. Empty populations, incomplete real feedback and nonfinite values must fail before backward. A nonzero constant a with zero standard deviation is rejected rather than silently amplified. These are validation requirements, not permission to omit difficult scenes.

The FP32 counterexample in the proposal was reproduced: rewards `[1,1,1,1-2^-24]` center to `[0,0,0,-2^-24]`. Adding a second constant scene then subtracting the global residual mean would inject positive values into all four originally zero advantages. Using global centering only to calculate the variance does not inject that signal into the numerator. It does not, however, remove the original residual inside the nonconstant scene.

Ignoring the final FP32 cast, a fixed positive scale multiplies the entire FPO surrogate and its gradient, including its clipping branch, by that same scale. Candidate ordering, within-rollout relative gaps and zero entries are unchanged. BC is unchanged. A scalar cannot improve the policy/BC gradient cosine. The old-current ratio itself is independent of this advantage scale.

Different rollouts have different scales. Two batches whose original advantages differ by 100x can have almost equal normalized advantages. Consequently this adaptation changes relative policy weights across batches and against BC. It is not an unbiased correction, absolute gap preservation across the training distribution, a safety preference or proof of better optimization. Including zero groups can increase the normalized amplitude of an active candidate; the loss still divides by the complete N. This does not prove that the total gradient of a sparser batch increases, because its active fraction, directions and model Jacobians also matter.

Current feedback reuse is one optimizer update and the old/current snapshots agree before that update. Thus ratio is approximately one while gradients are computed. Ratio clipping is not a bound on the movement after the optimizer update. Adam state, joint BC/FPO clipping, and subsequent on-policy data can change materially even when only this scalar rule changes.

## Distributed and recovery requirements

All ranks must finish local input validation and a common failure vote before any statistics collective. Wrapping a helper that can raise before its first all-reduce inside an outer `collective_call` is insufficient: one rank may enter the outer failure collective while another has already entered the statistics collective. The proposal must require matching collective order, then common N/sum, common mean, and common second central moment. There is no rank-local normalization or averaging of rank standard deviations.

The same detached scale must be reused by every microbatch and all MC chunks of that optimizer rollout. At the existing 24-scene tail, N=96 and the sample-variance divisor is 95; policy and BC retain denominators 96 and 24 respectively. No padding, duplication, drop or accumulation divisor is introduced. Empty local statistics shards are mathematically defined, while the existing trainer's topology restrictions remain enforced.

The normalized method is a new identity. Original defaults must not silently change. Record its definition, numeric execution, old-policy/feedback identities, N, moments, scale, exact raw/new advantage hashes and consumed boundary. A resume must reconstruct the same incomplete rollout's scale and verify it, never borrow a previous update's scale. Real distributed error propagation, numerical reduction tolerance and resume are implementation tests still to be done; the CPU evidence here does not certify them.

## Evidence executed here

`test_design_math.py` contains a small independent mathematical reference, not production code. Eight pytest cases passed in 2.044 child CPU seconds with 481,619,968 bytes peak child RSS; full command and output are preserved. They cover:

- The actual legacy centering function, the one-ULP global-recentering counterexample, constant score 0/.5/1 groups, and all-zero policy output.
- Incomplete feedback, empty/singleton/nonfinite/trainable input and zero-variance nonzero input rejection.
- Full population vs rank partitions at 32, 24 and 5 scenes, an empty local statistics shard, and correction=1 vs population correction=0.
- Positive-scale surrogate loss and gradient identities using the actual FPO objective; unequal rank slices and uneven microbatch sums with exactly one rank average.
- Cross-batch weight changes, inequivalence of separate microbatch scales, zero-group inclusion, and epsilon placement outside the standard deviation.

There was no actual distributed process group. Partition tests verify mathematical aggregation, not collective failure behavior, NCCL order or topology-independent bitwise identity. FP64 moment comparison used synthetic independent references; it does not authorize an arbitrary tolerance for real CUDA execution.

## Remaining decision boundary

The observed boundary3584 policy/BC norm ratio is contextual evidence only, not a target multiplier or a full-training gradient estimate. Complete TRAIN analysis must first establish per-update scale distributions, nonzero/zero support, the four fixed version windows, and roundoff and safety/EP-only strata. A single pooled RMS over all updates is insufficient because the proposed denominator is recomputed per rollout. Empty event strata remain undefined evidence, not absence of risk.

If those artifacts support proceeding, the next step is a narrowly versioned implementation and independent CPU/distributed correctness checks, followed only by a separately admitted bounded profile. No coefficient, epsilon, sampling, reward, safety gate or model-architecture search is approved here. No formal training is approved by this document.
