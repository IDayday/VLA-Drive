# Independent correctness review: fixed C1 boundary3584 signal support

Status: **PASS for this bounded, read-only analysis of existing evidence.** No blocking mathematical or identity error was found in the claims about these 32 scenes. This is **not approval for training, a new diagnostic run, population expansion, or reward changes**.

Reviewer: `/root/road_proxy_correctness_review`, independent of author `/root/consequence_next_hypothesis`.
Author commit `cafc25a6096703b78a83b1b0d803a480545111af`; integrated commit `08ecd377a01d536a409d52e9b9a0c38325fff658`. All five reviewed author artifacts matched their bytes in that exact integrated commit. The frozen original mathematical runtime was `83679937fe9702be39f0184154fdf9ebdc4eb98a`.

## Actual evidence, not a fixture substitute

The reviewer read the authorized real `RESULTS.pt` **once**: 3,441,888 bytes, SHA256 `8019dd29b17efa79c1fd1a7211bdf09a0b5be93931db75cd4db95ad9a3fb122f`. It contains the original four ranks × eight scenes × four candidates from C1 `boundary_003584`. All 128 real candidate records were checked against REQUESTS, ADMITTED, COMPLETE and the original four rank receipts. No scene was dropped, replaced or selected by its outcome.

The independent replay checks original rank/scene/candidate order, rollout ID, policy version and state hash, original float32 feedback rewards, runtime and scoring protocol, trajectory hashes, identical per-scene reference inputs, saved reference/executed-state hashes, valid scores, and native event traces. TRAIN provenance is bound to the original run and small data-identity receipts; the full training manifest was deliberately not opened. Neither DEV nor Navtest was accessed.

`REPLAY_RESULT.json` records the exact read ledger and numerical results. The replay ran 2,159 recorded assertions across the real records and calculations, plus two output-immutability/size assertions after result assembly. These assertion counts are not independent experimental samples or separate scientific tests. Nine synthetic controls supplement the real-data replay; they do not replace it.

| Independent real-data recomputation | Result |
|---|---:|
| Scenes / source logs / candidates | 32 / 32 / 128 |
| Valid / invalid candidate scores | 128 / 0 |
| C1 nonzero-advantage scenes / candidates | 16 / 64 |
| EP-only among active C1 scenes | 16 / 16 |
| Same-population C2 / C3 / C4 active scenes | 0 / 0 / 0 |
| NC / DAC / TTC / Comfort exactly one | Each 128 / 128 |
| D / C / T event-bearing candidates | 0 / 0 / 0 |
| Unordered within-scene pairs | 192 = 32 × 6 |
| Exact float32 score ties | 96 pairs, 16 scenes |
| Exact float64 score ties | Same 96 pairs, same 16 scenes |
| Safety-vector ties | 192 pairs, 32 scenes |
| Nonempty-event pairs under every tie condition | 0 |
| Temporal-difference fraction conditional on nonempty events | Undefined (`null`), denominator zero |

## Formula and attribution checks

The pinned official scorer defines NC and DAC as multiplicative factors and uses EP/TTC/Comfort weights 5/5/2, with DDC weight zero. The analysis correctly uses the **already scored, fixed-reference** EP. It does not normalize progress over these four candidates. Independent reconstruction with the official grouping of operations,

`R = NC * DAC * (5*EP + 5*TTC + 2*Comfort) / 12`,

matched these saved scores with maximum error zero. The author's differently grouped sum of the three already-divided terms has maximum error `1.1102230246251565e-16`, below its predeclared `1e-12` bound. This small difference is floating-point evaluation order, not a changed score definition.

The original FPO implementation uses float32 reward values, masks, a group sum/count and within-scene subtraction. Its actual selected function was loaded as an AST-only definition and run on the saved rewards; it exactly matched the author's CPU float32 mean-based center for all 128 entries. This verifies this CPU arithmetic replay, not a new CUDA bitwise or parameter-gradient experiment.

Each decomposed term includes the **same NC×DAC gate**. A varying gate cannot be causally assigned to TTC merely because the gated TTC term changes. Here all safety and comfort values are one, so only EP varies; the narrower EP-only statement is supported by the actual data.

The independent scalar-loop decomposition reproduced:

- `sum(A²) = 5.9478613771091204e-05`.
- Gated EP signed covariance fraction `1.0000011338555224`.
- Gated TTC and Comfort fractions both zero.
- Float32 rounding fraction `-1.1338555223760897e-06`.
- Maximum absolute rounding residual `7.5120938303197e-08`.
- Exact additive reconstruction after retaining that residual.

The negative and greater-than-one fractions are valid signed algebraic shares and must not be clipped. They are not causal percentages, parameter-gradient shares, BC/FPO gradient ratios, or Adam update contributions. The report correctly keeps those interpretations separate.

The C2/C3/C4 counts are the corresponding reward definitions applied to **this same C1 candidate population**. They are not an evaluation of independently trained C2/C3/C4 models. This distinction must survive any later summary.

## Event and denominator checks

All real road arrays retained 41 Boolean states and eight right-closed intervals; all were false. Original at-fault collision and TTC violation events were empty, with the correct `None` first times and empty-event summaries. The analysis did not substitute a terminal time for absence or interpret TTC projection offsets as physical event time.

Every scene contributes exactly `C(4,2)=6` pairs. The 192 pairs are correlated within 32 scenes, not 192 independent scenarios. Each tie condition has its own explicit denominator. Since no relevant event pair is nonempty, a conditional timing-difference rate cannot be estimated. A reported zero numerator is not evidence that terminal aggregation loses no temporal information elsewhere.

The supplemental controls cover same-terminal/different-time events, empty events, collision severity changes, TTC projection-vs-state time, bin boundaries, constant rewards, float32 ties and signed covariance cancellation. The reviewer independently verified the flow/time counterexample algebra: both physical points depend on all ten flow parameters, and `d(event)/d(theta_k)=0.5` for each. This disproves a universal one-to-one physical-time/Flow-index or nearest-token assignment. It is not an actual NAVSIM, LQR, or policy sensitivity measurement.

## Findings and boundaries

1. **No blocking error in the fixed-population conclusions.** Identity, full inclusion, formula, centered-value decomposition, tie counts and empty-denominator treatment agree with independent recomputation.
2. **No empirical safety-process comparison is identifiable here.** The population has no D/C/T event-positive pair. This is an absence of support in this batch, not an argument that safety-process supervision is useless.
3. **Do not infer gradient dominance or training changes.** The independently running gradient diagnostic must provide its own result. This review did not read its outputs, rerun it or alter its admission.
4. **No population-wide claim.** The report's full-TRAIN follow-up is a proposal, not executed or approved by this review. Any separate ledger aggregation needs actual version, exposure, unique-scene and event-support denominators; these 32 scenes cannot supply them.
5. **No new derivative claim.** The toy example is a mathematical counterexample. The official scorer remains an offline source of saved labels, not a differentiated module in this review.

## Resource and reproduction record

Executed once:

```sh
cd /mnt/project/VLA-Drive-st-diagnostics-final-correctness-20261018
timeout --signal=TERM --kill-after=5s 115s \
  /root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261018/consequence_signal_support_correctness_v1/check_existing_independently.py
```

The checker is not an automatically repeatable job: its result is immutable and the one authorized real-ledger read has been consumed. Re-execution requires a separate authorized read budget and output version.

Actual replay: 1.173481 CPU seconds, 1.169601 wall seconds, 425,603,072-byte peak RSS, one PyTorch thread; logical inputs 3,836,606 bytes. Additional review work read only small source/report text, remaining within the 16MiB total allowance. CUDA was not initialized. Model/checkpoint/image/GT/full-manifest/metric-cache reads, new score calls, policy rollouts, optimizer updates and dispatches were all zero. No implementation was changed; only independently owned evidence was added.
