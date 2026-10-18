# Completed gradient diagnostic: numerical result usable, supervisor failure retained

Independent reviewer: `/root/road_proxy_correctness_review`.

**Conclusion:** the immutable mathematical `RESULT.json` is usable for the registered, fixed C1 boundary3584 gradient comparison. The external supervisor did **not** complete successfully: its original `STOPPED_NO_AUTO_RETRY` / `Missing live RSS counter` receipt remains a failure. Neither result is overwritten or relabeled. No model execution is requested or repeated by this review.

Exact bindings:

- Frozen diagnostic source: `83679937fe9702be39f0184154fdf9ebdc4eb98a`.
- Registration: `6a48f942e8b57f568d74be0bd92cab389daa6140d0a0307896acd40da3358770`.
- Shared-capacity wrapper: `8afa0e9a9a0cdfe7b974081dd11fc5c264474fd5a081d7376781903567a6b579`.
- Mathematical result: `/mnt/project/st-world-policy-20261016/audits/consequence/gradient_balance_boundary3584_v3/RESULT.json`, SHA256 `7ffd159b3dda244b40dec03ad09c3142f47bc50c395d1be70bda9127c8203a2b`.
- Failed supervisor receipt: `gradient_balance_boundary3584_v3_controls/RUNTIME_ATTEMPT/SUPERVISOR_RESULT.json`, SHA256 `80aedaec45221c99c903ec01442d50f9fb2cecfcfac76cd96f1c79acebaa937c`.

## Why the mathematical result is complete

The previously reviewed diagnostic source writes `DIAGNOSTIC_COMPLETE` only after all four ranks complete, original conditions and ten-step Euler outputs match, old/current CFM50 match, ratio equals one, private/old parameter and buffer hashes match, frozen encoder gradients remain absent, and caller RNG is restored. The final immutable JSON uses the existing atomic temporary-file/fsync/replace writer. The child prints its final status only after that write.

The actual result contains all four rank reports, eight original encoder microbatches, eight BC and eight policy backward contributions, eight `bit_exact` current/old CFM summaries of 800 entries each, the original native checkpoint at completed3584, and cursor epoch1/offset409 with 32 indices. The recorded source, official-feedback receipt and scene order bind to the separately reviewed EP-only analysis of the **same** 32 scenes. It is not boundary4095 or a different feedback population.

The result records `original_weights_unchanged=true`, `caller_RNG_unchanged=true`, zero optimizer steps, zero new score calls and no metric-cache read. These statements are supported by the checked source's fail-closed completion path and the actual complete receipt. This reviewer did not independently reopen large model/RNG/checkpoint assets or rerun the model, and does not claim an additional live-state comparison.

The child log ends with `DIAGNOSTIC_COMPLETE` and no error; the supervisor records child exit code0 and `live_owned_remaining=0`. The final monitoring failure therefore does not imply that the numerical routine threw an exception before completing. It still prevents calling the overall supervised attempt a clean PASS.

## Independent arithmetic checks

The small-file checker recomputed all derived quantities for the 359 parameter tensors and four groups, then independently summed the parameter sufficient statistics into each group. It verified finite values, counts, squared norms, dot products, cosine, norm ratio, combined-gradient norm, cross terms and first-order scalar directional derivatives. All match within the declared double-precision arithmetic tolerance. There are 819,503,620 trainable parameter elements in these summaries.

| Group | BC L2 | Policy L2, weight1 | Policy / BC | Cosine |
|---|---:|---:|---:|---:|
| All | 0.1620324978 | 0.0000993534 | 0.0006131694 | −0.0130164 |
| DiT | 0.0545407061 | 0.0000408924 | 0.0007497588 | +0.0525445 |
| Qwen projection | 0.0387364569 | 0.0000298905 | 0.0007716380 | −0.1449560 |
| Other action parameters | 0.1475782119 | 0.0000854720 | 0.0005791641 | −0.0125970 |

For the complete parameter vector, policy/BC is **0.0613169%**, not 0.613% or 6.13%. This supports the narrow observation that the separately measured raw gradient is dominated in norm by the BC term on this batch. The whole-vector cosine is only weakly negative; it is not evidence of a large, universal conflict.

The dot product is `−2.0954378098020477e−7`; policy squared norm is `9.871093183988339e−9`. Consequently, for the **unpreconditioned combined negative-gradient direction** with weight1, the local first-order policy-loss change per unit learning rate is `+1.9967268779621644e−7`. That scalar sign is consistent with the formulas. It is not a prediction of the actual Adam/ZeRO2 step, clipped update, finite-step loss change, PDMS change or long-run convergence.

All gradients were produced by separate BF16 microbatch backward passes, accumulated in FP32 on CPU, with chunked float64 dot/norm calculations. The result explicitly sets `exact_ZeRO2_update_claim=false`. The original joint backward's rounding, rank prescaling/reduction order, clipping and optimizer preconditioning were not reproduced as an exact update experiment.

The independently checked support analysis found that all 16 nonzero-advantage scenes in this batch are EP-only; NC/DAC/TTC/Comfort are one for all128 candidates. Thus this particular policy gradient is driven by progress distinctions. It does not quantify suppression of an observed safety-event gradient. No coefficient change or new training is approved from this single batch.

## Supervision and resource boundary

Preserved supervisor status: `STOPPED_NO_AUTO_RETRY`, reason `MONITOR_OR_CHILD_SETUP_FAILURE`, final error `Missing live RSS counter`. The source of the missing counter is consistent with the final process-exit observation window, but this review does not repair or reinterpret the receipt as successful monitoring.

Available resource evidence:

| Measurement | Actual |
|---|---:|
| Diagnostic wall / supervisor wall | 437.120870s / 442.451667s |
| Diagnostic peak process RSS | 26,418,532,352 bytes |
| Supervisor peak observed tree RSS | 27,565,686,784 bytes |
| Nominal admitted reads, including hashes and loads | 73,841,978,011 bytes |
| Supervisor observed physical reads | 30,819,545,088 bytes |
| CUDA allocated / reserved peak | 7,385,335,808 / 7,908,360,192 bytes |
| Output bytes before final receipt | 308,644 bytes |
| Remaining owned children, recorded | 0 |

Observed values are below the registered 1800s, 64GiB RSS, 192GiB read and 8MiB output limits. Nominal admitted reads, physical IO, and the diagnostic's process `rchar` are different counters and must not be substituted for one another. The available counters show no observed overrun. They do not establish a successful final whole-process RSS observation or hard continuous resource enforcement. The cleanup receipt reports zero owned children; this reviewer did not query or signal remote processes.

The original failure is retained. No automatic retry is approved or needed to read the already-complete mathematical result. Any future supervisor fix should use a small process-lifecycle fixture, not rerun this costly model diagnostic simply to replace the receipt.

## Review execution

```sh
cd /mnt/project/VLA-Drive-st-diagnostics-final-correctness-20261018
timeout --signal=TERM --kill-after=5s 55s python \
  reports/st_world_policy/20261018/gradient_balance_correctness_v1/check_completed_result_v3.py
```

The pure-Python checker read 377,447 bytes of small JSON/log/source files, took 0.037556 CPU seconds with 20,762,624-byte peak RSS, and performed 4,063 recorded checks plus the immutable-output check. Additional narrow inspection remained within the 2MiB total task read allowance. It imported neither PyTorch nor any model code and accessed no GPU, image, GT, metric cache, full manifest or checkpoint weights. It did not dispatch anything. Feature-side research remains paused.

Machine-readable independent review: `COMPLETED_V3_RESULT_REVIEW.json`.
