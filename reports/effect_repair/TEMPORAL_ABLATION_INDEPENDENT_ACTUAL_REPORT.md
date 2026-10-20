# Actual dense temporal-grid ablation: independent read-only comparison

The matched data and reporting checks pass. The current single-seed experiment
**does not establish a recommendation benefit from dense temporal-grid loss**.
Its point estimate is worse than the otherwise identical zero-weight control;
the paired-log confidence interval includes zero. No training recipe was changed
after seeing these results.

Both original evaluations are COMPLETE at optimizer step1000 / seed42. Their
checkpoint sampler proofs match exactly: 32000 consumed scene presentations and
identical scene/pair RNG states. The comparator checks Base, native pair bank,
context, labels, capacity, initialization, optimizer/scheduler, actual consumed
streams and numerical runtime. Only the dense `[41,3]` event-grid objective
weight differs, `.2` versus `0`. Zero does not remove the terminal first-event
labels, event transitions or all temporal information.

The source requested 233 validation scenes. This analysis admits all **225
eligible scenes / 114 logs / 4615 native intervention pairs**, with exact pair
membership; eight source-defined no-pair scenes remain separately inventoried.
Their absolute Base/selected PDMS stays null. The original233-scene reports and
their coverage flags remain unchanged. This is held-out TRAIN bank analysis,
not a full DEV/navtest policy evaluation.

| Objective | Selected native increment, PDMS points | Regret, points | High→zero scenes | New NC / DAC / TTC scenes |
|---|---:|---:|---:|---:|
| Dense temporal `.2` | -.7787 | 3.3076 | 4 | 1 / 3 / 6 |
| Dense temporal `0` | -.0187 | 2.5476 | 2 | 1 / 1 / 7 |

All displayed increments and regrets are the internal `[0,1]` values multiplied
by100. The common within-bank oracle opportunity is +2.5289 points. Full minus
zero is **-.7600 points**, paired-log 95% interval **[-2.1093, +.3280]**, 2000
bootstrap repetitions / seed42. This interval covers held-out log sampling for
one trained seed; it is not training-seed uncertainty or proof of safety.

Full's scene-weighted direction accuracy is `.9005`, versus `.8928` without the
dense objective, while its actual recommendation utility is worse. Full's
trained dense-head MAE is `.00791`; zero's untrained-head diagnostic is `.02252`.
Neither sign accuracy nor lower auxiliary error demonstrates successful physical
credit assignment or safe trajectory improvement. The zero control still has
negative mean utility and severe mistakes; it is not a validated deployment rule.

The one actual derivation took30.95 seconds on CPU with two Torch threads. Guards
made model forward, native builder and simulator scoring raise immediately if
called. They were not called. No GPU, optimizer update, new simulation or policy
inference occurred. All **10 original artifact hashes** stayed unchanged.

The complete derived JSON is4,163,548 bytes, SHA256
`345cfc9a68474367df6641da1747ce86fbea5ea70aaf0bbb8acd595470dc9c08`.
`ELIGIBLE_REPORT_INDEPENDENT_ACTUAL_TEMPORAL.json` pins its actual path, original
receipts, exact sampler proofs, complete structural inventory, full command and
execution script/log hashes. Full label/event/risk diagnostics remain in that
immutable report; no original evaluation was overwritten. This read-only run
uses the reporting source already covered by the54-test independent affected
suite; it does not claim an additional test run or any Route A/B policy result.
