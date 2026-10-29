# Qwen-World independent IL loss designs

Full12146 canonical v1.1 Test scenes, seed2101, single clean-x K10 shift5 ODE. All forks use the same63k native model/Adam/RNG/data boundary and2000 added updates. The shared CONTROL is original mean L1, not a new weak IL baseline. Test has been observed.

| Fixed endpoint | Updates | PDMS | Change vs matched CONTROL65k (points) | Previous RL comparison |
|---|---:|---:|---:|---|
| TIME_RESIDUAL_065000 | 65000 | 88.146292 | -0.235514 | — |
| DELTA_065000 | 65000 | 88.131260 | -0.250546 | — |
| HUBER_065000 | 65000 | 88.344878 | -0.036928 | — |

Only one loss factor changes per arm: bounded TRAIN residual-based horizon weights; GT-increment matching; or physical-tolerance Huber. Original clean-x codec, sigma distribution, four auxiliary losses, data ordering and frozen parent auxiliary bank are retained. Calibration matches only output-layer gradient magnitude, not full-model gradients. The shared control was separately qualified for exact loss/gradient and real eight-card execution.

The JSON paired fields contain physical-log bootstrap, native submetrics, G+/G− and failure transfers. A2000-update window is an initial controlled test, not proof that an entire loss family is ineffective. Qualification, failed attempts, model loading, checkpoint saving and evaluation costs remain in private owned-worker receipts.