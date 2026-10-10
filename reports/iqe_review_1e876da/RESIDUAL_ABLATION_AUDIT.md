# E6 residual_v2

The legacy implementation `base+(private-base)` is **DEGENERATE_RESIDUAL_ABLATION**. It remains available for historical artifact loading, but new strict configs reject the legacy variant; historical results are not removed or admitted as valid residual-learning ablations.

`iqe/ablations/residual_v2.py` implements `output_raw=frozen_base_raw+delta_raw`. The private branch directly predicts an unrestricted delta; every relevant final regression output layer is zero-initialized. No subtraction of Base, clipping or delta penalty occurs. Default full-trajectory IQE is unchanged. Normalized XY and unconstrained sin/cos residual channels use the existing physical decode/atan2; physical and normalized-domain definitions are not mixed. Output IL is strictly equivalent to fitting target−Base for the frozen additive term.

Actual32-fit-pool diagnostic used4 audited observations and2 updates: initial delta/raw error0; post-update delta max **0.2717189491**; raw IL **2.95381546→2.64714932**. Base hash unchanged and frozen raw maximum difference0. Private trainable parameter count **10,964,384**. Residual evaluation reads the frozen Base trajectory and performs **two Decoder calls** per candidate. Those costs must be charged in E6 rather than hidden.

Automated tests verify zero init, nonzero updates, correct raw/physical transformation, no Base gradient/update and that changing the frozen Base term changes final output (the legacy algebraic cancellation would fail). [Real evidence](evidence/RESIDUAL_V2_PROBE.json). E6 engineering READY; matched-budget scientific comparison remains UNTESTED.
