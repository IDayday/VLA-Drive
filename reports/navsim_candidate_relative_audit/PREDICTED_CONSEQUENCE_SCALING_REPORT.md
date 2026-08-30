# Predicted-Consequence Data-Scale Control

All splits are by complete `log_name`; model strength is selected only from outer-train OOF candidate fidelity. Planning deltas compare against a direct planner with exactly the same online inputs.

| Run | Scenes | Predictor | Delta Spearman | Pairwise consequence Spearman | Variance recovery | Planning pairwise delta [95% CI] | Regret delta [95% CI] | Fidelity |
|---|---:|---|---:|---:|---:|---|---|---|
| formal_500 | 500 | mlp_delta | 0.12723538190737874 | 0.2970739017766492 | 0.02087384080803929 | -0.021838252939764824 [-0.052098810366812125, 0.0075049059707743785] | 0.0063845332091053315 [-0.0033585072339822864, 0.0217649265192449] | FAIL |
| scale_2000 | 2000 | mlp_delta_strong | 0.1880460424557925 | 0.5190308616711734 | 0.04215598494077408 | 0.0087582483503299 [-0.005785969276786476, 0.022862691786223504] | -0.0009058548949717228 [-0.009750799826946953, 0.006139304556983566] | FAIL |

The 2,000-scene run improves candidate-specific fidelity and yields favorable pairwise/regret point estimates, but planning intervals cross zero and candidate variance remains far below the declared recovery gate. It is evidence of a positive scaling trend, not a robust utility claim and not a method failure.

Figure: `figures/predicted_consequence/data_scale_control.png`.
