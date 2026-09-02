# Final Verdict

VERDICT=LORA_DENSIFIER_PASS

## Executive answers

1. **Base selected PDMS**：0.936907。
2. **Base Oracle@64**：0.993342。
3. **64 条候选均值**：0.797153。
4. 总剩余缺口 0.063093 中，coverage gap=0.006658（10.55%），ranking gap=0.056436（89.45%）；ranking/coverage=8.477。
5. 在 V_B<0.90 的 1905 个场景中，Oracle@64>=0.95 为 82.78%，Oracle@64<0.90 为 5.41%。
6. scorer 失败时通常不是“好候选不存在”：V_B<0.90 时 E[N_0.95]=9.717，主导类别为 RANKER_LIMITED（82.78%）；CANDIDATE_LIMITED 仅 5.41%。少量 SPARSE_GOOD 场景仍存在，见 `subset_summary.csv`。
7. scorer 成功/失败关联最强的特征为：N_0.99 (d=1.651), N_0.95 (d=1.359), selected_true_ego_progress (d=1.335), kendall_tau_b (d=0.985), spearman (d=0.954), N_0.90 (d=0.801)。这些是描述性关联，不是因果效应；grouped 5-fold CV 的 depth-3 tree AUC=0.9985，L2 logistic AUC=0.9985。
8. **Base64+IdealExtra1**（真实 PDMS 选候选的乐观上界）：oracle +0.002701，frozen-scorer selected +0.001786。
9. **IdealExtra8/16**：Extra8 oracle +0.002701、selected +0.004775；Extra16 oracle +0.002701、selected +0.004438。它们是否优于 Extra1 必须看 selected gain，不能把 oracle-equivalent 的真值筛选写成 LoRA 已实现增益。
10. **另一完整 64 条候选**：union oracle +0.002701，frozen-scorer selected +0.002267；128 条联合评分使用同一个场景上下文、完整 set self-attention，不是逐候选拼分。
11. **set-size control**：Duplicate8 selected shift=+0.000205，Duplicate16 为 0.000007；duplicate oracle 严格不变。
12. **固定 64 条预算**：Base56(predicted Top-K)+Ideal8 selected gain=+0.004589；几何去冗余策略为 0.004707。
13. **研究方向**：应输出 8/16 条近优候选进行增密，单条候选不足。 按预注册阈值，仅 primary Ideal8 配置通过 frozen-scorer 安全门；这不外推到任意候选分布、16 条或完整 64 条 bank。8 条固定预算 pilot 可先冻结 scorer，但真实 LoRA bank 必须重跑同一安全门。
14. **限制**：Ideal1/8/16 使用真实 PDMS 挑选，属于 upper bound；alternative checkpoint 只是本地 pseudo-expert，不是 LoRA；PDM reference 使用 evaluator/future 信息，仅是 diagnostic；本轮没有训练 LoRA，也没有验证联合重训后的 scorer。

## Base candidate audit details

| Metric | Mean |
|---|---:|
| N_0.50 | 54.951 |
| N_0.80 | 50.163 |
| N_0.90 | 40.235 |
| N_0.95 | 30.940 |
| N_0.99 | 23.491 |

Top-K oracle (K=1,2,3,4,6,8,16,32,64): 1:0.936907, 2:0.946057, 3:0.951173, 4:0.955174, 6:0.960296, 8:0.964259, 16:0.973017, 32:0.982248, 64:0.993342.

## Gate status

| Gate | Status | Evidence |
|---|---:|---|
| F0 forward/export | PASS | 12146 scenes, max abs error 0.000000, failures 0 |
| F0 true-PDMS parity | PASS | 128 pairs, efficient-vs-official max error 0.000000; selected-vs-official max error 0.000000 |
| F1 Base-64 | PASS | 12,146 scenes / 136 logs / 64 candidates, no dropped token |
| F2 scorer analysis | PASS | grouped 5-fold CV and 10,000 scene/log-cluster bootstraps |
| F3 candidate injection | PASS | complete-set frozen scorer plus fixed-budget and duplicate controls |
| F4 automatic verdict | PASS | `VERDICT.json`, pre-registered priority rules |

## Candidate-injection results

| Bank | Interpretation class | Scenes | DeltaOracle | DeltaSelected | Extra selected | Saturated false replacement |
|---|---|---:|---:|---:|---:|---:|
| alternative | deployable_pseudo_expert | 12146 | 0.002701 | 0.002267 | 0.4699 | 0.0452 |
| structured16 | deployable_structured | 12146 | 0.002532 | -0.005583 | 0.6201 | 0.1059 |
| jitter8 | set_size_jitter_control | 12146 | 0.001067 | -0.001679 | 0.5528 | 0.0524 |
| intermediate_full | extra_base_control | 12146 | 0.001471 | -0.096473 | 0.1992 | 0.0796 |
| intermediate_low | extra_base_control | 1905 | 0.009379 | -0.130645 | 0.3633 | nan |
| structured256_low | deployable_structured | 1905 | 0.018706 | 0.030106 | 0.8819 | nan |
| oracle_neighborhood_low | diagnostic_oracle_upper_bound | 1905 | 0.017563 | 0.043573 | 0.2877 | nan |
| reference_upper | diagnostic_reference_upper_bound | 12146 | 0.001733 | 0.000346 | 0.0375 | 0.0220 |

The primary verdict bank is `alternative_scaling134k`. Its target-available rate is 46.60% among Base candidate-limited scenes, where target availability means external oracle >= Base oracle +0.05.

## Bottom-tail selected-PDMS change (primary bank)

| Setting | All scenes | V_B<0.90 | Bottom 5% | Bottom 10% | Bottom 20% |
|---|---:|---:|---:|---:|---:|
| IdealExtra1 | +0.001786 | +0.010970 | +0.033456 | +0.016968 | +0.008738 |
| IdealExtra8 | +0.004775 | +0.029723 | +0.089640 | +0.045061 | +0.023763 |
| IdealExtra16 | +0.004438 | +0.033114 | +0.101003 | +0.050404 | +0.026196 |
| FullExtra64 | +0.002267 | +0.034218 | +0.110688 | +0.052453 | +0.024820 |
| Duplicate8 | +0.000205 | +0.004314 | +0.014528 | +0.006842 | +0.003367 |

These tail subsets are fixed by the original Base V_B ordering. Their changes are not obtained by re-selecting an evaluation subset after injection.

## Refinement-stage Extra-Base control

| Bank | Stage | Stage candidate mean | Stage selected | Stage oracle | Final64 union DeltaOracle | Frozen DeltaSelected |
|---|---:|---:|---:|---:|---:|---:|
| intermediate_full | 0 | 0.230671 | 0.246535 | 0.615854 | +0.001054 | -0.049985 |
| intermediate_full | 1 | 0.318753 | 0.328804 | 0.657602 | +0.001306 | -0.051678 |
| intermediate_full | 2 | 0.379496 | 0.363849 | 0.600076 | +0.001081 | -0.050184 |
| intermediate_full | 3 | 0.307393 | 0.337993 | 0.648883 | +0.001303 | -0.052948 |

## Interpretation boundaries

- **Deployable bank** means generation itself uses only current scene/Base predictions. It does not mean a learned LoRA was evaluated.
- **Frozen scorer** numbers are actual joint set scores from the unchanged Base scorer.
- **Ideal/oracle upper bound** uses true PDMS for candidate subset selection and is not deployable.
- **PDM-reference diagnostic** uses evaluator/future information and is never an inference algorithm.
- **Speculative** statements are restricted to the recommended next experiment; no causal LoRA gain is claimed.

## Reproducibility

Exact paths, hashes, versions, overrides, bootstrap seeds, bank parameters, and gate evidence are recorded in `manifest.json`; commands are in `commands.log`; runnable instructions are in `REPRODUCE.md`.
