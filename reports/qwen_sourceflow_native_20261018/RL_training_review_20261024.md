# Qwen-Flow RL训练检查与必要代码修复（2026-10-24）

两条正式RL正常推进，没有发现本轮采样、优势、old概率复用或optimizer恢复必须改动的错误。新增的代码修复属于离线评价身份与反馈归属校验，不改变actor、sampler、loss、LR、reward或正在运行的不可变源码。

## 检查范围与实际证据

- 启动核查时CANONICAL八rank一致25,682/50,000，SOURCE一致22,216/50,000；原IL为58,274/100,000，均继续运行。该状态是00:12 UTC快照。
- 两组0—20k实际scene顺序逐rank相同：5000份fresh全局rollout、40000次scene曝光、320000条真实candidate/组，四次inner不重复计环境样本。
- 逐项核验1/5k/10k/20k对应的4份真实global behavior batch，共32个scene groups、256条已评分轨迹/组；没有额外model forward、评分或更新。
- 从保存的mean/std/x_next独立用torch.distributions.Normal重算原Gaussian及源逐维relative clamp，再做H/D mean。raw logprob误差为0，clamp最大误差2.3842e-7，dimension-mean最大误差3.5763e-7；这是CPU/CUDA归约细差，不是fresh GPU ratio噪声。
- 保存的组优势与跨8rank未筛选原始reward共同std重算一致：advantage最大误差0，FP64 std误差不超过5.56e-17。每组全部8条合法轨迹保留，四个inner使用同一chain/old概率/优势摘要。
- 两份20k原生checkpoint全部文件hash、八rank事务、实际Adam/scheduler step=20000、480份FP32有限optimizer state均通过；不是只检查requires_grad。
- 运行源码所有登记文件hash与不可变434cba8快照一致。正式训练不使用本次改动后的分析代码，不需要重训、迁移Adam或重采behavior。

## 匹配20k训练窗口（不是部署分数）

| 项目 | CANONICAL | SOURCE |
|---|---:|---:|
| 有效PG group比例 | 99.7775% | 99.7150% |
| 全满分group | 2 | 3 |
| 全零分group | 76 | 92 |
| 其他等分group | 11 | 19 |
| 有NC/DAC/TTC差异group | 24288 | 26625 |
| 仅progress产生score差异group | 822 | 604 |

每组分母均为40000个scene groups。此前旧实验的高等分比例不是当前瓶颈；但非零优势多也不自动说明探索有益，不能通过制造失败来解释成功。

| 正优势组成 | CANONICAL | SOURCE |
|---|---:|---:|
| 仍有NC/DAC/TTC违规的比例 | 1.2122% | 3.8347% |
| 组内存在完全安全候选、正优势仍给违规候选的比例 | 1.0484% | 3.5829% |

SOURCE更偏进度的目标确实更常给违规候选正相对优势。这是需要在部署结果中检查的目标权衡，不是看到负面现象就直接认定概率实现出错；两条reward对照继续到登记终点。

两组所有已读fresh first-inner ratio最大误差均为0，clip fraction为0；2—4 inner出现真实更新与部分PPO clipping。匹配0—20k的第四inner平均clip fraction约1.42%/1.84%，不能解释成clip完全阻止更新，也不因出现clipping而放宽0.005。anchor与PG的loss标量不能直接证明梯度冲突或anchor过强，本轮不因此改beta。

## 已完成的完整DEV（1696场景/16 logs）

| 检查点 | CANONICAL | SOURCE |
|---|---:|---:|
| 同一53k IL | 91.011577 | 91.011577 |
| RL5k | 92.313500 | 92.839250 |
| RL10k | 91.938446 | 92.699140 |
| RL20k | 92.566839 | 92.691059 |

这些是原单候选shift5 ODE，非G8最好候选。20k SOURCE的DEV均值略高，但相对IL的NC下降；不能用DEV92分宣布达到Navtest目标。新Navtest按用户要求固定两条共同已完成的20k检查点，并非挑各方法最高点。50k固定终点、DEV选择规则和原配方均未改变。

## 发现并修复的分析边界

最小CPU复现表明旧离线reader会接受两类不一致：修改identity字段但未更新seal；或outer scene与内部官方feedback token错配、外层row仍有有效seal。真实轨迹/仿真状态全部内嵌于已seal的JSON，没有需要另找的NPZ payload，不能误报为独立轨迹文件未校验。

新增共享evaluation_integrity入口并接入完整汇总和paired reader：复验identity/protocol seal；校验outer token/log与官方feedback及protocol一致；校验完整scene人口和log归属、有限8×3原动作/请求、保存请求与实际送评分请求相同；保留合法0分和超出[-1,1]的有限动作，不做裁剪或填0。

16项针对性CPU回归通过。用新v3汇总重验完整IL53k Navtest与两条20k DEV，所有PDMS、分项、场景数量、物理推理数量和成本与旧v2完全一致，旧结果保留。当前没有发现真实identity编辑或feedback错配；修复不解释此前DDP RL效果，也不构成本轮方法已经提分的证明。新Navtest继续使用原434cba8数值入口，新分析代码只在其完成后校验/汇总。

## 成本、状态与结论边界

首次全journal检查的完整更新窗口为25,828/22,344，cost ledger对应已提交25,800/22,300；两种边界单独记录，没有用当前进程wall除以错误累计更新。此前未共享评价时估计剩余训练约13.49/17.85小时，当前共享Navtest会增加时间，不能当固定完成承诺。RUNNING invocation是未关闭观察，不等于失败或回滚；没有虚构无法追溯的rollback专用时间。

本轮判定：核心已查数学/训练合同通过；正常路径数值通过；恢复点完整性通过，未重复新增故障注入，沿用既有实际恢复资格；评价生成/评分不变，补强分析关联校验；成本保留已测范围与不可追溯部分；已有完成结果通过更严格检查并保持原值。完整Navtest仍在执行，完成后另给自身IL→RL的配对结果，不能仅以这些PASS宣布RL成功。

## 已执行命令

```bash
CUDA_VISIBLE_DEVICES='' /mnt/project/flow-safe-improvement-20261013/venv/bin/python -m scripts.analysis.qwen_sourceflow.audit_saved_training \
  --run /mnt/project/qwen-sourceflow-native-20261018/runs/RL_CANONICAL_seed42 \
  --source /mnt/project/qwen-sourceflow-native-20261018/snapshots/RL_parent53k_user20261023 \
  --updates 5000 10000 20000 --output /absolute/path/to/new_audit.json
```

SOURCE使用其原run目录同样执行。分析输出存在时拒绝覆盖；不向活跃训练发信号。证据：[CANONICAL真实buffer/恢复检查](RL20k_saved_audit_canonical.json)、[SOURCE真实buffer/恢复检查](RL20k_saved_audit_source.json)、[匹配训练窗口](RL20k_matched_training.json)、[最小旧reader复现](RL20k_prepatch_analysis_gap.json)、[16项回归](RL20k_integrity_tests.json)、[中期Navtest登记](RL20k_interim_registration.json)。
