# D46 AUG迁移修正

用户确认先完成准确迁移、观察真实效果，并授权终止旧RL、删除旧checkpoint。当前仅新开 `AUG6_NATIVE_CALIBRATED`；不恢复历史AUG/FPO。

主报告：`/mnt/project/qwen-world-performance-rl-20261103/report.html`。
本轮独立报告：`/mnt/project/d46-aug-migration-20261104/report.html`。
后台控制器：`tools/d46_aug_migration/owner.py`。真实阶段与完整成绩以报告/运行账本为准；本文件不把已派发当作已取得收益。

## 方法

复用官方 pinned JaT 采样、重评分和损失内核；复用本项目原生D actor、分布式trainer、官方PDM评分与checkpoint体系。

| 项目 | 本轮实际设置 |
|---|---|
| 父模型 | 同一完整D IL46，SHA `386a0a27bd197cb002501bcb90c343659001d2e6d4f7bad0cbd5e31a91f8477d`；新Adam0 |
| 场景/候选/更新 | B64/G16/M2；两节点16卡，每rank4个不同场景 |
| 实际采样链 | K10 shift5；7步确定性前缀，末3步官方lambda_t JaT探索 |
| 根噪声 | 和正式评测相同namespace/seed2101/scene token，CPU生成，同场景G共享根 |
| 探索尺度 | 全103288合法TRAIN的逐航点q01/q99范围映射到原IL坐标；采样、ratio、KL统一8×3协方差 |
| 输出精度 | 真FP32动作线性投影；actor/reference/部署一致，保留原Parameter与state_dict键 |
| 优势 | SafetyEP局部信用，所有有效64×16候选统一population std，cap5，padding不计 |
| 优化 | LR1e-6，AdamW(.9,.95)，clip0.2，JaT参考KL系数0.1，max_grad_norm1 |
| 调度 | 曝光warmup.08E/hold.60E/cosine4E/floor.05；先登记3E，再据效果延长 |
| 参数范围 | 共享文本attention LoRA及原生动作/时间接口，共7,010,307可训练参数；视觉冻结 |
| 保存 | 每100更新完整状态；最近10份，额外保留u0、最佳和正在评测的点 |

这是本地D适配，不声称完整复现CoT-GRPO、预测奖励或论文Stage II。部署路径保持/教师吸收留作后续独立改进，此轮不混入。动作坐标与官方评分代码不变。

FP32可能改变u0输出，不能把所有增益算作RL效果。预定完整NAVTEST：u0/200/400/800、0.5/1/2/3E；B1 K10/shift5/seed2101，12146场景、44物理logs、invalid0。既报相原IL46 89.49552098，也比较本轮FP32 u0。测试标签不流入训练；最终模型选择标记NAVTEST_SELECTED。

## 资格与正式任务分开

CPU算法14项与调度2项检查通过。真实两节点16卡资格已完成第1次Adam与保存，16rank fresh ratio误差0，原生B1入口对照通过；后续通过同一pending恢复再执行第2次更新，检查Adam、scheduler、RNG和冻结权重。

资格完成后，控制器用全新formal输出、同IL46、新Adam启动。不会从资格权重继续训练。科学源冻结于 `95b1d79`；调度代码独立冻结，不热改GPU源。唯一后台控制器事件触发评测与报告；日常30分钟更新、完整结果立即更新。

实际命令、进程、资格和完整结果写入 `/mnt/project/d46-aug-migration-20261104/operations/`，不将大体积临时验证文件推入Git。

## 旧任务清理

旧FPO成熟/Verified最终停在3323/643更新，相关27个训练/评测job均EXIT0。按用户授权，删除D44/D46共21个旧run下225个checkpoint；删除前这些文件去重分配字节合计207,784,361,984（不是独立测量的文件系统释放量）。代码、配置、日志、逐场景评测与分析保留，D IL和IL46父权重保留。

清理收据：`/mnt/project/d46-fpo-two-routes-20261104/operations/TERMINATE_OLD_RL_20261104/COMPLETE.json`。
旧FPO 1E评测被取消，保留其部分结果，但不发布不完整平均分。七个已完成点的最终归档见 [OLD_RL_FINAL_RESULTS.json](OLD_RL_FINAL_RESULTS.json)。

资源只使用本任务五容器白名单；本机D IL不抢占，无法归属的Tsh2 GPU负载不处理。主训练Tsh5+Tshzt；WMzt用于并行完整评测。
