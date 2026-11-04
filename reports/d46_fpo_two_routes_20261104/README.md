# D46 FPO 两条研究线 · 2026-11-04

用户重新授权两条具体路线；其它RL终止状态保留。主开发与控制在 vlawm-tsinghua-zt3。

- **FPO_MATURE**：原FPO自身native1711/inner1接续，16卡Tsh5+WMzt，冻结科学源82d5f784；继承原Adam、scheduler、私有RNG、pending MC4材料。IL46原始初始化不变，历史完整最佳90.32032058。
- **FPO_VERIFIED**：同一完整IL46、fresh Adam独立初始化。B64/G16/M2、r16共享28层attention LoRA、SafetyEP+批级std、MC4 FPO不变。真实TRAIN B1对照，严格安全舒适且PDMS≥.9、改善≥.02的候选才作为教师；G32扩展只挖教师，不进入FPO ratio。CFM MC2和真实B1最后一步监督；保持初始同run IL46与自身高质量B1，bank全TRAIN容量，部署不含bank或评分器。

正式权重不继承资格权重。新代码复用稳定原生Trainer/Checkpoint，仅增加状态初始化和附加目标接口。17项针对性CPU检查通过；实际八卡首个更新、pending内层保存与冻结参数检查通过；中途恢复及梯度校准正在进行，未据资格宣称提分。

采用原warmup0.08E/hold0.60E/cosine4E/floor0.05，成熟配方LR2e-6。正式配置与校准收据封存在运行账本。先按1/2/3E完整单B1评测复核；原生6E预算不暗改，外层3E复核保存边界。改进线额外早评400/800更新，避免盲目长训。

只有三个执行容器现有可归属空闲容量；Tsh2有未知持续负载，不驱逐。其它临时获准资源优先用于可恢复分片评测；checkpoint/scene结果位于共享存储，失联后用rank锁和STOP fence只重派缺失分片，保留原失败/成本，不伪造RESULT。

报告：`/mnt/project/d46-fpo-two-routes-20261104/report.html`。主报告也将展示两条最新路线。原完整B1协议K10/shift5/seed2101、12146场景/44物理logs/invalid0、官方评分不变。NAVTEST只作预定评测/模型选择，标签不进入训练。正式启动以运行账本和全部rank实际更新证据为准。
