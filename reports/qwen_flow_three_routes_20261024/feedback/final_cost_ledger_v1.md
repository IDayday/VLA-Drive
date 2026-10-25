# 反馈路线一页成本账本（既有日志，只读）

> **验收口径更新（2026-10-25）**：本页所有DEV/VAL、TRAIN、候选搜索或拟合结果仅作诊断，不作为方法有效性、性能提升或方法排名依据。方法效力状态统一为**待完整NAVTEST确认**。原实验日期、数值和历史记录保留；详见[当前验收口径](navtest_acceptance_20261025.md)。

范围：512 TRAIN＋256 DEV_SELECT；所有时间都是原记录区间/worker占用，不是独占GPU计算时间。共享项只计一次。

| 阶段 | 逻辑/物理工作 | 记录时间（秒） |
|---|---|---|
| 共同bank（跨三路线） | K32池24576＋独立5ref3840＝实际28416评分；768 Qwen；76800批velocity forward | 7 worker累计7646.742；并发job span 1102.382 |
| F1_BLIND | 逻辑K32=24576，其中缓存8-prefix6144；真实新增18432评分/2304批；额外模型前向0 | TRAIN script 123.283＋DEV 62.858；scorer子CPU 308.172 |
| F2_PDMS | 逻辑K32=24576，其中缓存8-prefix6144；真实新增18432评分/2304批；额外模型前向0 | TRAIN script 124.464＋DEV 64.936；scorer子CPU 309.523 |
| F3_FEEDBACK | 逻辑K32=24576，其中缓存8-prefix6144；真实新增18432评分/2304批；额外模型前向0 | TRAIN script 127.443＋DEV 64.722；scorer子CPU 309.013 |
| F4_NOVELTY | 逻辑K32=24576，其中缓存8-prefix6144；真实新增18432评分/2304批；额外模型前向0 | TRAIN script 131.268＋DEV 65.121；scorer子CPU 312.783 |
| S1训练 | 256更新；GT4096/teacher0；FM前后向项4096；新在线候选0 | update合计3304.014；训练phase worker wall 3402.014 |
| S2训练 | 256更新；GT4096/teacher1896；FM前后向项5992；新在线候选0 | update合计4152.371；训练phase worker wall 4252.118 |
| S3训练 | 256更新；GT4096/teacher1896；FM前后向项5992；新在线候选0 | update合计3871.008；训练phase worker wall 3970.247 |
| S1主评测 | 3840逻辑＝3840真实单输出/评分；38400批velocity；768 Qwen；native namespace/B1 | TRAIN＋DEV累计worker 4061.780 |
| S2主评测 | 3840逻辑＝3840真实单输出/评分；38400批velocity；768 Qwen；native namespace/B1 | TRAIN＋DEV累计worker 4090.368 |
| S3主评测 | 3840逻辑＝3840真实单输出/评分；38400批velocity；768 Qwen；native namespace/B1 | TRAIN＋DEV累计worker 4087.962 |
| 共用S0主评测 | 3840真实输出/评分，跨路线复用 | env ledger 3927.275；外层worker 3942.287 |
| S1旧辅助评测 | 另3840真实输出；15360批velocity；namespace与chunk4不同，完全不作主配对 | generation 2016.194＋scorer wait 99.330；训练后aux/收尾约2513.679 |

F1–F4总真实新增73728评分/9216批；scorer子CPU累计1239.492秒。F0缓存审计143.484秒、novelty拟合18.396秒，没有新评分，但缓存历史不是免费。
CPU campaign的11个实际子进程累计worker 955.655秒、并发占用区间并集520.644秒；owner历时1136.789秒还含等待bank封存。最多两组各4workers、affinity8核、零GPU派发。
三学生主评测累计worker 12240.110秒，与实际并发job span 3726.556秒分开，不能相加为端到端wall。

S1 3304.014秒是update计时合计；3402.014秒是训练phase worker wall（另约98秒加载/hash/保存等）；5915.693秒是包含旧aux的完整invocation。三者不矛盾，旧记录不覆盖。S1后段差值是近似计账，不是独立阶段timer。

资格与校准单列：v1实际128评分/4更新（外部proposal资格未通过），v2实际224评分/4更新；其env ledger分别154.697/160.550秒，fit-window分别99.699/101.480秒，不能误称完整job wall。v2外层worker168.726秒；v1缺独立外层result，不伪造。β校准30个FM前后向项、16 Qwen、0更新/评分，env ledger76.755秒。所有资格权重未用于正式初始化。

未统一计时的只读分析/文档CPU耗时不可从现有日志准确追溯；本账本未伪造补齐。它是路线二及共享输入账本，不是三路线全项目费用。机器可读数值与来源hash见同目录`final_cost_ledger_v1.json`。

公开报告复核：全部20个本地Markdown链接有效；16个JSON可解析，9个带标准self-seal的新增摘要全部通过；没有原始scene/candidate记录字段、8×3/41×11数据数组或manifest原始scene-token值。允许的数组仅为聚合CI、学习曲线、shape/配置等。`terminal_results_v1.md`公开/私有字节相同。结论限定为TRAIN几何部分吸收，未夸大物理反馈增量或Navtest92。

## 2026-10-25账本范围补充

本页为形成时的既有搜索/三学生成本快照，原数值和原链接审阅计数保留。新增四场景表达probe的80次真实评价及训练费用另见[费用补充](execution_cost_addendum_v1.json)，不回写旧账本。该probe及本页TRAIN/DEV评价仍只作诊断，方法性能待完整NAVTEST。
