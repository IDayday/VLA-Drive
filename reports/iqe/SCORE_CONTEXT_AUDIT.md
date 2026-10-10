# Official score context audit

结果：**PASS，限定 NAVSIM v1.1 非响应交通单轨迹 PDMS 协议**。源码 `/mnt/project/local-interaction-mask-v2-artifacts/20260927/reference_sources/navsim-v1.1`。采样40×0.1s；候选8×0.5s；固定 official `metric_cache.trajectory` 为参考，不用 GT 代替参考，不做随候选集合变化的归一化池。

真实证据 `evidence/SCORE_CONTEXT_EVIDENCE.json`：场景 `0002d40d9c7753d8`、stage_val。reference backend 与原框架 `tools.local_interaction_mask_v2.score_async.score_chunk` 正式入口对齐；另外23次独立评分检验重排、删除、复制、追加和chunk大小（两条 probe，其中一条为GT，另一条横向偏移）。所有总分、命名分项、上下文/参考/协议hash一致。容差预设1e-12，实际比较完整 ScoreRecord 相同。

探针 GT 得分0.98528708031098（明确零一制），用于评分接口测试，**不是模型收益**。协议hash `570cb579994926588ef536162c81e573180e5661c9e8ab9efe0ed58a661c2966`。每候选独立子进程、scorer/simulator/observation实例，scene+seed+repetition配对，不依赖候选顺序。失败无效标签保留，不能当有效零分。

原 `metric_cache_navtrain_full` 的 LZMA cache 类为训练专用 MetricCache，含 pdm_progress，缺 `.trajectory`，因此拒绝用其直接执行正式reference评分。新增 `prepare-metric-contexts` 用真实raw logs/maps在独立目录生成完整官方cache。96场景生成完成，2CPU workers，197.36秒；原cache未更改。

分项为NC、DAC、progress、TTC、comfort、direction；无TLC。未实现/宣称reactive traffic或v2/EPDMS。真实CPU双轮已对固定44场景生成并评分全部候选：第一轮88条，第二轮132条；每轮只新增44次评分，旧标签分别复用44/88条。evaluate逐条将选中轨迹重新送入正式单轨迹backend，均与银行标签一致；audit-frozen另重新生成/评分旧输出。证据见ROUND_1/2_scoring_cost、frozen_audit和REAL_ROUND_SUMMARIES。该小规模执行不代替科学验收。
