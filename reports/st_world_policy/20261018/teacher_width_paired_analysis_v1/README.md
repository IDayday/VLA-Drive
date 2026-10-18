本记录绑定已经完成 5000 次更新的 PCA256 和原 SP2/PCA384，使用同一 1696 场景、16 logs 的完整固定开发集。它是新增维度对照的分析绑定，不是补写历史预注册，也不改变原空间筛选或授权继续训练。

两组均保持教师 384×288、原生 18×24 网格、每视图 W48/K96、同一训练 PCA 基底、原精度和原 20k 学习率时间轴。共同坐标在 mean256 与 mean384 下的损失系数不同；因此不能把维度效果解释成严格等权重的容量对照。

实际调用原 `evaluate_campaign.py`，继承完整清单与评分 receipt 检查、按 log 配对 2000 次 bootstrap、seed 20261017，以及零分、安全分项和停止代理诊断。没有重新实现统计、读取 navtest 或从部分有效场景构造均分。`BINDING_CHECK.json` 记录原函数对两组实际 registration/approval/identity/export 的检查；尚未运行评分结果分析。

在 PCA256 的 export 与 official_score 均完整后执行：

```bash
bash /mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261018/teacher_width_paired_analysis_v1/analyze.sh
```

该命令只读取两组已完成的预测与评分，不启动 VLM 或仿真。任一清单、候选、评分或上下文不完整时非零退出，不报告有效子集结果。输出为独立不可变分析目录，不覆盖已有空间实验。

需要一起回答：256 是否改善真实规划；零分和硬安全失败是否增加；差异是否集中在少量 logs。较低训练 FM、较小输出头、PCA 方差保留率都不能单独回答这些问题。16 个开发 logs 和一个训练 seed 的区间仍是探索性证据。
