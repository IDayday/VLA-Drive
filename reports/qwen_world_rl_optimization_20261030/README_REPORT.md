# Qwen-World RL 研究报告归档

固定持续报告：/mnt/project/qwen-world-rl-optimization-20261030/report.html

本归档保留七节中文研究报告、八组标准科学图（SVG/PNG/PDF）、展示/采集源码和小型结论快照。报告版本时间见 editorial_snapshot_summary.json。当前分数、停止状态和正在进行的诊断均从完整结果或独立收据汇总；TRAIN 与完整 NAVTEST 分开，partial 不发布均分。

- reporting/render_report.py：证据采集、完整性检查、原子输出和唯一 watch owner。
- reporting/research_report.py：研究结论与图表展示层，每次生成重新加载。
- reporting/report_events.py：原有 owner STATE 的 inotify 通知和去重。
- reporting/figures/：八组图件；PPO算法、数据路线、NFT与训练panel使用分开的图。
- report.html：本次归档页面；部分原始证据仍链接运行目录，权重、cache、scene raw 与 NPZ 不归档。

现有 watcher 保持 1800 秒常规更新及阶段事件更新。生成一次可运行：

```bash
/root/miniconda3/envs/navsim/bin/python reporting/render_report.py --root /mnt/project/qwen-world-rl-optimization-20261030 --once
```

这条命令仅汇总现有证据，不会训练、推理、评分或连接服务器；不要重复启动 watch。科学统计及运营收据由研究主进程另行归档，见 analysis/ 与 research_decisions/。

本次展示更新以场景条件安全—驾驶效率为研究主线，加入20k匹配配对、TRAIN4096最终三模型诊断及原生optimizer表征可训练审计。10k BatchStd历史最佳保留，后续20k优势未持续；SafeComfort仅是通过候选内局部突出EP。资源按最新有效时间窗和独立部署收据动态区分，报告侧不操作主机。

2026-10-31更新：报告优先展示SafetyEP/MemoryLR20完整10k匹配及NC风险，动态读最终TRAIN偏好manifest和纵向zero诊断。新授权收据允许hosts列表，权限激活与库存证据分开；最新REQUEST覆盖旧8h时间窗。报告watch保持原PID和固定30分钟deadline，不操作科学运行。
