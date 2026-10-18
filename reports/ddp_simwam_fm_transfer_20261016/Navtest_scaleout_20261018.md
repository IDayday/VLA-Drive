# 完整 Navtest：40 GPU 扩容与已完成结果复用

2026-10-18 用户释放资源并要求尽快完成全部 Navtest。00:35 UTC 实际核验：
40 个不同物理 GPU 上的 40 个 worker 均已通过原始观测入口一致性检查并开始
处理新场景。此文是执行快照，完整分数仍须等待全部 12146 场景验证；不发布
偏置的分片均分作为最终结果。[机器可读记录](Navtest_scaleout_20261018.json)。

| 固定模型 | 主机 | GPU 数 | 00:35 已封存模型×场景 |
|---|---|---:|---:|
| A SFT20k | tsinghua-zt | 8 | 5082 / 12146 |
| A RL20k | tsinghua-zt2 | 8 | 4550 / 12146 |
| B SFT100k | tsinghua-zt4 + vla-zt GPU0–3 | 12 | 6670 / 12146 |
| B RL20k | tsinghua-zt6 + vla-zt GPU4–7 | 12 | 3878 / 12146 |

每个场景分别执行 E_NATIVE、E_SHIFT5 单候选 seed2101，不能取两者最大值。
每条请求分别给 v1.1 和官方 v1.0 评分；新增版本没有新增模型推理。
历史 F-SFT/R_SOURCE/R_CANONICAL 的完整双版本结果继续复用。
未占用已撤销主机，未停止其他真实工作，Tsinghua 压力脚本不恢复。

最近同一 worker invocation 内的 `delta(new)/delta(elapsed)` 实测：A SFT
约 7.19 场景/秒，B SFT 5.99，B RL 6.55。此时 A RL 的完整计时窗口尚不足。
B RL 剩余推理约 21 分钟；加上 CPU 全人口校验与配对分析，00:35 时估计还需
约 30–40 分钟。这是实测速率外推，包含共享存储和负载的不确定性，不是保证。
`elapsed` 已排除模型加载，不再次减去 loading；加载成本另记。

## 扩容没有重复模型计算或改变科学合同

原八个进程在完整场景边界退出；原始预测、评分、权重和调用记录均保留。
共复用 12790 个模型×场景：A SFT4001、A RL3998、B SFT2398、B RL2393。
重新分片只改变执行 manifest 的 shard 数及相应身份引用。每条已保存的
动作、请求、实际仿真状态、分项和噪声均进行 dtype/shape/逐值摘要往返检查，
含 BF16 与浮点位模式。原目录没有覆盖；新记录保留旧 payload 和新 payload
的对应摘要。既有 v1.0 分数同样复用并绑定新的预测 seal。

固定权重和终点评价锁未改。实际模型 worker 继续使用
`45d2318d2dec265583860a26f891d8b79bb6068a`；原数值 dispatcher 为
`d58485b08078a6319b2835793d8c79ffbbd3e9d5`。未改变 batch1、精度、输入、
时间网格、后处理或评分归一化，也未启动新的训练、seed 或候选筛选。

扩容中两次派发问题有独立失败证据：一次 GPU 加载期间暂时无法归属占用，
一次远端容器没有向其他容器发起 SSH 的配置/凭据。前者重新现场核验后才派发；
后者由唯一协调端完成真实 PID/start/完整命令检查，向原 dispatcher 提供
绑定全部 receipt hash、120秒有效期的检查结果。原 queue 锁内再核对完整
receipt 集合，本机 PID 仍直接复查；新增 receipt 或过期证据均拒绝。
没有伪造空闲资源、放宽未知 owner 保护或复制凭据。

新 CPU owner 仅接管既有运行进程，没有重复启动。v1.0 评分在四台授权主机各用
8 个 CPU worker，追随封存预测；独立 canonical 汇总最多4个并行，双版本
汇总最多8个并行。分析继续调用原锁定代码，只有并发数改变。

## 验证、成本与实际入口

已执行9项无损重新分片测试、3项 PID 证据及队列改变测试、2项继承成本测试。
40 个真实模型 worker 的原始入口检查均通过。两个新增调度 CLI 和成本 CLI
均执行过 `--help`；帮助命令本身不作为真实推理证据。

```bash
python -m pytest -q tests/rl/test_navtest_repartition.py
python -m pytest -q tests/rl/test_dispatch_liveness_relay.py
python -m pytest -q tests/rl/test_repartition_cost_report.py
python -m scripts.analysis.fm_transfer.repartition_evaluation --help
python scripts/analysis/fm_transfer/dispatch_liveness_relay.py --help
python -m scripts.analysis.fm_transfer.repartition_cost_report --help
```

模型与评分已经运行，不应重新执行派发命令。当前实际控制记录位于：

```text
/mnt/project/fm-transfer-20261016/resources/navtest_scaleout_20261018/control_v4_parallel_analysis/
/mnt/project/fm-transfer-20261016/resources/pdms_dual_version_20261017/scoring_dispatch_v3_scaleout/
/mnt/project/fm-transfer-20261016/resources/pdms_dual_version_20261017/analysis_handoff_v5_parallel_scaleout/
```

扩容后的模型记录位于 `evaluations_scaleout/`，v1.0 为
`pdms_versions/v1.0/rescores_scaleout/`，双版本派生摘要为
`pdms_versions/analysis_scaleout/`，均在上述私有根目录下。

原 GPU invocation 原样继承，加载、旧推理和新推理耗时均计入。旧 v1.0
调用与新调用身份不同，成本补充入口显式合并二者；旧统计不覆盖。主动排空的
评分 attempt 留有 `FAILED_PRESERVED` 状态和关闭时间，它不表示合法样本被
判为零分，也不应被删去从而把原评分成本写成0。未完成的在途调用成本未知
部分明确保留。worker 时间之和、实际 job wall、共享 GPU 占用时间分开报告。

待完成的门槛是每模型12146场景/136logs、两种网格、两版分项及自身 SFT 的
配对安全转移。40卡已启动和代码测试通过均不代表 RL 已有效，更不代表92分。
