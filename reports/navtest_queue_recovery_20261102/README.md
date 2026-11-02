# 2026-11-02 Navtest 评测队列恢复

本次处理评测阻塞和新增资源，不修改训练算法、部署动作、参数精度、模型 batch、seed、评分版本或场景人口。

## 发现的问题

- IL B36k 的旧 parity worker 在已撤回的 wm-zt2 上正常返回，但实际为 `STOPPED_AT_SCENE_BOUNDARY`，新推理为 0，未产生 parity PASS。协调器仍在，因而长期表现为 RUNNING，实际不能进入评测。
- IL C40k、D34k、B44k 已完成导出；固定且不受 B/D 保护的两台机器只有约 21–29 GiB 卡上余量，小于既有 32 GiB / parity 48 GiB 准入阈值。
- 原 IL worker 不识别 Safety RL cohost，单纯看到显存空余也不能安全共享。
- G 有三个评测分片在 11月1日22:15 的临时资源撤权时，因明确的主机权限错误退出。H 有两个本地权限检查在 SSH/Popen 前拒绝的尝试，状态留在 dispatch uncertain。
- 另有 F/G 各一个旧 worker 在 CUDA 子进程创建前被共享准入拒绝，但未写 RESULT，造成虚假的 RUNNING。实际没有对应 rank 输出锁和推理子进程；截止后的主机退出审计证明旧节点已清空。
- H STATE 中保留的 SPEC 错误是历史记录。重新计算当前可用 gate 的完整 spec，未发现与已保存 spec 不一致；本次不改身份、评分版本或原结果以消除旧错误。

## 已执行的修复

1. 七个确切的过期/未启动尝试逐个审查，保留原失败凭证、尝试编号、已完成场景和全部身份。只重新排队原分片，不伪造 EXITED/COMPLETE，不将失败写成 reward=0。
2. B36k 的旧 parity 使用绑定原 `spec.json` hash 的 `OPERATOR_REQUEUE.json` 明确排除；尝试号仍递增，不重新采用更早的失败尝试。
3. IL burst 增加独立 `--worker-source`。采用已经执行过真实恢复的资源 worker `/mnt/project/VLA-Drive-abcd-il-c-recovery-20261101`（`17c23cd18ce8aa6f05c2849ce0d01c90f00c8b51`）；actor/export/scorer 仍从原来的 `--source`、相同 `--cwd` 和原科学 PYTHONPATH 加载。probe 和实际准入使用同一 worker 的可见 owner、UUID、锁和显存检查。
4. 四个现有 IL burst CPU owner 与唯一每2k cadence CPU owner已接续。已有 GPU worker未发信号。重排 G/H 时只接续本任务 CPU 协调器，已有健康训练、评分和 Review 进程继续。
5. 用户新增授权仅为 `vlawm-zt4`、`vlawm-zt5`，临时不限时。Compare/Safety 的新不可变授权记录仅包含固定四台加这两台；其他已到期或撤回主机不重新开放。F/G/H 训练 placement 先去掉新增两台，让它们优先清理评测积压，训练科学配置及预算不变。

wm-zt4 实际连接成功，部署前八卡约 81,153 MiB 空闲，无 GPU 进程。wm-zt5 SSH 当时连接拒绝；用户表示正在恢复连接，未在该主机宣称已部署任务。

本次没有执行此前考虑的固定节点训练暂停/评测时间窗，因为用户新增资源后无需暂停。B/D 及其他 IL/RL GPU 训练没有收到停止信号。

## 实际执行证据

2026-11-02 01:13–01:18 UTC 的有限检查已经看到真实场景前向与独立评分记录：

| IL 检查点 | native/export parity | 实际状态 |
|---|---|---|
| B36k | 等待重新校验 | 旧零场景中断已审查，尝试已重排 |
| C40k | PASS | 新场景记录持续产生 |
| D34k | PASS | 多个分片持续产生新场景 |
| B44k | PASS | 多个分片持续产生新场景 |

一个过程快照中 C/D/B44 的当次 invocation 已处理/复用分别为 361 / 781 / 1,871 条。这不是完整评测分数，也不是不考虑旧缓存的全局完成人口比例。完整分数仍要求 12,146 场景、invalid=0、所有分片合法结束。

后续有限检查确认 B36k 的第二次 parity 已 PASS，已进入真实场景评测；四组均在推进。对应当次 invocation 处理/复用计数为 B36=43、C40=2,467、D34=1,817、B44=2,434，收据 `IL_CURRENT_PROGRESS_VERIFIED.json`。这些是带时间戳的运行快照，不替代最终 SUMMARY。

新 worker 仍为单候选 clean-x K10、shift5、seed2101、canonical NAVSIM v1.1、原请求轨迹与原官方 reference 配对评分。候选之间的 EP 归一化语义未改动。

原始证据目录（实际存在，未上传私有数据或权重）：

```text
/mnt/project/qwen-world-il-research-20261030/maintenance/evaluation_resource_recovery_20261102/
    REQUEST.json
    grants/
    IL_ADOPTION/COMPLETE.json
    revival_training_20261101/COMPLETE.json
    improvement_training_20261101/COMPLETE.json
    PRECHILD_GHOSTS/
    ACTUAL_EVALUATION_PROGRESS.json
```

## 已执行命令

```bash
/usr/bin/python3 -m unittest discover -s tests/il_report -p test_burst_recovery.py -v
/usr/bin/python3 -m unittest discover -s tests/il_report -p test_navtest_cadence.py -v
/usr/bin/python3 -u -m tools.il_report.recover_evaluation_shards_20261102
/usr/bin/python3 -u -m tools.il_report.recover_evaluation_shards_20261102 --prechild-ghosts
/usr/bin/python3 -u -m tools.il_report.adopt_periodic_queues_20261102
```

前两条分别为 8/8 和 7/7 PASS。后三条是已经执行的一次性明确审查，遇到已有完成凭证会拒绝重复启动。不要复制命令重新启动第二套 owner。

阶段报告仍复用原 IL/RL 报告 owner；每2k评测规则和半小时/重要事件报告机制不变。时间分布加权85,711步终止、其他暂停/终止决定及 B/D 保护均保留。

## wm-zt5 连接恢复后的实际接入

用户通知 SSH 已恢复后，实际发现八卡空闲、共享代码/数据挂载正常，但新容器没有节点本地的 `/root/miniconda3/envs/ddp`，导致共享 venv 的 Python 符号链接失效。没有修改共享 venv，也没有升级依赖。

从已验证的 wm-zt4 将同一节点本地运行时复制到 wm-zt5 的独立 staging，再以相同绝对前缀原子发布；约 9.233 GB、38.37 秒，两端 tar 返回0。随后两端验证结果一致：Python3.10.20、Torch2.5.1+cu124、Transformers4.57.0，以及 numpy/scipy/shapely/OpenCV；Python、Torch `_C`、FlashAttention2 二进制 hash 一致。BF16 GPU 小张量运算和现有 IL/RL evaluator 导入均通过。资格脚本首次发生的操作员语法错误在导入前退出，两端原始失败凭证保留，修正后通过；不是模型或评分故障。

现有调度器已自动接入 wm-zt5，未新建重复 owner。实际 NVML 快照确认八个物理 GPU 均有评测进程；该快照为22个进程（14个IL、8个RL），不能把已派发但仍在加载的24个 worker都写成已占用GPU。C40k、B44k和G16 34k分片在该节点已分别产生113、116、110条新场景记录。它们仍是过程计数，不是最终Navtest成绩。

实际证据：

```text
/mnt/project/qwen-world-il-research-20261030/maintenance/wm5_runtime_recovery_20261102/
    OWNER.json
    QUALIFICATION.json
    GPU_ACTUAL_DISPATCH.json
    ACTUAL_PROGRESS_VERIFIED.json
```

无其他主机重新授权，无现有训练暂停、Adam重置或 B/D 干扰。运行时二进制、权重及数据不提交仓库。
