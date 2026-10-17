# 评价任务先筛选可用资源，再分配主机

2026-10-17 02:29 UTC，里程碑评价控制器完成一次CPU调度修复和接管。两条研究路线、36项既定任务、训练和数值评价源码保持；24项完成、12项等待后续检查点。没有新增方法、seed、候选或推理，没有停止任何训练或真实GPU任务。

## 已复现的问题与修复

原逻辑先将ready任务与主机列表配对，然后逐个检查容量。当第二台显存不足、第三台空闲时，第二个任务在本tick不会尝试第三台。单个剩余任务也可能持续停在不可用的首台。文件锁与显存检查没有失效，但已授权空闲资源可能被遗漏。

新`capacity_dispatch_plan`先按既定顺序对未承接活动任务的主机做两次资源观察，过滤不可用者，再保持原任务顺序分配。已有export后的两次检查和实际派发前的授权检查继续执行。SSH错误仍上抛，GPU UUID变化仍拒绝；未知owner、低于24000 MiB的显存余量不被视为空闲；不会复制任务或抢占实际训练。没有ready任务时不增加GPU轮询。

02:23 UTC左右的真实只读对照如下；这是瞬时资源证据，不是持续空闲承诺。

| 主机 | 两次观察中的第二次最小空闲显存，MiB | 容量判定 | 说明 |
|---|---:|---|---|
| training-vlawm-tsinghua-zt | 30888 | 可共享 | 已有真实可见任务，保持 |
| training-vla-zt | 19703 | 不足 | 部分卡真实任务占用较高，保持 |
| training-rl-zt2 | 81221 | 可用 | 当时八卡空闲 |
| training-vlawm-tsinghua-zt4 | 62858 | 可共享 | 已有真实可见任务，保持 |

在旧主机顺序下，旧计划将DEV/TRAIN诊断分配给zt/本机；修复后同一顺序分配给zt/rl-zt2。根据本次实际余量登记的新优先顺序为rl-zt2、tsinghua-zt4、tsinghua-zt、本机；同一只读预演将两项分配给rl-zt2/zt4。预演使用B75k已登记的任务描述，不表示其检查点已经产生，也没有实际启动worker。后续每次真实派发仍重新检查资源。

## 已执行验证与接管

- 13项针对性CPU测试通过：既有排他/去重/本机路径/授权测试，以及首台繁忙、第二次观察拒绝、空队列、活动主机、全部不可用、SSH异常和物理GPU身份变化。
- 新源码`3a23d3cdaaa7fbb86ea60f9674d350f40b440e24`以`git archive`建立独立快照。配对分析器和其使用的FM合同、通用合同、原子写入工具与旧controller快照hash一致。范围内另一新增文件仅为既有报告的绘图脚本，不被控制器调用。
- 旧owner PID322120/start1363146860通过原`STOP_AFTER_TICK`在active=ready=0的边界退出；状态`STOPPED_CONTROLLER_ONLY`，旧锁释放。目录和日志保留，不向训练进程发信号。
- 新快照实际执行完整`--once`：同一36项job，24完成/12待办，0派发；97份既有分析JSON逐文件hash不变。
- 持久owner PID378345/start1364278483随后启动，实际工作目录、PID启动时间、owner记录和持有的排他锁均核验。目录为`/mnt/project/fm-transfer-20261016/milestone_controller_v5`。旧v4保持停止，不能另起第二owner。
- 旧只读观察进程因其绑定的v4退出而结束。这是预期的控制器接管，不是B训练故障；没有重启B。B的原torchrun PID286180/start1361098950持续运行，72k原生恢复点已发布。

| 组件 | 仍使用的源码 |
|---|---|
| SFT | `4e0ce1f45b6778895648689228a5628432bd2616` |
| RL、筛选、正式派发 | `225b9af660cb0210208e2c46fc4216d0f151275b` |
| 原始观测推理/评分 | `45d2318d2dec265583860a26f891d8b79bb6068a` |
| 新CPU里程碑控制器 | `3a23d3cdaaa7fbb86ea60f9674d350f40b440e24` |

修复不改变A终点或B已完成评价的有效性。没有实测新派发后的吞吐增益，不能将此写成训练提速结果。02:27 UTC附近，B从68k到72,369的4,369步完整窗口耗时1913.97秒，约2.28269 updates/s，包含期间保存开销；剩余SFT估计3.36小时，仅是当时的短期外推。B仍需自己的终点筛选、八卡RL验收及20k RL；最终Navtest准入没有提前。

机器证据见[接管摘要](capacity_evaluation_handoff_execution.json)。完整私有资源快照、旧owner信息、只读预演、`--once`日志及接管收据保存在`/mnt/project/fm-transfer-20261016/resources/milestone_capacity_migration_20261017/`。

## 实际执行入口

以下先附加`--once`执行资格检查，再按原命令启动持久owner。它是已执行记录；重复运行前先核验现有owner，不能与其并发启动。

```bash
cd /mnt/project/fm-transfer-20261016/source/3a23d3cdaaa7fbb86ea60f9674d350f40b440e24
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH="$PWD" /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.fm_transfer.milestone_controller \
  --evaluation-source /mnt/project/fm-transfer-20261016/source/45d2318d2dec265583860a26f891d8b79bb6068a \
  --dispatch-source /mnt/project/fm-transfer-20261016/source/225b9af660cb0210208e2c46fc4216d0f151275b \
  --host training-rl-zt2 \
  --additional-hosts training-vlawm-tsinghua-zt4 training-vlawm-tsinghua-zt training-vla-zt \
  --include-rl \
  --control-dir /mnt/project/fm-transfer-20261016/milestone_controller_v5
```
