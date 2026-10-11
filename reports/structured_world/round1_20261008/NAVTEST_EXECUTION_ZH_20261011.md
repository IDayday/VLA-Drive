# Navtest 动态资源执行记录（2026-10-11）

本轮按用户要求直接评估完整 Navtest，不再派发新的开发集评分。
训练仍采用原登记的初始化、数据、损失、100k scheduler 与 native 恢复合同。
推理调度和评分流式化属于执行改动，训练/模型/评价源码仍锁定 `0c54f1c`。

可复用的 [skill](../../../skills/navtest-parallel-eval/SKILL.md) 安装于
`/mnt/project/skill/navtest-parallel-eval/SKILL.md`，并通过项目 `AGENTS.md` 发现。
GPU 数、主机数、分片数均在每次派发前依据当前资源决定；历史部署的卡数不能作为默认配额。
先去重物理 GPU UUID，再核验进程身份、显存、持续负载、实际运行环境和输入资产。
场景通过带 generation 的小工作单元动态领取，快的 GPU 可以处理更多场景。
改变工作进程数后，完成条件仍是全局唯一场景集合和文件哈希，不是新一轮进程计数之和。

## 实际输入与数值验证

[当前输入核验](NAVSIM_NAVTEST_INPUTS_VERIFIED_20261011.json)覆盖完整 12146 场景、
136 logs、36438 张当前相机图像；地图、未来标签和 LiDAR 不进入部署输入。
输入身份为 `b14585d89aecd8f67f088c6cc7eb467dcc12c9a654d9fac0a952626ff5759781`。
[heading 核验](NAVSIM_NAVTEST_HEADING_VERIFIED_20261011.json)直接使用历史准备器的
`Quaternion(...).yaw_pitch_roll[0]`，避免额外 float32 矩阵往返引入审计误差；
未修改冻结的输入构建器或训练 ego 状态。

[独立运行环境](NAVSIM_NAVTEST_RUNTIME_STAGING_20261011.json)通过 Python/Torch ABI、
包版本、全部文件/链接、完整模型 CPU 导入和冻结源码核验；未修改共享训练环境。
[实际 FP32 短样例对照](NAVSIM_NAVTEST_FP32_PARITY_20261011.json)覆盖六个部署环境，
q0、q_final、out_of_range 与未修改的原推理入口逐位一致。
这些短样例用于迁移资格验证，不作为实验成绩。

正式推理保留 native FP32 master 恢复、strict state loading、FP32 前向、TF32 关闭、
原 10 步 FM、单候选和 scene 绑定 seed42。每个 checkpoint 先生成只读 FP32 导出，
绑定原 native checkpoint 的完整哈希，再供独立工作进程加载。

## 实测调度与干扰

G3 50k 首次派发经资源刷新实际启动 44 个独立 GPU 工作进程，分布于六个容器。
之前扫描的 40 张卡只是当时快照；skill 没有固定其中任一数量。
实际 PyTorch peak reserved memory 为 13.08 GiB，并额外保留物理显存余量。
空闲 GPU 的单场景耗时约 1.3 秒，共用训练 GPU 的约 2.3 秒。

CPU PDMS 池在 GPU 推理前启动，逐场景消费原子发布的预测，分别评分 q0/q_final。
本次实际使用一个主机的 16 个 CPU 进程，原生数值库每进程一线程；其 CPU 预算依据
affinity、cgroup quota、当前负载和预留容量登记。CPU 评分保持跟随推理，未等待完整预测库。
当前实现没有宣称部署了多主机 CPU 评分；如需扩展，必须先实现互斥的场景/variant 归属。

持续共用 GPU 后，G1/G3 近期训练单步从约 3.2 秒升到约 9.2 秒。
推理工作进程退出后恢复至约 3.15 秒。因此仅有剩余显存不足以认定资源合适，
后续派发需要把对真实训练的干扰纳入准入，优先使用空闲 GPU。
这些是执行成本观察，不是改变训练配方或损失的理由。

## 中断与恢复边界

首次尝试在 03:04 UTC 因 `vlawm-zt5` SSH 超时及工作进程 stdout BrokenPipe 中断。
原失败记录和尝试成本保留。独立核验确认 8699 个预测身份与哈希有效，剩余 3447 场景。
已评分记录保留；它们不构成完整 Navtest 成绩。
只在证明旧工作进程退出、确认传输故障和 fencing 旧租约后续跑。
科学错误不能通过清除失败标记来续跑；每次恢复必须保存绑定原失败哈希的证据。

最终成功需要两套轨迹各有 12146 个对应场景、所有状态为 ok、failed=0、valid=true、
实际 CSV 场景集合与均值一致，并匹配冻结 canonical identity。
不能仅依据一个 COMPLETE 文件或进程的自报计数判定完成。

## 压力脚本与共享锁

[真实共享锁验证](NAVTEST_SHARED_LEASES_VALIDATION_20261011.json)在本地及两台远端执行，
验证 Lustre flock 互斥、原子发布、generation fencing 和六进程无丢失写入。
[压力租约测试](NAVTEST_PRESSURE_LEASE_VALIDATION_20261011.json)的 27 项 CPU 测试
仅操作自有临时进程和模拟 GPU inventory；真实 GPU 压力脚本恢复尚未实测。

`navtest_pressure_lease.py` 只识别确认的 `/mnt/project/gpu_stress.py` 入口，
释放前保存原 argv、环境、用户、cwd、限制、PID/start-time、子进程及 UUID 绑定。
私有环境保存在仓库外，不输出或提交。实际推理工作进程退出后恢复原脚本；
如果真实任务新占用了 GPU，保存延期恢复义务。没有停止原脚本时不会新建占卡任务。
本次检查未发现需要释放的压力脚本，没有停止真实训练。

## 可执行入口

在本任务 worktree 中运行，下列入口均有实际实现：

```bash
python tools/structured_world/run_navsim_navtest.py --inventory
python tools/structured_world/run_navsim_navtest.py --enqueue \
  --run /mnt/project/structured-world-fgtr-round1-artifacts/20261008/formal_navsim_local_G3_G2_v3_io/G3_EVENT_LOCAL \
  --tag milestone_0050000 \
  --output /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225
python tools/structured_world/run_navsim_navtest.py --serve \
  --output /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225 \
  --routes /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225/ROUTES.json \
  --inputs /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_current_v1_0c54 \
  --canonical-python /root/miniconda3/envs/navsim/bin/python
```

若 `controller.lock` 已持有，不启动第二个控制器。GPU 数由当前准入的 route 决定。
`--recover-infrastructure` 要求明确的 request key 和 recovery evidence，不能盲目重置输出。
完整结果之后使用 `analyze_navsim_navtest.py` 做场景配对、log 聚类 bootstrap 和失败转移分析。
同模型 q0 是内部诊断；历史 S0/S3 的架构与计算成本不同，不能把差值全部归因于新监督。
