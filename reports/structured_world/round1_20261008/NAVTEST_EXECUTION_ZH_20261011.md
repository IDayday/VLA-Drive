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

03:15 UTC 续跑重新选择三台空闲服务器的 24 张 GPU，保留同一 checkpoint、
input identity、scene seed 和全部有效记录。完整预测于 03:20 UTC 核验完成，
CPU 评分随后覆盖两套各 12146 场景，均 failed=0、valid=true。
两次尝试共保存 68 份工作进程记录。记录中的推理阶段耗时合计 5.34 GPU-hours；
从每个 GPU 进程 LOADING 到 COMPLETE/FAILED 计算的占用时间为 6.81 GPU-hours，
后者包含加载，但不包含独立资格短测、离线输入准备或中断恢复空档。
不能把任一数字称为整个研究准备和训练的总成本。

## 第一个完整规划结果

[G3 50k 完整结果与身份](G3_50000_NAVTEST_RESULT_20261011.json)及
[配对分析](G3_50000_NAVTEST_ANALYSIS_20261011.json)已保存。

| 对象 | 完整 Navtest PDMS |
| --- | ---: |
| G3 50k，内部 q0 | 86.6842 |
| G3 50k，最终轨迹 | 88.5195 |
| 历史 S0 50k | 88.2431 |
| 历史 S3 50k | 88.5037 |

最终修正相对同模型 q0 增加 1.8353 点，log 聚类 95% 区间为
[1.4836, 2.2143]；道路合规增加 1.4243 点、进度增加 1.8907 点。
修复 241 个 q0 零分场景，同时新引入 50 个零分场景。
这证明本模型内部修正具有实用作用；q0 不是独立训练的原 DDP 对照。

相对历史 S0 50k 为 +0.2764 点，区间 [-0.2829, 0.7927]；
相对历史 S3 50k 为 +0.0158 点，区间 [-0.4524, 0.4535]。
两项历史差异区间都包含零，目前接近历史强参照，没有稳定优于历史方案的证据。
相对 S3 修复 224 个零分场景，同时新引入 224 个零分场景，需要关注失败转移。
当前模型的内部 q0 还低于历史完整模型，因此内部提升不能全部解释为净方法收益。

这是 seed42 的 50k 中间观察，不是四组共同 100k 终点，也不是第二个训练 seed。
尚需 G1/G3 同步长 Navtest，以及 G0/G1/G2/G3 的共同终点来区分未来标签、事件目标、
局部采样各自的价值。额外几何准备、FGTR 与计算成本也必须保留在历史比较的解释中。

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
