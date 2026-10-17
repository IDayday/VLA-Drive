# 固定终点之后的最终评价衔接

2026-10-17 04:21 UTC核验：A的20k RL已完成；B的原八卡SFT进程到87,775/100,000，八rank一致、最新已发布恢复点86k。既有阶段队列26/36完成，10项等待B后续检查点。**最终Navtest尚未开始，B尚未开始正式RL。** 本页增加的是最终评价的CPU衔接，不修改模型、训练、采样、评分或既定预算。

新增入口 `scripts.analysis.fm_transfer.final_handoff` 已在不可变源码 `ad73d13ec67d880a8be123f9526b575958cf8aff` 启动。唯一CPU owner PID398934/start_ticks1364944372，状态 `WAITING_FOR_BOTH_ENDPOINTS_AND_DEV`。它没有生成最终LOCK、资源派发阶段或GPU worker。当前状态是时间快照，后续应读实际进程及状态文件。

## 复用与准入

实际命令继续调用已验收的 `d58485b08078a6319b2835793d8c79ffbbd3e9d5` 中的 `final_lock`、`dispatch_final_evaluation` 与 `analyze_evaluation`；这些文件相对本次HEAD的git diff为空。实际数值推理/评分保持 `45d2318d2dec265583860a26f891d8b79bb6068a`。SFT仍用4e0ce1f，正式RL仍用225b9af，没有热修改运行快照。

控制器先等待原 `final_lock.readiness`：两组完整SFT、两组20k原生RL与各自关闭的训练成本账本、全部36项阶段评价和配对结果。证据齐全后才调用原锁定入口；文件存在只是初筛，后者继续验证完整checkpoint/hash、原生optimizer预算、自己的SFT父身份、固定终点及已登记评价身份。不能因A的10k较高或B的50k较高而选择它们。

最终只评价四个固定模型：A自身SFT、A20k RL、B100k SFT、B20k RL。每个模型都是12146场景/136 logs，seed2101，分别记录E_NATIVE和E_SHIFT5；总计97168条逻辑单候选读出，不是best-of-two，也没有新增seed。

准入后重新读取当前服务器授权。对仍授权且可访问的主机做两次GPU UUID、显存、可见owner与CPU affinity/quota/load观察，排除未知占用、显存不足和重复UUID。共享仅在既有授权和容量范围内；保留原dispatcher的24000 MiB显存余量门槛及每个worker一个评分进程。适用GPU数量决定分片，可使用部分卡，不要求完整空闲八卡主机。当前实现按卡数分配原有stride分片，尚未证明不同负载主机上的最优均衡或加速倍数，不把它描述为新的动态工作窃取系统。

派发只调用原dispatcher，由它在每次启动前再次验证模型、队列锁、GPU及保留记录。CPU加载期间也保留UUID。没有引入新的模型worker或数值路径，也不会停止真实训练或未知进程；本次没有释放/恢复任何占卡进程。

## 完成、失败与重复启动

每次外部命令保存intent、实际命令、PID/start、stdout和完成状态。只有相同命令成功退出的阶段可复用；丢失响应、部分派发、失败或不一致记录均停止，不能据此自动重跑。

一个模型完成需要全部分片报告身份匹配，以及实际PID/start对应的worker退出；SSH超时不是退出证明。随后调用原分析器验证全部场景、轨迹及score/hash，再核对完整人口、固定权重、seed、网格和最终LOCK。两组各自的RL−SFT配对必须与独立均分之差一致，固定SFT-safe人口也要一致；四份完整结果和两份合法配对齐全后才发布控制器COMPLETE。

全局 `final_handoff.owner.lock` 防止不同control目录各启动一个owner。实际重复启动已被flock拒绝，原owner和状态未变。训练、B终点衔接和阶段评价各自原owner继续，不被替换。

失败时保留所有输出和实际worker，不自动kill或retry。需要先检查具体失败阶段及实际进程，使用原dispatcher的显式恢复机制和新的接管记录；本入口不提供未经审查的自动失败接管。不能直接重复启动一条失败命令，或将旧分片的结果覆盖为新身份。

## 已执行验收及边界

本模块加上已有最终准入/派发测试共**68项CPU测试通过**，覆盖：

- 缺少终点证据时不探测资源、不建立最终LOCK、不派发。
- 部分可用主机、不可达主机、未知GPU owner、UUID别名、CPU余量与正在使用的保留设备。
- 1/3/8/17/32/64卡分配后每个模型仍精确覆盖12146个逻辑场景。
- 不完整或重复派发、worker仍运行、完成报告身份不符、SSH观测失败。
- 四份完整人口和两份自身SFT配对；拒绝缺场景、错reference、错配对均值或错安全人口。
- 成功命令幂等复用，以及失败/响应不明命令拒绝自动重放。

真实CPU `--once` 在当前未完成终点状态下通过，耗时1.984秒；实际常驻owner和重复owner拒绝也通过。**尚未执行最终Navtest派发和完整最终评价**，不能将mock完成路径当作四个模型已经跑完。本次新增模型推理、评分、optimizer更新均为0。

首次CPU资格运行发现历史d584源码封存使用`git_sha`字段，初版只接受`commit`，因此在建立控制器或派发前拒绝。已明确支持两种已有schema：所有出现的commit身份均须等于目录SHA，且extraction标记必须严格为true；额外创建时间/用途metadata不冒充源码身份。冲突身份和未完成封存仍拒绝。失败日志及不可变436e2b8源码保留，不是训练故障，也没有影响已有结果。

私有源码打包器另曾拒绝仓库原有的一个报告symlink；当时未解包任何文件，未发布快照。后续仅原样保留该已跟踪报告symlink且不跟随它。没有修改报告链接或读取其外部数据来绕过检查。

机器可读验收、源码/测试hash及实际状态：[final_handoff_validation.json](final_handoff_validation.json)。详细私有账本位于 `/mnt/project/fm-transfer-20261016/qualification/final_handoff_v1`；未上传权重、缓存、轨迹或场景图片。

## 实际运行入口

以下同一命令已先加`--once`执行等待资格，再去掉`--once`启动现有owner。**现有owner仍运行时不要再次执行**；普通状态读取不需要重启。将环境与cwd绑定到同一个不可变快照：

```bash
cd /mnt/project/fm-transfer-20261016/source/ad73d13ec67d880a8be123f9526b575958cf8aff
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/mnt/project/fm-transfer-20261016/source/ad73d13ec67d880a8be123f9526b575958cf8aff \
/mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.fm_transfer.final_handoff \
  --control /mnt/project/fm-transfer-20261016/final_handoff_v1 \
  --hosts training-rl-zt2 training-vla-zt training-vlawm-tsinghua-zt \
    training-vlawm-tsinghua-zt2 training-vlawm-tsinghua-zt3 \
    training-vlawm-tsinghua-zt4 training-vlawm-tsinghua-zt5 training-vlawm-tsinghua-zt6
```

`status.json`、`owner.json`、`controller.log`在该control目录；已有owner停止时也必须先核对实际PID/start和阶段证据，不能把响应超时视为可以重启。`STOP_AFTER_TICK`仅用于控制器完成当前tick后退出，不停止其已有worker或训练；本次没有创建该标记。
