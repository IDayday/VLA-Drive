# 固定TRAIN筛选的多机分片入口

为把已空出的GPU用于两组新实验，`dispatch_screen`新增显式`--ranks`。全局仍为24个固定分片、101592个TRAIN场景、各自完整SFT、seed2101、原生10步ODE、每场景单候选和batch1前向。可将0–7、8–15、16–23分别放在三台允许的八卡主机，每卡一个worker；这只改变执行位置，没有新增候选或修改困难阈值。

未指定`--ranks`时仍可以在一台主机放置所有分片。每个分片保留原来的`tokens[rank::24]`、scene绑定RNG和实际评分。实际worker继续使用不可变源`225b9af660cb0210208e2c46fc4216d0f151275b`，新增调度代码独立版本化；不改正在运行的SFT或评价源码。

共享的dispatch锁覆盖合同检查和启动回执写入。各次派发必须属于同一SFT/model/source/global-shards合同，分片不可重叠。部分启动只允许明确选择尚未派发的分片；失败分片须人工核查后走既有逐场景恢复入口，不能借此自动重启或覆盖回执。身份不符、缺少旧合同、已有未登记worker或重复rank一律拒绝。

每个目标GPU保留16000 MiB余量，并按待启动worker每个10000 MiB预留；三机各八分片时每卡入口要求26000 MiB空余。每次启动还会重新读取主机授权与实际显存。不抢占任何真实或未知任务。预留是容量准入规则，不是假称实测独占显存。

已执行验证：

- 27项针对性测试通过，包括原有screen/endpoint准入、三台分片对101592场景恰好覆盖一次、无重复、跨来源拒绝和部分启动边界。
- 真实CLI对A的14k checkpoint拒绝启动，未创建screen输出或GPUworker；完整20k父模型仍是必要条件。
- 在vla-zt持有实际共享文件锁时，rl-zt2及tsinghua-zt均不能取得该锁；释放后两端均可取得。验证使用CPU，不增加策略样本。
- `--help`实际执行；本记录发布时多机正式screen尚未启动，需要等待完整SFT。不能把准入测试当作筛选完成或吞吐提升的实测结果。

可执行入口，在新dispatcher的不可变快照中运行。下面`--source`固定的是旧已验收worker源码；在目标主机依次使用三个互不重叠的rank列表：

```bash
python -m scripts.analysis.fm_transfer.dispatch_screen \
  --sft-checkpoint OWN_COMPLETE_SFT \
  --output OWN_FULL_TRAIN_SCREEN \
  --source /mnt/project/fm-transfer-20261016/source/225b9af660cb0210208e2c46fc4216d0f151275b \
  --shards 24 --ranks 0,1,2,3,4,5,6,7
# 第二台：--ranks 8,9,10,11,12,13,14,15
# 第三台：--ranks 16,17,18,19,20,21,22,23
```

最终只有24个rank全部完成且所有payload hash经`build_hard_manifest`核对后，才发布固定困难清单。新SFT基线、真实八卡RL恢复与部署验收依然是正式RL的前置条件。

实际验收记录位于`/mnt/project/fm-transfer-20261016/resources/screen_partition_qualification/`。端到端吞吐将在正式筛选后报告；不提前承诺线性加速。
