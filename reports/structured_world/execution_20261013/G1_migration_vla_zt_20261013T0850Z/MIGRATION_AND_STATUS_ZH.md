# G1 迁移、资源撤回与 Navtest 状态（2026-10-13）

G1 已从最近完整的 80,000 步检查点恢复到 `training-vla-zt` 的八张 A800。
恢复前逐位检查了模型及 BN buffers、所有 rank 的 FP32 master、Adam moments、step/LR 与 RNG。
恢复游标为 epoch25、offset625、exposure2,559,800；截至 09:12:41 UTC，已实际运行
80,001–80,033 共33次更新，曝光、更新顺序、有限损失/梯度及 FP32 master 更新检查通过。

用户已撤回 `training-vlawm-zt`、`training-vlawm-zt2`、`training-vlawm-zt3`、
`training-vlawm-zt5` 的使用权限。共享 dispatch policy 的全局禁止项、各任务活动许可列表、
Navtest quarantine 已同步更新；正在使用的 Navtest 调度代码实际拒绝这四台主机。
没有启动旧控制器、新优化目标或开发集评测，没有终止其他真实任务。

## 原任务在迁移前已经中断

原 G1 torchrun 在 08:28:42 UTC 收到 SIGTERM，最后日志为84,873步。
该事件发生在本次操作前；现有日志不能确定信号发送者。本次迁移未向训练进程发送信号。
原监督进程已以 FAILED 退出，原 GPU 进程不再存在。
最近提交完整状态是 `latest_0080000`，没有更晚的完整/暂存检查点可恢复。
因此恢复边界本身是精确的，但80,001–84,873这4,873次未保存更新必须重跑，不能称为零进度损失。

原 metrics.jsonl 保留且继续原生追加，不截断或覆盖中断前记录。
`METRICS_SEGMENTS.json` 给出了205,482,984字节的续训段起点；统计时应区分旧的未提交分支
与新的重放段，不能把重复 update ID 当作额外模型训练曝光。实际重跑成本仍应计入工程成本。

## 训练与存储合同

- 实际训练源码仍为 `0c54f1c49e7ed2a76b22826b654d6c72625f5b6d`，冻结 worktree 保持干净。
- 同一 G1 identity `8f331622551d1f1c1c439a6d01411affe012a1777a63c61d8525470dc1e3ded5`。
- `io_preserving_v1`，global batch32、microbatch4、world size8、FM repeat8、原10步proposal，
  同一几何/未来表示/FGTR/监督/优化器/100k scheduler；不重置warmup，不更换精度方案。
- 仅更换 Python 环境路径和 labels/DINO 的存储路径，三处 argv 差异被显式检查。
  包版本、Python/Torch ABI、关键模块源码/二进制 SHA 与执行源码摘要匹配。
- labels、DINO、检查点、运行记录、扩展编译缓存均使用 `/mnt/project`。
  图像继续复用接收机已有且与共享副本身份一致的 RGB 副本，不新拷贝整个数据集。
  共享 Python overlay 仍引用接收机原有 DDP base；没有升级或修改现用环境。
- 原生检查点17个文件总45,395,933,904字节全部重新核验SHA256。通过同文件系统硬链接保留
  一份父状态引用，未复制大权重；未为这次短验证额外生成中间权重。
- 101,592场景人口的元数据身份不变；64个真实场景/16批的有序预取检查通过，直接读取与有序预取所得的当前图像、
  导航、ego、标定和全部目标逐字节一致，Python/NumPy/Torch CPU RNG 未被数据读取消耗。

## 运行条件和边界

vla-zt已有八个 IQE rolling-cache producer，各占约10.5GiB；它们全部保留。
G1 使用全部八卡，但当前不是独占服务器。对热身20步后的13步，更新中位数为
**6.341秒**、P90 **7.735秒**；rank0峰值allocated为34,249,770,496字节。
这是共用资源条件下的短窗口，不是性能优化benchmark，也不能代表无争用吞吐。
没有为了迁移改变训练数学定义；逐位恢复不等于后续CUDA浮点执行永久逐位相同。
`REPLAY_LOG_SANITY.json` 另记录前33步与旧未提交分支的loss差异和相同LR/数据游标。

90k检查点继续由既有cadence捕获；100k结束后由迁移监督进程向同一个Navtest队列入队。
不恢复开发集或旧自动Navtest观察器，不重复派发已有80k结果。

## 本次核验到的最新 Navtest

核验时间：2026-10-13T09:12:41.337114+00:00

全部为同一NAVSIM v1 canonical Navtest、12,146场景、FP32/TF32关闭、单proposal10步、
场景绑定采样seed42，主结果为 `q_final`。四组这些最新检查点均无评分失败。
最新完成的一项仍是06:49:44 UTC的G0 60k，之后没有新的完整评分；调度队列当前无在途任务。

| 组 | 最新已评步数 | q0 PDMS | q_final PDMS | 已观察最佳（不同步数，非共同终点） |
|---|---:|---:|---:|---|
| G0 | 60,000 | 88.0481 | 89.0247 | 60,000: 89.0247 |
| G1 | 80,000 | 88.8408 | 89.1511 | 60,000: 89.2823 |
| G2 | 50,000 | 87.1948 | 88.3943 | 45,000: 89.3858 |
| G3 | 100,000 | 88.6159 | 88.7913 | 80,000: 88.9706 |

这些最新权重步数不同，不是共同终点消融；单次训练seed的历史最高分也不能替代训练复跑。
G0 60k比50k高0.1348分；既有log-cluster bootstrap区间跨零，暂不能把该提升视为稳定收益。

## 已执行入口与审计

实际记录目录：
`/mnt/project/structured-world-fgtr-round1-artifacts/20261008/G1_vlawm_zt2_migration_20261013T0850Z`。
完整训练 argv、原 argv、父检查点哈希、GPU UUID、进程身份与保留的缓存进程见
`TARGET_LAUNCH.json`；恢复证据见 `MIGRATION_VERIFIED.json`、`RESUME_BOUNDARY_VERIFIED.json`。

已执行的CPU检查与启动命令如下；启动器保留同run/物理GPU锁并拒绝重复启动。
运行中的监督脚本不修改，仓库中的脚本副本仅作归档。

```bash
/usr/bin/python3 -B /mnt/project/structured-world-fgtr-round1-artifacts/20261008/G1_vlawm_zt2_migration_20261013T0850Z/verify_parent.py
/usr/bin/python3 -B /mnt/project/structured-world-fgtr-round1-artifacts/20261008/G1_vlawm_zt2_migration_20261013T0850Z/resume_g1_on_vla_zt.py
/usr/bin/python3 -B -u /mnt/project/structured-world-fgtr-round1-artifacts/20261008/G1_vlawm_zt2_migration_20261013T0850Z/resume_g1_on_vla_zt.py --execute
```

既有原生安全暂停入口是run目录下的 `STOP_AFTER_CHECKPOINT` 标记；遇到该标记，
训练在完整optimizer边界保存后退出，监督进程记录PAUSED而不盲目重启。
后续恢复仍需核对最新完整检查点和当前资源，不可将本次80k父状态伪装成未来最新进度。
禁止回到用户已撤回的四台主机。旧run、历史结果、负结果和原checkpoint来源保留。
