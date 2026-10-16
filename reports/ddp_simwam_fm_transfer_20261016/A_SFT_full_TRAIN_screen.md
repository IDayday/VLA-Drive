# A20k SFT：完整TRAIN单候选筛选

2026-10-16完成。A_DDP_TIME固定20k SFT端点在全部101,592个RL_TRAIN场景、1,176个log上，以native十步ODE、seed2101、每场景单候选、canonical NAVSIM v1.1评分。冻结观测前缀使用已与原部署对齐的condition cache；不存在候选择优或外部轨迹。完整原始观测DEV基线另见[20k SFT结果](final_A_SFT_update020000.md)。

所有24个分片正常结束，101,592份轨迹payload与评分身份逐条核验通过。没有缺失场景、失败评分填零、去掉失败样本或未发布评分请求。全量汇总与物理成本见[A_SFT_full_TRAIN_screen.json](A_SFT_full_TRAIN_screen.json)。

## 人口与固定困难清单

| 单候选事件 | 场景数 | 占完整TRAIN |
|---|---:|---:|
| PDMS严格等于1 | 39,087 | 38.4745% |
| 0.9≤PDMS<1 | 41,138 | 40.4933% |
| 0<PDMS<0.9 | 20,718 | 20.3933% |
| PDMS严格等于0 | 649 | 0.6388% |
| NC/DAC/TTC同时完全通过 | 100,503 | 98.9281% |
| NC未完全通过 | 14 | 0.0138% |
| DAC未完全通过 | 642 | 0.6319% |
| TTC未完全通过 | 461 | 0.4538% |

安全事件可重叠。NC=0.5不算完全通过；三个分项的未通过计数不能相加后当作联合失败数。

固定阈值`canonical score < 0.9`得到**21,367个场景、1,064个log**，占TRAIN的21.0322%。其中20,278个场景在该固定采样下NC/DAC/TTC全部通过，占困难清单94.9034%；其余1,089个有安全分项未完全通过。细分如下：

| 困难子集类型 | 场景数 |
|---|---:|
| 安全分项通过但质量/进度较低 | 20,278 |
| 仅TTC未完全通过 | 433 |
| 仅DAC未完全通过 | 627 |
| DAC与TTC未完全通过 | 15 |
| NC与TTC未完全通过 | 13 |
| 仅NC未完全通过 | 1 |

| 官方分项，0—100 | 完整TRAIN | 固定困难子集 |
|---|---:|---:|
| PDMS | 94.1901 | 82.8719 |
| NC | 99.9897 | 99.9509 |
| DAC | 99.3681 | 96.9954 |
| TTC | 99.5462 | 97.8425 |
| EP | 87.4263 | 65.4065 |
| Comfort | 99.9026 | 99.5367 |
| DDC | 98.5968 | 98.5889 |

这说明本次固定困难人口仍主要偏向安全但进度/质量较低的场景；并不意味着这些场景的所有随机候选都安全，也不意味着组内必然有有效优势。以上是每场景一个固定ODE样本，**不是G8全满分、全零分或等分组普查**。正式RL的有效组和正优势行为分类必须由其fresh rollout另行统计。

清单只包含RL_TRAIN，未混入DEV/Navtest。每个场景对应的score、token与payload身份保存在私有清单，不上传仓库。清单不随当前RL策略表现改变，也不根据Navtest选择阈值。

## 实际成本

- 三台授权主机各8个worker，共24个全局分片，每个分片4,233场景。
- 逻辑单候选评分101,592次；物理新推理101,592次；复用0次；未发布评分请求0次。
- 首worker启动至末worker结束：1,736.657秒，约28.94分钟。
- 累计worker wall：41,237.029秒，约11.455 worker-hours；不能写成独占GPU计算时间。
- 累计模型加载760.413秒；生成31,174.464秒；评分等待59.553秒；worker评分2,940.347秒。
- 生成计时包含condition读取、十步ODE和CUDA同步，不是纯GPU kernel时间；各worker时间求和不等于端到端wall。
- 另行执行完整payload/hash核验232.661秒，使用8个CPU线程；它不产生新的候选或评分。

## 身份与后续用途

- A20k原生model SHA256：`ac52e9346080ce0e6bf27441935ac9aa6acbff239330b316bee36b560486579a`。
- SFT身份：`5018b5ccfae3cd7280475428720d4de06ea9bf6a5586008fb79076ddb7f82ae3`。
- 困难清单内容fingerprint：`6e1d47d30e4ac91ca60a46f36b2a9ef70989e5d5081c67da883ccd2ba00f3d24`。
- 困难清单文件SHA256：`e490778ca017a4e90571d2e66c61f0bea1c4d8d64ef6f14d74d8c0874b294d10`。
- 筛选/验收/正式RL数值源码：`225b9af660cb0210208e2c46fc4216d0f151275b`。分片调度使用`b80cdadcac785a79927d1f7fdad9a482d0d0214e`，没有改变数值worker。

私有资产实际路径：

```text
/mnt/project/fm-transfer-20261016/screens/A_DDP_TIME_SFT_update020000
/mnt/project/fm-transfer-20261016/hard_manifests/A_DDP_TIME_SFT_update020000.json
/mnt/project/fm-transfer-20261016/hard_manifests/A_DDP_TIME_SFT_update020000.verification.json
/mnt/project/fm-transfer-20261016/analysis/A_DDP_TIME_SFT_full_TRAIN_screen.json
```

已执行命令（在对应源码根目录运行）：

```bash
# 锁定数值源码225b9af：完整清单发布前验证全部轨迹payload。
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.fm_transfer.build_hard_manifest \
  --screen /mnt/project/fm-transfer-20261016/screens/A_DDP_TIME_SFT_update020000 \
  --output /mnt/project/fm-transfer-20261016/hard_manifests/A_DDP_TIME_SFT_update020000.json

# 汇总代码56874b6及后续仅报告提交；只读汇总已完成的评分，不再推理。
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.fm_transfer.analyze_screen \
  --screen /mnt/project/fm-transfer-20261016/screens/A_DDP_TIME_SFT_update020000 \
  --hard-manifest /mnt/project/fm-transfer-20261016/hard_manifests/A_DDP_TIME_SFT_update020000.json \
  --output /mnt/project/fm-transfer-20261016/analysis/A_DDP_TIME_SFT_full_TRAIN_screen.json
```

汇总入口拒绝覆盖既有分析，复核时应使用新的输出文件名。下一阶段只从A20k SFT新建R_SOURCE正式RL；本筛选分数是其训练人口基线，不是RL效果，也不是Navtest分数。
