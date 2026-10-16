# DDP与SimWAM的Flow Matching：可复用范围及基模差异

2026-10-16。回答“能否直接采用SimWAM的FM设计，从而直接复用其Flow-GRPO”。本次读取实际F-SFT配置、DDP动作头、锁定SimWAM源码及现有适配器；未改变任何正在训练的模型、配方或部署入口。

结论：**Flow-GRPO核心已经直接复用；两者FM的线性路径和velocity目标可以通过时间/符号转换对接。尚未迁移的是SimWAM的动作SFT时间分布、时间加权损失、动作表达及Wan/MoT条件网络。** 继续迁移这些设计可以形成新的基模对照，但不再是无损替换采样器，也不能承诺自动继承上游RL收益。

这里的“SimWAM”限定于公开commit `1433ee7e06a62bd8b89da74455df0d82bda7b726`。公开SFT默认配置不是已发布91.9权重完整训练历史的证明；不得把论文、当前GitHub和模型卡的设置拼成已验证的单一训练配方。

## 1. 相同的FM数学，不同的训练分布

用`y`表示各自编码后的干净动作，`epsilon`表示各自动作空间中的标准高斯：

```text
DDP:    x_t = (1-t)*epsilon + t*y       target = y-epsilon
SimWAM: x_sigma = (1-sigma)*y + sigma*epsilon
                                       target = epsilon-y
```

设`t=1-sigma`、`v_SimWAM=-v_DDP`，两条线性插值路径和ODE方向一致。
SimWAM沿sigma从1到0积分，DDP沿t从0到1积分；这个符号差异本身不是两种FM算法，也不是尚未修复的bug。

源scheduler将均匀变量`u`变换为`phi(u)=5u/(1+4u)`；模型实际接收sigma时间，积分用相邻sigma之差。不能在适配中又额外乘一次`phi'(u)`。

| 项目 | 本轮DDP实际F-SFT | 锁定SimWAM公开动作FM配置 |
|---|---|---|
| 监督目标 | velocity `y-epsilon` | velocity `epsilon-y`；另有未启用sample分支 |
| SFT时间采样 | `b~Beta(1.5,1)`；`t=(.999-b)/.999` | `u~Uniform(0,1)`；`sigma=phi(u)`，train_shift5 |
| SFT动作损失 | 所有动作元素的普通MSE | 每样本动作MSE乘`scheduler.training_weight(time)`后平均；有padding则按有效动作归约 |
| 模型时间接口 | `long(t*1000)`，1000个bucket | 浮点`sigma*1000`，进入sinusoidal/time MLP；不是整数bucket |
| 原部署网格 | 十步均匀Euler | 配置infer_shift5的十步Euler |
| 本轮DDP RL网格 | 已采用源shift5；另保留原网格评价 | source sampler直接使用其scheduler |

源velocity分支的时间权重来自`exp(-2*(sigma-.5)^2)`，减去锁定网格的最小值，再用该网格的均值常数归一化。它在`training_loss()`的**动作loss路径实际使用**，不是只定义了一个未调用的函数。`sigma_clamp_min=.1`主要服务可选sample/x-prediction的换算；不能把它叠加到已启用的velocity目标。

DDP的`noise_s=.999`留下约0.14996%的轻微负t尾部。这是已知继承行为；本次没有证据证明它导致RL弱收益，也没有据此改原模型。

源码：[DDP动作头](../../../starVLA/model/modules/action_model/GR00T_ActionHeader.py)、[源scheduler](https://github.com/H-EmbodVis/SimWAM/blob/1433ee7e06a62bd8b89da74455df0d82bda7b726/src/simwam/models/wan22/schedulers/scheduler_continuous.py)、[源training_loss](https://github.com/H-EmbodVis/SimWAM/blob/1433ee7e06a62bd8b89da74455df0d82bda7b726/src/simwam/models/wan22/simwam.py#L557)、[源模型配置](https://github.com/H-EmbodVis/SimWAM/blob/1433ee7e06a62bd8b89da74455df0d82bda7b726/configs/model/simwam_navsim.yaml)。

### 时间采样与loss权重需要一起看

本次以100,000点确定性中点积分，直接调用锁定源`training_weight`；没有模型forward或新场景评分。全部换成DDP的生成时间t后：

| t范围（0为纯噪声端） | DDP采样占比 / 普通MSE质量 | SimWAM采样占比 | SimWAM归一化loss权重质量 |
|---|---:|---:|---:|
| t<0 | 0.150% | 0 | 0 |
| [0,.25) | 34.995% | 62.500% | 37.223% |
| [.25,.5) | 29.553% | 20.833% | 37.931% |
| [.5,.75) | 22.821% | 10.417% | 19.727% |
| [.75,1] | 12.481% | 6.250% | 5.118% |

只搬train_shift5而漏掉时间权重，会得到不同于源SFT的监督分配。上表是对loss系数的积分，不是实际梯度贡献，也不是驾驶收益；网络Jacobian和各时间段拟合误差尚未纳入。[执行结果与源文件hash](source_check.json)

## 2. 动作坐标决定探索和保持的物理含义

| 项目 | DDP | SimWAM锁定absolute配置 |
|---|---|---|
| 未来轨迹 | 当前ego坐标系的8个未来位姿 | 当前ego坐标系的8个未来位姿 |
| 模型通道 | `[x_norm,y_norm,sin(heading),cos(heading)]`，8×4 | `[x_norm,y_norm,heading_norm]`，8×3 |
| XY归一化 | 原SFT的均值/标准差 | 固定边界的线性min/max映射 |
| 每单位归一化XY对应物理尺度 | 8.805105m / 2.277741m | 33.37m / 21m |
| heading解码 | sin/cos经atan2 | 归一化角度线性还原 |

源absolute和DDP的XY都不是相邻点增量累加。SimWAM代码虽然支持relative分支，锁定配置没有启用；不能为“更像源”自动切换它。

同一个归一化sigma或MSE系数，在两者中对应不同米级扰动。相同源anchor逻辑权重[.2,1,1]移植到DDP四通道[.2,1,.5,.5]后，同米级XY偏移的相对惩罚约由1:12.63变成1:74.72。这是明确的适配差异；不是精确KL，也不单独证明全部安全退化由它导致。X1已经是独立命名的方向权重对照。

只改XY归一化可以写出可逆仿射映射，但若同时要求root仍为标准各向同性高斯，原模型生成分布一般不会自动保持。3维heading改4维sin/cos还涉及维度和非线性变化；不能给一个末端解码wrapper就宣称中间链及Gaussian概率等价。

来源：[源数据编码](https://github.com/H-EmbodVis/SimWAM/blob/1433ee7e06a62bd8b89da74455df0d82bda7b726/src/simwam/datasets/navsim/navsim_dataset.py#L359)、[既有物理尺度分析](../../ddp_simwam_x_anchor_20261015/flow_design_review/action_scale_comparison.json)。

## 3. 条件网络与动作专家不是可以互换的state_dict

实际DDP配置为24层、hidden1536、交错self/cross-attention、AdaNorm，动作位置使用learned embedding；Qwen特征通过qwen_proj作为条件。原SFT配置的dropout为.2，本轮PG/重算在eval模式，关闭dropout。

SimWAM公开配置为30层、action hidden1024、Wan式ActionDiT，使用RoPE及时间调制，通过MoT每层读取video专家的K/V，并接收文本/ego条件。其SFT实现同时计算video/action FM；默认isolated mask使动作看当前条件帧，不看未来noisy视频帧。RL的当前帧video K/V可冻结缓存，不能据此误认为源依赖部署时GT未来视频。

DDP和SimWAM都有动作self-attention，不能把原DDP说成完全缺少轨迹点之间交互。源配置中的Wan派生动作初始化文件也不是DDP权重的替代文件；层数、宽度、attention/condition接口均不同。

保留Qwen而接入源ActionDiT在工程上可做，但需要训练新的条件映射、适配每层K/V及验证动作专家；这会形成新的SFT基模，不是把模块复制进去就能保留原88.9116起点。若连video专家、VAE、MoT一起迁移，则是新模型系统，需要独立基线、资源和部署成本评价。

来源：[源ActionDiT](https://github.com/H-EmbodVis/SimWAM/blob/1433ee7e06a62bd8b89da74455df0d82bda7b726/src/simwam/models/wan22/action_dit.py)、[源MoT](https://github.com/H-EmbodVis/SimWAM/blob/1433ee7e06a62bd8b89da74455df0d82bda7b726/src/simwam/models/wan22/mot.py)、[源冻结条件接口](https://github.com/H-EmbodVis/SimWAM/blob/1433ee7e06a62bd8b89da74455df0d82bda7b726/src/simwam/models/wan22/simwam_grpo.py#L307)。

## 4. 我们已经直接复用了什么

当前`RecipeActor`继承移植的源`ActionGRPOMixin`，`SourceLoss`继承抽取的源trainer数学方法。`action_velocity()`只连接DDP动作专家、时间bucket和速度负号；不是另一套手写SDE或PPO。

- scheduler、LoRA、随机步骤选择、`sample_action_chain`、`_sde_mean_std`及`action_chain_logprobs`复用源实现。
- 源创新clamp、逐维relative logprob clamp、H/D mean代理保留。
- 源global reward std、group mask、PPO loss及转移均值MSE anchor保留；四次inner由已有训练执行层落实。
- DDP必要适配包括当前观测条件、4维动作/anchor映射、时间方向、概率FP32、分布式恢复及独立canonical评分。

[实际适配器](../../../starVLA/rl/simwam_recipe/trainer_adapter.py)、[上游文件与方法锁](../../ddp_simwam_recipe_20261015/upstream_lock.json)、[源数学及真实模型验收](../../ddp_simwam_recipe_20261015/validation.md)。实现一致性不是效果保证；源概率尾部clamp和部分随机步骤仍属于经验代理，不称完整混合链的无偏梯度。

## 5. 下一步最有辨识力的迁移边界

可以区分三层：

1. **接口/数学层**：已通过adapter直接复用源Flow-GRPO，不要求更换DDP骨干。
2. **动作SFT配方层**：尚未完整对齐源时间采样＋时间加权MSE。若开展新基模研究，先保持Qwen、8×4动作、原DiT、数据和有效SFT曝光，只将这套明确命名的源时间目标作为单因素，与原DDP时间目标匹配比较，再分别接入相同Flow-GRPO。必须先测SFT本身，再测RL增量，不能把新的SFT收益算成RL收益。
3. **动作网络/表示层**：更换ActionDiT、归一化或3维heading属于新的基模设计。需重新建立SFT/reference和部署基线；不要与上一层同时改变，导致无法判断原因。

第二层可以继续使用现有网络权重作初始化，改变时间损失之后需重新监督训练/校准并建立新SFT身份；仅切换推理scheduler不要求重新训练，但已测的同一SFT两种网格表现不同，不能冒充完整FM设计迁移。

这份分析不启动新的训练矩阵。LAST_ODE/SAFETY25的既定20k及固定评价继续，不能中途把其SFT时间目标、动作维度或网络改掉。现有证据支持研究具体的预训练与RL适配，不支持断言DDP的线性FM公式错误，也不支持承诺换成源模块即可达到92。
