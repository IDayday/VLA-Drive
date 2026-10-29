# B：共享 P32 / A8 / R16 决策主干

本文件记录已实现的结构与已执行的局部验收，不代表完整 IL/RL 已完成。

## 来源和实现边界

- 本地基础：`367bffd27aff5785fce6352481b89e2a433a9ee1` 的 `qwen_world_postcot/actor.py`、`memory.py`；复用 `CleanXBlock` 参数组织及 `Attention` 的 SDPA。
- 外部设计参考：WEIRDLabUW/unified-world-model，提交 `b7d4067c8859775e5b08f681e76d538015cdd445`。
- 实际检查 `models/uwm/uwm.py`、`models/common/adaln_attention.py`、`models/common/attention.py`、`models/uwm/obs_encoder.py`。UWM 采用联合 token 主干及 AdaLN；其动作采样保留下一观测噪声输入。本轮不迁移该 DDIM/noise 目标或未来噪声部署。
- 指定提交的完整 Git tree 未发现 LICENSE/COPYING/NOTICE 文件，不能据此声称已获某种开源许可证。因此不复制 UWM 源码到本仓库。共享 token 主干作为设计参考，本轮计算实现基于本仓库自己的 CleanXBlock。

外部读取资产在 `/mnt/project/qwen-world-four-frameworks-20261029/upstream/UWM`；仅作来源核验，不是部署依赖。

| 上游文件 | SHA256 |
|---|---|
| `models/uwm/uwm.py` | `add93268d82b784cc733c84f8a09e98115c0acc370fc9894bc9373976ef39e26` |
| `models/common/adaln_attention.py` | `4249637e96f4d0e6a1405ce2e87bb3f83bb5c1c36fdcc603acb5b45d58ac512f` |
| `models/common/attention.py` | `9c6e123d6f3ee079d1ec8a416b0fc7c4c5870fea9624d4b0c31787afc9e99431` |
| `models/uwm/obs_encoder.py` | `4f474dd7b4c2e0d5c3d27e73fb46f27289fc5ab49584ad7e25fe28f3a2817dbc` |

## 实际结构

`ArchitectureB` 默认20层、hidden1024、16头、FFN4096、8×3连续 clean-x。P32、A8、R16使用同一套逐层 self/cross Q/K/V/O、FFN、AdaLN modulation。只有轻量 role embedding、P/R query 和动作接口区分角色，没有三套大主干。

C160 通过每层普通 scene cross-attention 读取；P/A/R 之间使用方向性 self-attention。P只读P，A读P/A，R读P/A/R；C由三种query均可读取。所有role内部双向。布尔mask的True表示允许。

- P的条件是固定sigma0时间编码＋合法ego＋固定P role，不接收动作状态或生成sigma。
- A条件是当前sigma＋同一ego＋A role。
- R条件是clean sigma0＋同一ego＋R role。
- P缓存保存该层真实AdaLN归一化和调制后的self K/V，以及该层scene归一化后的cross K/V；不是用最终P统一投影。
- 每步A只处理8个动作token并读取P缓存。精确candidate replay在sigma0运行A，记录其各层真实输入K/V，再由R读取；candidate hidden没有detach。
- P/A/R共享权重更新后，B的当前P缓存失效。RL每个inner的当前logprob调用必须重新构建有梯度的P；`SharedPrefix.detached()`只供显式冻结数据/reference边界使用，不能据此永久冻结current prefix。

`forward_full()` 是资格测试oracle：把P/A/R同时送进同一共享主干并施加完整方向mask。正式路径使用等价逐层缓存，不需要每个flow step重新处理P/R。

## 参数和部署

通过meta device实际实例化并求和：

| 范围 | 参数数 |
|---|---:|
| 完整B决策模块（不含Qwen/C/readouts） | 467,171,331 |
| 20个共享block | 461,844,480 |
| 独占R query | 16,384 |
| R role向量 | 1,024 |
| 裁剪R后的决策模块 | 467,153,923 |

160个实际self/cross attention Linear对应20层×2种attention×Q/K/V/O。LoRA范围应根据这些实际模块生成，不沿用旧硬编码名单。`strip_training_only()`应只在导出copy上调用，删除R query与R role；共享主干/P/A权重保留。后置任务不经过 `action_output`，因此post-only backward没有其梯度是预期行为。

沿用AdaLN-Zero初始化：第一步self/FFN gate为零，R的直接candidate路径尚未解锁，不能把这时candidate输入梯度为零当成detach。测试实际执行一次Adam更新，随后验证candidate/action_input梯度非零；也在非零gate状态检查完整输出和梯度等价。普通cross残差从初始即参与学习。

## 已执行验收

```bash
/mnt/project/flow-safe-improvement-20261013/venv/bin/python -m pytest -q \
  tests/world_compare/test_b_shared_actor.py
```

13项通过，覆盖：B1/B2、sigma0/0.37、完整方向mask与缓存输出/P/R一致、所有参数及scene/candidate梯度一致、P不依赖动作/sigma、R变动不影响部署动作、candidate-to-scene索引、padding/all-invalid有限、post梯度进入候选嵌入和共享主干、零gate一次真实更新后解锁、eval()+grad非重入checkpoint、参数更新后P变化、删除R前后完整K10 shift5 ODE逐值一致。

这些是CPU小宽度/小层数的结构资格测试。默认完整467M尺寸只做meta参数统计；真实完整模型分布式数值、RL fresh ratio、吞吐、IL100k/RL50k和Navtest仍由公共集成流程完成，不在此虚构训练效果。
