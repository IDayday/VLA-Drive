# A/C 架构实现与有限资格验证

## 实际来源与计算边界

- 本地基线：`367bffd27aff5785fce6352481b89e2a433a9ee1`。
- 直接复用本地 `CleanXActor` 的参数化、时间/ego 调制、输出头及 `ReadBlock`/SDPA。
- 阅读锁定 SimWAM `1433ee7e06a62bd8b89da74455df0d82bda7b726` 的 MoT `forward_action_with_video_cache`，复用动作无关逐层前缀的生命周期思想；已有 vendor 保留其 MIT 许可。
- 本轮的紧凑 `WorldStream`、R 共享路径以及 C 的双 softmax / 分组 K/V 是明确的新比较实现，不声称逐字节复制 Wan 视频专家。

## A：训练期世界内化

`ArchitectureA.actor` 是原独立 clean-x actor：20×1024、16 heads、FFN4096、8×3。
`prepare_prefix` 显式接收 `scene/scene_valid/ego`，不读取旧 `StudentMemory.actor_memory`，因此不能悄悄拼入 predictive32。
训练前置 W32 仅从当前 C 读取；actor 不读取 W。训练后置 R16 用同一 world block 的参数，读取对应层 W 与精确候选在 sigma0 replay 的 actor hidden。
hidden 不 detach，post-only backward 到达 actor shared attention/action input 及当前 scene；最终 action output 不在该路径中，无梯度是设计预期。
RL/部署可用 `need_auxiliary=False` 跳过 world 计算；`prune_training_components()` 删除整个 world 分支。

## C：在线世界前缀与独立 actor

共用同一 `WorldStream` 类，20×384、6 heads、FFN1536、W32/R16。
W 第 l 层输出映射到 actor 第 l 层；20 层分为 4 组，每组5层共享 K/V 适配权重，每层仍输入自己对应的 W 状态。
actor 原 scene cross attention 保留。C 与 W 使用两个独立 SDPA softmax，共享 actor Q/O，按 `u_C+tanh(g_l)u_W` 合并后只调用一次 O，因此不会重复计入 O bias。
`world_gate` 初始化 `1e-3`。没有新增每层独立大 Q/O。
R 的 actor→world 反向桥接只用于训练。导出保留 W/共享 world blocks/world→actor K/V，删除 R query 与 post action bridges。

## 公共组件 API

```python
prefix = architecture.prepare_prefix(scene, valid, ego, need_auxiliary=True)
clean = architecture.denoise_x0(state_fp32, sigma_fp32, prefix)
pre = architecture.pre_reasoning_states(prefix)  # [B,32,384]
post = architecture.replay_candidates(candidate, candidate_to_scene, prefix)
# post: [candidate_count,16,384]
```

prefix 提供 `index_select` 和显式 `detached`；重复 candidate 使用场景索引，不复制原始图像。
这些是底层组件，外层公共合同仍需绑定 scene/policy/precision/mask identity。
C 的当前 world 与桥接参数在 RL 可训练，必须每次当前策略 logprob 调用有梯度重建 prefix，禁止跨 inner 缓存旧图。
checkpoint 为 non-reentrant，显式在 `eval()+autograd` 时生效。

## 参数统计

以下从实际默认实例在 meta device 上按 tensor `numel()` 统计，**不包含公共 Qwen、C160 provider 和任务 readout**。这不是 FLOPs 或吞吐 profile。

| 项目 | A | C |
|---|---:|---:|
| actor（C 包含 world→actor bridge/gate） | 467,119,107 | 470,276,119 |
| world stream（含训练后置部分） | 49,496,704 | 49,496,704 |
| 两者合计 | 516,615,811 | 519,772,823 |
| 裁剪后部署部分 | 467,119,107 | 518,184,087 |
| 被裁剪训练专用部分 | 49,496,704 | 1,588,736 |

## 已执行验证

```bash
PYTHONPATH=. /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m pytest -q tests/world_compare/test_ac_architectures.py
```

结果：**14 passed，3.72s**；另已 `py_compile`。CPU 小宽度、真实模块路径，覆盖：

- 8×3 / W32 / R16 shape、有限输出与 FP32 clean 输出。
- 改 action/sigma 或执行 R 不改变已构建 W，也不改变相同部署动作。
- A 原 actor 权重严格加载后前向与旧 `CleanXActor` 一致；改变 world 不影响 A 动作。
- C gate0 与原 actor 一致，覆盖非零 O bias，避免双 bias 的隐性误差。
- post 梯度到 actor/current scene/world，且不要求未执行的 action output 有梯度。
- C 的小非零 gate、grouped K/V 与 W 主路径获得真实梯度；actor/world 无参数 alias。
- 重复/乱序 candidate scene 索引的缓存结果与逐场景重算一致。
- 部署裁剪保持动作输出相同且参数确实减少。
- `eval()+grad` 的 eager 与 non-reentrant checkpoint 输出/参数梯度一致。
- 无当前有效 token、跨架构 prefix 明确拒绝。

这些测试不替代真实 Qwen/teacher/scorer、分布式有效 update、BF16、恢复、RL fresh ratio 或完整训练。尚未独立启动 GPU 训练；由公共主流程集成并验收后派发，不用资格权重作为正式初始化。

