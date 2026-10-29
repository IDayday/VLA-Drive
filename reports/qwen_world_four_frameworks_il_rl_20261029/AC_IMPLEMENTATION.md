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

首次结果14项通过；补充 C160 边界与严格部署载入后为 **18 项 A/C 测试**，与10项 track、17项公共loss、6项profile测试一起实际运行 **51 passed，5.27s**；另已 `py_compile`。CPU 小宽度、真实模块路径，覆盖：

- 8×3 / W32 / R16 shape、有限输出与 FP32 clean 输出。
- 改 action/sigma 或执行 R 不改变已构建 W，也不改变相同部署动作。
- A 原 actor 权重严格加载后前向与旧 `CleanXActor` 一致；改变 world 不影响 A 动作。
- C gate0 与原 actor 一致，覆盖非零 O bias，避免双 bias 的隐性误差。
- post 梯度到 actor/current scene/world，且不要求未执行的 action output 有梯度。
- C 的小非零 gate、grouped K/V 与 W 主路径获得真实梯度；actor/world 无参数 alias。
- 重复/乱序 candidate scene 索引的缓存结果与逐场景重算一致。
- 部署裁剪保持动作输出相同且参数确实减少。
- 拒绝旧 `StudentMemory` 或混入 predictive 的192-token输入，包括 A 的无辅助部署路径。
- 筛出的部署 state 与真实裁剪拓扑 strict load 成功，不依赖 `strict=False`。
- `eval()+grad` 的 eager 与 non-reentrant checkpoint 输出/参数梯度一致。
- 无当前有效 token、跨架构 prefix 明确拒绝。

这些测试不替代真实 Qwen/teacher/scorer、分布式有效 update、BF16、恢复、RL fresh ratio 或完整训练。尚未独立启动 GPU 训练；由公共主流程集成并验收后派发，不用资格权重作为正式初始化。

## 当前对象与未来 tracks：v2_native_yaw 已实际准备完整 TRAIN/DEV

新增 `starVLA/rl/qwen_world_compare/track_targets.py` 与
`tools/qwen_world_compare/prepare_targets.py`，固定未来实际时间 1/2/3/4 秒、匹配容差 0.06 秒。
首先按当前标注和三视图可见性复用原 `current_objects` 最多32个对象；未来沿同一 `track_token` 关联，不利用未来存在性重新筛选，不在每个 horizon 重新 Hungarian。
缺失当前身份保留当前 object 但 future mask 为 false；缺失未来、重复未来身份和无效 annotation 明确 mask，不制造零值监督。

原 builder 的 lidar2ego 与官方 ego 平面路径已实测核对：从 GitHub 固定 commit
`3e8291bfa89ff247231e0227778840cd0a036896` 下载的 `navsim_scenario_utils.py`
与本地 v1.1 文件逐字节一致，SHA256
`28f0ee850c35157ab5a9600a3aa68f0f5b0a848b278e7b322b10a000916723f9`。
测试直接提取该原函数做算术 oracle，不把当前整个工作区 Git SHA 当作 NAVSIM commit。
所有被读取的原始 log 帧逐一验证 lidar2ego 为单位旋转/零平移；官方平面 ego→global→固定当前 ego 采用同一 rear-axle/yaw 约定，不混入 roll/pitch。

先运行32个真实 TRAIN 场景（覆盖31个 log）的 smoke，随后运行完整准备；仅使用本地 CPU8 workers，无 GPU。输出：

`/mnt/project/qwen-world-four-frameworks-20261029/targets_v2_native_yaw`

| 项目 | TRAIN | DEV |
|---|---:|---:|
| 完整 scene | 90,890 | 1,696 |
| per-log shard | 1,041 | 16 |
| target 文件总字节 | 786,648,303 | 14,744,193 |
| 有效 track×horizon | 7,560,257 | 143,469 |
| 当前选中 track×4 的可能数量 | 8,394,272 | 158,144 |
| 有效比例 | 90.0645% | 90.7205% |
| 当前无可见 object 的场景 | 2,230 | 39 |
| 坐标审计 raw frames | 587,905 | 11,736 |
| 坐标审计 raw boxes | 57,369,077 | 1,175,597 |

TRAIN/DEV lidar2ego 最大偏差均为0，平面坐标往返最大分别为 `4.688e-10m`/`4.655e-10m`。
TRAIN manifest SHA256 `4d6e4342b75c8e2f36ab7ec4e8336fce87b30561fbdb00d71ee4b155e235089d`；
DEV `01565ddeb78a7070c08dcf2870148f3555df4194a820fd2da03bc25c196ee2b3`。
v2准备进程 PID1727574，耗时约95秒；全 shard hash、全92,586行 shape/有限值/heading单位圆与精确人口验证及v1/v2比较再运行60.40秒，结果 `targets_v2_native_yaw/VALIDATION.json` PASS。
现有 TRAIN/DEV/Navtest 物理 log 隔离收据被复用并再次验证；未建设或读取 Navtest 训练标签。

```python
store = FutureTrackStore(".../targets_v2_native_yaw/TRAIN", max_cached_logs=4)
row = store.get(scene_token)
# row["current_objects"]: 原8维当前 boxes/classes/coverage
# row["future_states"]: float32[32,4,6], 当前固定ego坐标、米及sin/cos
# row["future_valid"]: bool[32,4]
# row["actual_dt_s"], row["future_time_valid"]: 真实时间及缺失mask
```

reader 按 log 做有限 LRU、核对 shard hash、将 numpy target arrays 标为只读。
未来坐标保持物理值；回归归一化在公共 loss 端完成。训练输入 API 不接收这些 targets。
完整 raw/target/shards 不上传 Git，仅提交 schema、准备器、测试、摘要和身份。

实际命令（smoke 在另一独立输出目录加 `--roles TRAIN --limit 32`）：

```bash
PYTHONPATH=. /mnt/project/flow-safe-improvement-20261013/venv/bin/python -u \
  -m tools.qwen_world_compare.prepare_targets \
  --source-data /mnt/project/qwen-world-postcot-20261025/data \
  --raw-logs /mnt/project/DriveDreamer-Policy/navsim_raw/navsim_logs/trainval \
  --output /mnt/project/qwen-world-four-frameworks-20261029/targets_v2_native_yaw \
  --roles TRAIN DEV --workers 8
```



### 保留并拒绝的 v1 坐标遗漏

首版 v1 虽核对 lidar2ego 为 I、执行了原生 `gt_boxes_oriented_box` oracle，却给两边传入同一个自行构造 yaw，遗漏了更上游 `Scene._build_ego_status` 的角度定义。v1 使用 `atan2(R[1,0],R[0,0])`，官方实际使用 `Quaternion(*ego2global_rotation).yaw_pitch_roll[0]`；ego roll/pitch 非零时二者不同。

在正式训练/模型资格启动前定位并修复为直接调用官方同一 pyquaternion 接口。新增测试直接提取官方 `_build_ego_status`，以非零 roll/pitch 检查，而非重复同一个错误前处理。固定3e8291的 `dataclasses.py` 也已下载核对逐字节一致，SHA256 `17d96f48101982d4a0842bb4cc1de441672207ad9fed3e460bbbdcf74440a32b`。

v1缓存和原COMPLETE原样保留，旁路 `REJECTED_FOR_FORMAL_TRAINING.json` 明确拒绝；schema升级为 `qwen_world_compare_current_linked_future_tracks_v2_native_yaw`，新的reader不会接受旧v1。没有正式训练受影响。

v1→v2 的所有当前object与future-valid mask不变，但原未来物理坐标确有误差：

| 差异 | TRAIN | DEV |
|---|---:|---:|
| 有坐标变化的场景 | 88,283 | 1,644 |
| future xy 平均L2差 | 2.673mm | 1.888mm |
| future xy 最大L2差 | 28.409cm | 13.338cm |
| heading 最大差 | 0.005773rad | 0.002181rad |

不能因平均误差小就将其视为原生坐标一致。此处修复了新数据接口，既有Qwen-World历史训练和评分未修改。

## 公共损失与微批次独立复核

`tests/world_compare/test_common_losses.py` 实际17项通过（4.45秒）：无效NaN标签/事件的零损失与有限梯度、NC=.5软标签、pair排序0.02阈值、当前Hungarian映射沿用未来、未覆盖对象不制造负样本、BF16预测与FP32目标、A真实post梯度，以及micro1/2/4下相同global8后置场景及全update分母。

复核中发现并在正式运行前修复了新公共head的两个工程问题：无效标签在BCE/CE前应清洗；`bt/ft`不能以BF16预测dtype建立后再赋FP32真实标签。后者会直接触发index-put dtype错误，而非只影响打印精度。旧科学源码不因此被热修改。

WA query采样新增valid-aware CPU helper：从真实valid patch中无放回，尽量平衡camera/tubelet；不足128时padding坐标带独立false query_valid，gather后与teacher mask相交。每scene独立stateless RNG，micro1/2/4逐值相同。未以重复padding增加有效监督。
