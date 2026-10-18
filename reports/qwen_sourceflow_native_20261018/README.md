# QwenSourceFlowVLA：原生 Qwen 与 SimWAM 动作策略

本任务替代旧 DDP/B 系列实验。当前阶段为实现与资格验收，**尚无新模型 IL/RL 或 Navtest 成绩，不宣称达到92**。旧两条170k任务已按用户指令停止，全部八个rank在合法恢复边界一致；停止点为25416和25109（后者保留inner1及旧链），两个后续评价控制器均已退出。见 `old_experiment_stop.json`。

新分支：`feature/qwen-sourceflow-native-vla-20261018`，起点901e3b41。独立资产根为 `/mnt/project/qwen-sourceflow-native-20261018`。旧工作区、权重、结果及其他项目未修改。

## 已实施

- 锁定 SimWAM 1433ee7e 的 ActionDiT、MoT、连续FM scheduler、LoRA与许可证；原始代码仅调整日志import和命名空间。逐文件身份见 `configs/qwen_sourceflow/upstream_lock.json`。
- 官方Qwen3-VL-2B-Instruct权重SHA256已与固定HF revision 89644892核对；processor/config/tokenizer文件也与官方Git blob一致。没有加载WorldAction、DDP或其他驾驶权重。
- 原生Qwen processor、因果前向、mRoPE、DeepStack；不调用语言generate/LM logits。三视图按left/front/right，当前导航与当前状态先于图像。完整视觉token和有效文本分开提取；没有action-query瓶颈。
- 独立QwenConditionProvider：固定单调深度映射、共享视觉bridge、逐层独立K/V及RMSNorm、当前time=0和横向相机tile RoPE；语言bridge保留4096输入，当前8维proprio经源Linear追加context。实际状态为vx/vy/ax/ay和四维导航one-hot（包括unknown），不是先验猜测的7维。
- CachedActionRunner直接使用源MoT缓存动作执行。没有构造假video expert，没有在线Wan视觉网络。padding扩展仅改变mask形状，保留源action计算。
- 源3维absolute normalizer、连续sigma*1000、shift5、noise−data速度、源时间weight、单候选10步ODE。
- IL入口包含1000步以内接口阶段、完整模型阶段、异构LR、global32累积、独立恢复边界与成本账本。当前仍须完整模型/多卡验收，不能将入口存在等同于训练完成。

## 已执行证据

`qualification_math.json`：相同非零外部K/V下，源MoT与迁移runner在B1/B2的输出和全部动作参数梯度逐值一致；有padding输出与逐样本无padding输出一致；源FM函数和ODE符号对照通过。

`qualification_qwen_interface.json`：真实TRAIN三视图通过完整通用Qwen2B和**仅用于接口资格的小动作专家**，625个Qwen张量、21个provider张量、95个小动作网络张量均有非零梯度。此证据不替代完整30层模型的验收，更不代表驾驶效果。该次官方processor实际grid为每视图1×22×42，合并后共693视觉token；按实际grid构造位置，未硬编码为756或压缩至8。固定目标面积384×672，实际官方smart resize按原比例及32像素倍数取整，不另行裁剪/重复resize。

`data_TRAIN.json` / `data_DEV.json`：101592/1696场景，从原始Scene GT构造3维标签。TRAIN少量原路径缺图已定位到 `/mnt/navsim/trainval_sensor_blobs/trainval` 的同一原始JPEG副本，保留路径身份，无黑图替代、无删场景。超出源范围的GT保留；TRAIN有678、DEV有2个场景原始帧时间存在间隔异常，仍按官方Scene索引和0.5秒名义间隔，不自行重采样标签；此数据属性会单列分析。

## 尚未完成

通用Wan2.2-5B全部权重下载、官方插值转换与严格加载；完整模型前向/梯度、图像/导航依赖、批布局/恢复验证；正式IL、父模型锁定；两组50k RL与固定单候选Navtest。不会以随机初始化替代缺失的Wan资产。

## 已登记范围

IL：完整TRAIN，seed42，global32，FM repeat1，接口1000计入100k上限，20/40/60/80/100k观察，DEV PDMS选RL父模型并公开100k终点。只用通用Qwen与通用Wan初始化。

RL：共同选定IL，公开SimWAM hard list与TRAIN交集，G8/K10，末三步SDE，LoRA16/32，源概率裁剪/共同std/x-anchor50/三维权重[.2,1,1]、inner4、AdamW5e-5、有效clip0。CANONICAL为5:5:2主线，SOURCE仅改变为10:5:2；各50k终点，不新增seed/课程/噪声头/replay。

主评价固定seed2101、single-candidate shift5 ODE、canonical NAVSIM v1.1；IL/RL使用相同入口。92是研究目标，不是已实现事实。

### 2026-10-18 second implementation checkpoint (not a training result)

Native 3D runtime now connects to the existing source GRPO loop through one
explicit collection hook. Legacy 8x4 contracts remain strict; a separately named
native identity admits only 8x3 and checks the frozen-observation descriptor.
Source sampler, advantage, PPO, mean anchor, AdamW and inner transactions are
shared. Native observation descriptors reconstruct the same frozen Qwen/provider
condition on resume; dense 30-layer visual KV is not serialized per behavior.

The two complete native RL configs differ only in variant/reward: canonical
5:5:2 versus source 10:5:2. Both retain 50k optimizer updates, four inner updates,
G8/K10/last3, source probability clamps, native anchor [0.2,1,1], seed42 and
single-candidate shift5 evaluation seed2101. They are not running yet; their
formal parent must be selected from all five registered IL DEV milestones.

Executed finite checks:

- `qualification_native_rl_math.json`: zero and nonzero LoRA actual sampling
  mean/std equal recomputation; four real tiny-model AdamW updates; all native
  A/B tensors change after the expected initialization stage; reference/base
  stay frozen; merged velocity agrees. This is synthetic arithmetic evidence.
- `qualification_recovery_shared.json`: two-rank CPU/Gloo real checkpoint fault
  injection remains PASS after the native hook (12 cases, including unfinished
  collection, publication failure, inner1/2/3 resume and journal rollback).
- `qualification_native_transaction_contract.json`: native 3D pending boundary,
  cross-rank advantages and rejection of a changed frozen-condition identity.
- `qualification_observation.json`: full generic Qwen, real TRAIN images,
  camera permutation and navigation dependencies, no future-label dependence,
  explicit absent-camera masking. FP32 padded batch versus individual hidden
  relative RMS errors are 2.45e-6 to 1.69e-5; largest absolute element difference
  is 0.0134. An initial elementwise 3e-4 threshold failed and its private log is
  retained. The revised test records the whole-feature relative error and does
  **not** claim bitwise equivalence. Formal prefill remains batch1, as registered.

All 12146 Navtest raw records are now prepared, solely for later locked evaluation.
No Navtest model score has been computed or used for design. Raw timestamp gaps
remain explicitly recorded (TRAIN678/DEV2/Navtest36); labels follow the official
Scene indexed future convention rather than silently interpolating/filtering.

New IL storage policy: retain every registered milestone and two complete rolling
recovery points. Only superseded new-run model/optimizer payloads may retire;
small manifests, hashes, RNG/cursors and retirement receipts remain. All previous
experiments, failed attempts, and qualification checkpoints are preserved.
This bounds storage instead of producing roughly 100 full optimizer snapshots.

Full Wan initialization, full-model/multicard optimizer and exact recovery
qualification, formal IL, both formal RL lines and driving evaluation are still
pending. No positive driving or 92-point claim is made.

### Generic initialization and complete-model qualification

The original official Wan download completed and verified all three shard SHA256
values. The duplicate HTTP transport was stopped once that verification finished;
its partial files and failed remote-network attempt remain recorded privately.
The conversion strictly loads all 825 Wan state entries and emits 820 action
backbone entries: 300 copied, 520 interpolated, 4 action-encoder/head tensors kept
random by the source policy. `generic_action_initialization.json` records every
axis, alpha, source identity and the actual 3.9GiB payload hash.

`qualification_full_model.json` uses real TRAIN images and the complete generic
initialized 30-layer 8x3 action expert. Action-loss gradients reach 625 Qwen,
156 provider and 824 action tensors; all are finite and nonzero. Peak GPU memory
is 27,393,522,688 bytes for this one-batch forward/backward qualification (no
Adam/DDP state). Dense visual condition is 383,201,280 bytes per scene, hence a
full persistent TRAIN KV cache would be about 39TB and is not constructed.

A two-rank launch uncovered an import side effect before any model update:
the legacy framework registry eagerly imports unrelated modules which initialize
NCCL under torchrun. Native train/load now import the exact same model class from
its independent module; the optional framework file only reexports it. No legacy
registry, Wan online backbone, or early distributed group is imported by the
native entries. Two-rank import qualification passes. The failed pre-training
attempts remain intact; they created no trained initialization/checkpoint.

The full TRAIN raw 3D codec round trip has maximum absolute errors
[5.722e-6m, 2.861e-6m, 1.788e-7rad], without label clipping. LoRA arithmetic
qualification additionally proves A has the expected zero task-gradient at
zero-B initialization and positive gradients in every projection after the first
inner update; B has task-gradients throughout. These are not reward improvements.
