# D：原生 Qwen3-VL 连续 clean-x 动作主干

状态：源码与 CPU 数学/梯度资格通过；尚未由本子任务启动 GPU、真实分布式 IL/RL 或 Navtest。不能将以下资格等同于训练完成或性能提升。

## 实际依赖与迁移边界

使用安装的 Transformers **4.57.6** 原生 Qwen3-VL 类，未升级环境。实际模块：

`/root/miniconda3/envs/navsim/lib/python3.9/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py`

该模块 SHA256：`dd63ed3b124232735b3dca1bfa28f9d6b0d3f7182afcb75dde8f3e724b2b22da`。

- 原生 `Qwen3VLTextAttention`、`Qwen3VLTextRMSNorm`、MLP、rotary embedding、视觉 tower、merger、DeepStack 均直接调用已安装实现。
- `_layer()` 的两个 residual sublayer 组织对应安装文件中的 `Qwen3VLTextDecoderLayer.forward`，保留原版权与 Apache-2.0 注释；唯一新增运算是 A 行 norm 后近恒等 time FiLM。
- AUGUstus 固定 [edf3422304e4b2987e839a9212cf1176782e7380](https://github.com/haha-yuki-haha/AUGUstus/blob/edf3422304e4b2987e839a9212cf1176782e7380/AUGUstus_SFT/model/model_qwen3p5.py) 已实际查阅，公开 LICENSE 为 Apache-2.0。它的连续 action suffix、块内双向 attention 和分段 prefix cache 是接口参考；本实现未复制 Qwen3.5 的 Norm、GatedDeltaNet、混合缓存或其 TimeEncoder。
- 不将这项必要的 Qwen3-VL 适配称为 AUGUstus 官方完整复现。

现有共同通用权重配置实际为：文本 28 层、hidden2048、MLP6144、Q16 heads、KV8 heads、head_dim128、RMSNorm eps1e-6，mRoPE section [24,20,20]；视觉 DeepStack 对应视觉层索引 [5,11,17]。Qwen input embedding 和语言层原始形状未改变。没有独立 500M actor，也不走 C160。

## 在线/训练路径

`encode_current(processor_batch, ego)` 只执行原生 token embedding、视觉/merger/DeepStack 编码、原生图像 placeholder scatter 和 `get_rope_index`。输出 DCurrentEncoding 不含文本 C/P 的 policy KV。

`prepare_prefix(current)` 依次执行：

1. C 的原生因果语言层，保留原图像 mRoPE/DeepStack；无 padding 的 SDPA 使用原生 causal fast path。带 padding 或 eager 使用明确 additive causal mask。
2. 一个合法当前数值 ego token 与 P32 suffix。ego 不读取 P；P 读取 C/ego/P，P 内双向。二者均不接收 action sigma。
3. 保存每层正确的 norm、Q/K norm 与 rotary 后 K/V，保存 P32 最终 hidden；没有保存一个 final P 再伪装成逐层状态。

`denoise_x0(state,sigma,prefix)` 从同一不可变 C+ego+P cache 开始，A8 块内双向，原生语言层直接输出 clean x。每层两个 Norm 后有独立 `256→2H` FiLM，只有 A 行应用，输出层全零初始化。time embedding 是独立的轻量 256 维 MLP。

`replay_candidates(exact_normalized,scene_index,prefix)` 对真实记录的候选直接执行 sigma0 A8/R16 suffix。A 不读 R，R 读同候选 A/R 与 CP。candidate hidden 不 detach；post loss 可更新 action input、共享语言层及 P，且不要求 action output head 得到 post-only 梯度。R labels 不作为模型输入。

每次 suffix 使用临时 `_OneLayerCache`，只拼接返回的新 tensor，不修改父 cache。显式 non-reentrant checkpoint 的 closure 每次创建独立临时 cache，支持 `eval()+grad`，不依赖 HF 的 training-only 开关。

## 同场景候选的前缀内存

相同 scene 的 G 候选使用 view/expand，不复制输入图像。进一步将候选 A/R 放在 query-token 轴，通过块对角 mask 禁止候选互读，原生 SDPA 只保存一份长 CP K/V；每层临时 K/V 长度为 `CP + G*A` 或 `CP + G*(A+R)`，不是 `G*(CP+A)`。每候选仍使用同样的 action position 与 sigma；无候选间信息交换。

该 packing 可关闭供数值 oracle 使用。缓存/非缓存和 packed/独立 batch 的输出及梯度已分别对照。正式 BF16 GPU fresh-ratio 仍须由共同资格入口验证，CPU 对照不能代替实际平台验收。

## 显式参数规模

根据真实 2B config 在 meta device 构建后逐参数统计，不加载额外驾驶权重：

| 项目 | 参数数 |
|---|---:|
| D 总参数，不含外部 readouts | 2,190,965,251 |
| pretrained language | 1,720,574,976 |
| pretrained vision backbone | 306,242,560 |
| pretrained merger/DeepStack projections | 100,714,496 |
| new action/ego interfaces | 4,245,507 |
| new reasoning/role tokens | 106,496 |
| new time MLP/FiLM | 59,081,216 |
| 其中逐层 FiLM | 58,949,632 |
| 可删除 R query | 32,768 |
| RL native text Q/K/V/O Linear | 112 modules |
| RL r16/alpha32 LoRA 参数 | 6,422,528 |

部署移除 R query，并移除公共训练 readouts；Qwen 语言层每个动作步仍全部参与，不能将“没有独立 actor”解释为计算量更小。`training_queries=False` / `discard_training_only()` / `deployment_state_dict()` 支持真实去掉 R 参数，去掉前后 denoiser 输出逐值相同。

## RL 生命周期适配

新增 `starVLA/rl/qwen_world_compare/rl_scope.py` 和 `policy_adapter.py` 复用现有 NativeRecipeActor 与 SourceLoss，未改变 rigorous SDE、clamp、dimension-mean logprob、advantage、PPO 或 source transition-mean anchor 数学。

- A：仅独立 actor self/cross q/k/v/o。
- B：共享 deployment trunk self/cross q/k/v/o，故 P 随 LoRA 更新。
- C：actor、在线 world self/cross attention 和实际 world→actor K/V bridge；不装 R-only bridge。
- D：仅原生 Qwen 文本层 q_proj/k_proj/v_proj/o_proj，vision/merger、FFN、norm、embedding、FiLM、action I/O 全部冻结。
- installer 按 Linear 对象身份去重，将同一个 wrapper 写回全部已有 alias；不会对共享 P/A/R 逻辑角色重复创建 adapter。
- 永久 reference 为本组 IL 模型的独立冻结 deepcopy，尚未采用共享 base 存储优化；必须实测其显存成本。
- ABC 可跨 inner 保留冻结 C；D 可保留冻结视觉/embedding 产物。每一次 logprob 调用都重建 current P/W/text C+P，保留梯度，同次 G8/三步共用，调用结束不留旧图。
- 全部概率仅在正 sigma 进行一次 `(FP32_state-FP32_pred_x0)/FP32_sigma`；actor 不接收 teacher、奖励或未来目标。

## 已执行验证

```bash
PYTHONPATH=. /mnt/project/flow-safe-improvement-20261013/venv/bin/python -m pytest -q \
  tests/world_compare/test_d_qwen_flow_adapter.py \
  tests/world_compare/test_rl_adapter.py
```

实际结果：**33 passed**，一次 legacy pynvml deprecation warning。测试使用本机 tiny native Qwen（3 text blocks，实际 native tiny vision/merger/DeepStack、GQA、padding），无网络权重下载，无 GPU 占用。

覆盖：原生 C forward/DeepStack 对照；C/P/A/R full-mask 与 cached 输出/梯度；同场景 packing vs independent batch 输出/梯度；无 old A cache 累积；R 不影响 A；interface 冻结 base 时新增模块仍有梯度；RL prefix 对文本 attention 更新敏感；current/ref 各一次 prefix；actual sampler mean/std 与重算逐值一致；fresh ratio <1e-4；四个实际 AdamW inner steps 且 old logprob 不变；所有 frozen base/reference 张量不变；B/C/D current KV 保留梯度；共享 Linear alias 只安装一份 LoRA。

尚未完成：全尺寸真实图像 GPU 数值、实际八卡更新、恢复和训练 profile；这些由主 campaign 集成后运行。本文件不报告未执行实验或规划分数。


## 新架构 loader、选择、导出和评测适配（后续集成任务）

新增 `loading.py` 只接受 `qwen_world_compare_IL_v1` / `qwen_world_compare_RL_v1`，不复用历史World构造器。IL使用同一 native inventory/seal 检查，再按本组 config 构造、完整 strict load；正式加载/导出复核模型字节。DEV选择可使用明确标注的 inventory-only检查，与完整已封存评测的模型hash关联，避免重读二十份大模型。

`deployment.py` 提供 `load_native_policy` / `export_policy` / `load_export`。导出不改变live模型；A删除训练期World，B删除R query，C删除R query/actor→R bridge，D删除R query，所有组删除readouts。部署重建同一拓扑后严格加载，保留LoRA wrapper，不做未验证merge。processor小文件、模型文件、模型源码及安装Torch/Transformers/native Qwen代码hash分别封存。

`parent_selection.py` 要求全部二十个完整DEV5k—100k后选择每组自己的IL；最大PDMS并列选择更早update。RL窗口固定0/5k/10k/20k/35k/50k，允许0步IL仍为最优。禁止Navtest结果参与选择，固定终点与选中模型角色分开绑定。

`tools/qwen_world_compare/evaluate.py` 复用原World的完整population、单轨feedback、exact轨迹hash和canonical评分服务；新的evaluation schema不伪装成旧World。正式评测读取sealed export，DEV与Navtest采用同一部署字节；Navtest绑定完整DEV selection以及il_endpoint/il_selected/rl_endpoint/rl_selected角色。每worker invocation独立记账，恢复复用数量与物理新推理数量分开，旧耗时不会因覆盖rank摘要消失。主表只由完整scene集合形成，异常/缺失不能从分母删掉。

已实际运行 export.py --help、evaluate.py --help（去掉PYTHONPATH仍可从仓库根目录运行），以及对应CPU测试23项PASS。四组真实tiny decision均验证完整K10的 IL native→export→strict reload、非零LoRA RL native→export逐值相同，覆盖剪裁、dtype/键/hash拒绝、模型选择和成本ledger。测试中仅用小型替代Qwen装载器/processor资产验证序列化合同；不宣称已经完成真实图像部署。

## 可恢复 RL runtime 与公共配方

`runtime.py` 使用同一 source clean-x canonical配方，仅更新四架构身份和真实缓存描述；`configs/qwen_world_compare/rl.yaml` 是唯一公共模板，运行时显式 --architecture A/B/C/D。15171 hard TRAIN成员/顺序复用原物理log隔离的stream，未按各组表现重新挑选。

合法 checkpoint 的 cond_bg 只包含 `{schema: qwen_world_compare_current_descriptor_v1, token, identity, repeats:8}`。identity绑定本组IL/reference、TRAIN manifest、processor文件、precision、mask与代码，恢复重读同一当前观测，不保存旧prefix graph。ABC仅缓存冻结C160，D仅缓存冻结vision/embedding/DeepStack产物；current P/W/text前缀每次概率调用重建。该descriptor合同已与新恢复验证器负责人确认。

runtime独立7项CPU测试PASS：四组同scene的sampling+old/current调用只做一次冻结输入prefill、三次当前prefix；D文本LoRA仍可训练、ABC C160无旧图；恢复descriptor可重建；错误parent/population/architecture/配方被拒绝。正式require_dispatch读取新task grant，未借用旧World服务器授权。

仍待主流程：全尺寸真实GPU/多卡数值与mid-inner恢复、profile、四组完整IL/RL及统一完整Navtest。本子任务没有启动任何GPU，也没有将代码/CPU资格标为实验完成。


## 主 IL 入口独立集成复核

仅新增 `tests/world_compare/test_il_integration.py`，未直接修改主训练器或主模型。最新源码下实际20项CPU测试PASS（21.01s）：16项模型/精度/归约检查，加4项无`PYTHONPATH`直接脚本入口检查。

- 完整默认 A/B/C/D 工厂均可在meta device按真实Qwen配置构造；所有参数FP32。总参数含公共readouts：A 2,710,812,616，B 2,661,697,096，C 2,713,969,628，D 2,220,500,040。四组预训练 language/merger 1,821,289,472、vision306,242,560完全相同。
- 实际主 `FourFrameworkVLA.forward` 通过tiny encoder注入，执行七项共同损失及backward；FP32/BF16 loss输出均FP32。FP32 full batch与相同global denominator的micro累计梯度相对L2为7e-8—2e-7，未发现重复除accum或场景重复权重。
- BF16跨micro布局的整梯度相对L2约0.20%—0.67%，cosine>0.99998；这是CPU AMP归约布局的实测差异，不宣称bitwise相同，也不用于放宽RL fresh-ratio门槛。
- 全invalid监督（含NaN无效标记）下七损失均精确0，无NaN；post选择仍按完整global32决定8个场景，micro1/2/4不改变该集合。
- D interface的action/P/time模块属于new参数组，穿过Qwen仍有梯度；初始LRscale为[0,0,1/3000]，step1000为1001/3000。joint initializer只清预训练Qwen moments，保留已经学习的新接口Adam状态。
- 复用的ZeRO2主循环在全局有效分母后调用`backward(scale_wrt_gas=False)`，每个完整accum窗口只接受一次有效optimizer update；旧completed_update_guard拒绝skip/nonfinite/错误计数。真实DeepSpeed多卡执行仍由GPU资格证明，CPU测试不冒充该项。

发现并及时反馈主负责人两项入口问题：直接脚本无PYTHONPATH时无法导入starVLA，以及deterministic CUDA未设置CUBLAS workspace；负责人已统一修复。另反馈ranking在BF16上先差分才转FP32，负责人已统一改为先FP32。

新增`profile.py`后，又复现直接脚本目录遮蔽标准库`profile`的问题：`torchvision → torch._dynamo → cProfile`导入本地profile文件，导致`AutoProcessor`的延迟导入失败。直接入口统一先从`sys.path`去掉脚本目录，再加入仓库根目录；本子任务修改所辖export/evaluate入口，负责人修改train/profile/campaign。四个直接入口现在均在删除环境`PYTHONPATH`时实际执行`--help`成功。

BF16布局验收保留FP32逐参数严格比较，并用BF16整梯度相对L2与cosine检验，避免因接近0的抵消坐标而扩大逐坐标容差。这是CPU AMP不同归约树的已量化结果，不改变RL实际采样/重算`1e-4`资格阈值。真实BF16 GPU仍需独立资格。此处只修工程/数值合同，不改变四组损失权重、预算或结构。复核修复后未发现阻止GPU资格的其他硬错误。
