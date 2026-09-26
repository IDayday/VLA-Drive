# Codex目标任务：在VLA-Drive中实现BEV与结构化世界认知V1

你在 IDayday/VLA-Drive 项目中工作。请直接开展代码开发、测试和有预算限制的真实实验，不要只提交调研、设计方案或空壳接口。

## 0. 最终目标与本轮范围

保留当前可复现的Qwen VLM和Flow-Matching DiT动作头，完成以下闭环：

允许的当前图像
→ 图像特征／BEV特征
→ scene与agent tokens
→ Qwen融合
→ 当前3D bbox与周车未来轨迹监督
→ 自车轨迹生成。

最终需要交付：
1. 可关闭、可训练、可推理、可保存恢复的结构化世界分支。
2. 至少一种真实BEV特征提供者，以及完整的外部特征输入接口。
3. 正确的当前目标与未来轨迹标签、匹配和掩码。
4. 同一输入、划分与评测协议下的有限规模对照实验。
5. 可复现实验记录，并将“工程就绪”与“规划有效”分别判定。

本轮实现WCog启发的结构化语义世界认知，不声称严格复现WCog。

暂不增加：
- 联合多车扩散。
- 轨迹VAE。
- Game-CoT。
- 未来RGB重建。
- 强化学习。
- 新scorer。
- 多专家路由。

bbox任务本身是感知，加入未来运动预测后才能讨论动态世界建模。本轮不声称识别了动作干预下的反事实动力学。


## 1. 自主执行方式与工作区安全

先读取当前工作区及上级适用的AGENTS.md、项目开发说明和已存在的运行配置。

检查分支、HEAD、git status、worktree、现有训练进程、可用设备、数据与检查点路径。只使用已分配或明确允许使用的资源，不争抢其他任务设备。

使用隔离worktree和新的：
feature/structured-world-v1-<实际日期>

不要覆盖原工作区，不要自动stash、reset --hard、clean -fd、force push或停止已有任务。

常规实现取舍、路径发现、兼容性修复、测试与预算内实验自行推进，不逐步请求确认。外部付费服务、资源扩容、破坏性操作不在本任务授权范围。

不要只因任务复杂就停在审计阶段。遇到数据／算力阻塞时，继续完成不依赖阻塞项的代码与测试，准确记录未执行项。

不修改原始数据、既有缓存和检查点；新增缓存使用独立版本化目录。

禁止伪造结果、过滤失败样本后只报均值、为达到预期结论改指标或事后换数据划分。

在计划内完成实现、修复和实验后形成最终报告，不无限搜索超参数或无限延长训练。


## 2. 基线选择与已知代码线索

先从本地有效检查点的配置、训练记录和评估入口反查对应代码，选择“可加载并可复现”的基线。

不默认main是实验基线，也不默认最近提交就是最好基线。

可参考的远端分支：
feature/add-agent-query

已核对参考提交：
9dd6b71b324a58de005dc669394295b9354d189c

该提交只作为移植接口的参考，不强制回退到它。需要移植时做最小变更，不整分支合并历史实验。

优先检查：
- starVLA/model/framework/QwenOFT.py
- starVLA/model/modules/action_model/GR00T_ActionHeader.py
- starVLA/dataloader/navsim_dataset.py
- starVLA/cache/navsim_feature_cache.py
- 8-train_agent_action.sh

重点确认以下线索在实际基线中是否仍成立：

1. minimal_agent把agent tokens放在action tokens之前。
2. 已有agent_dino_head是特征监督，不是完整bbox／motion监督。
3. 动作头主要接收action queries。
4. _build_qwen_batch可能在no_grad中提取视觉特征；不能只改requires_grad就宣布视觉可训练。
5. _find_token_positions可能通过argmax定位token；缺失token不能静默返回位置0，必须校验数量和唯一性。
6. Qwen3的image token、DeepStack特征、mRoPE、padding和缓存不能被新token破坏。
7. 训练前向与predict_action是否使用了不同构造路径。

记录到：
reports/structured_world_v1/BASELINE_MANIFEST.json

内容包括：
代码SHA、检查点标识与校验信息、配置、分词器、输入视角／时刻、归一化、轨迹采样、评估器版本、scorer、随机种子、数据划分及基线是否见过开发集。

先在同一批真实场景上保存原实现的固定随机种子输出。

新实现初始化且未更新原参数时：
world_enabled=false必须恢复原路径；
不插入新token；
不改变原prompt、采样与scorer。

联合训练之后不要求已变化的共享权重自动恢复原预测；恢复原策略须切回原检查点或关闭可分离的适配器。

历史辅助头是否加载／计算按原基线保持；研究对照再独立关闭其损失，不能将改变prompt后的模型冒充无损基线。


## 3. 参考实现：先读关键代码，再移植局部能力

只针对本任务查阅以下官方仓库，记录实际使用的commit、文件与license。

### SGDrive
https://github.com/LogosRoboticsGroup/SGDrive

重点文件：
- internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat_wm.py
- point_decoder.py
- qwen_wrapper.py

参考：
world queries的图像条件化、输入embedding替换、输出hidden提取和结构化监督。

不要照搬InternVL主体、静默截断或未经验证的attention mask。

### SparseDrive
https://github.com/swc-17/SparseDrive

重点文件：
- projects/mmdet3d_plugin/models/motion/target.py
- motion_planning_head.py

参考：
检测匹配结果如何分配到未来轨迹、有效掩码及多模态扩展。

第一版不搬整套时序实例队列。

### VGGDrive
https://github.com/WJ-CV/VGGDrive

重点文件：
- inject_utils/Qwen2_5_vggt_fusion_inject_cam.py

参考：
外部特征投影和残差交叉注意力。

第一版不替换Qwen类，不同时增加全层几何注入。

### BEVDet
https://github.com/HuangJunJie2017/BEVDet

参考：
单时刻camera-only的BEV构建与检查点。

不要误用4D历史帧、立体历史或LiDAR融合配置。

### Qwen-Drive
https://github.com/QwenLM/Qwen-Drive-1.0

仅补充参考共享VLM与感知任务的组织方式。
不得将Qwen3.5代码直接替换本项目Qwen3-VL。

可以替换已经失效的仓库路径，但必须核实实际文件，不猜测源码内容。参考模块满足需要后结束调研，进入开发。


## 4. 输入边界与标签契约

研究目标保持部署只读取允许的当前图像、导航、自车状态和已允许的自车历史。默认不增加历史图像或额外相机。

若实际检查点使用三前视等与单前视目标不同的协议：
先复现原协议并明确标注；
建立单前视对照时各组统一输入；
不能把两种协议的差值归因于世界模型；
不得静默扩成环视。

外部BEV编码器也必须满足相同sensor contract。

训练用的地图、bbox、track ID、未来轨迹或LiDAR标注只能进入标签生成／监督，不得作为部署特征。标定信息的使用也要显式记录。

定义独立结构，接口名字可适配现有工程：

ModelInputs:
  current_images
  camera_intrinsics
  camera_extrinsics
  image_transforms
  ego_state
  ego_history
  navigation
  optional_current_feature_cache

WorldTargets:
  current_boxes
  current_classes
  track_ids
  future_xy_in_ego_t0
  future_valid_mask
  current_supervision_mask
  annotation_valid_mask

WorldMemory:
  scene_memory
  agent_memory
  spatial_coordinates
  observation_support
  provider_metadata

训练标签不得进入encode_world或predict_action。

不得使用GT目标数量决定输入token数量。
不得用GT bbox初始化输入agent位置。
不得用GT bbox裁剪部署图像。
不得用GT未来选择高风险输入目标。

GT匹配只用于loss和分析。

当前输入时间必须<=决策时刻。标签可以来自未来，但独立存储。

缓存采用字段白名单，禁止直接把完整Scene对象递给推理provider。


## 5. 模型实现：先完成固定token接口

新增配置总开关和独立模块，避免把全部逻辑继续堆进QwenOFT.py。

可采用：

starVLA/model/modules/structured_world/
  contracts.py
  providers.py
  scene_agent_reader.py
  agent_heads.py
  matching.py
  losses.py
  action_adapter.py

### 5.1 不使用外部BEV的结构化监督基线

从允许的当前Qwen视觉特征产生固定数量的scene／agent queries，通过小型Reader进行图像条件化。

保留原图像token路径，不用world tokens替换它。

建议初始值：
- scene tokens = 64
- agent tokens = 32
- Reader内部维度 = 256

均可配置；Qwen和DiT维度从模型读取，不硬编码2048或固定层数。

原4个DINO agent tokens不能直接充当通用场景的唯一目标容量。

这些建议值是开发起点，不是WCog原论文参数。选择agent容量前统计训练集当前有效目标数量，记录超容量比例与处理规则。

输入顺序：

[原图像／文字／状态]
[scene tokens]
[agent tokens]
[原action tokens]

新tokens通过连续embedding注入，采用单独的token类型与必要的位置编码。

不要把BEV token误标为Qwen原生image token，不改变原图像grid对应关系。

第一次实现保留原因果mask；查询间需要交互时优先放在Reader中，不擅自引入全序列双向注意力。

当前bbox和未来轨迹头必须读取：
“经过Qwen后的agent hidden states”。

可有前置感知辅助头，但它不能替代这一核心监督接口。

### 5.2 当前目标和未来运动头

同一agent slot输出：
- 类别／存在性。
- 当前3D bbox。
- 与该实例对应的多步未来xy轨迹。

先单模态预测，未来步数、时间间隔和时间戳与现有动作监督对齐。

agent输出维度与ego动作编码分开，不能把ego的yaw/sincos等通道直接当作agent xy标签。

尺寸、角度和坐标按真实标注定义编码；建议yaw用sincos，尺寸使用正值参数化。

缺失高度、速度或朝向要掩码，不能猜测补齐。

### 5.3 动作读取方式

默认保持原Flow-Matching DiT及其action-query接口，利用Qwen中的world→action信息流。

额外实现可关闭的WorldToActionAdapter：

H_action_new =
  H_action
  + g * CrossAttention(H_action, H_world, H_world)

作为独立消融，不默认与所有改动同时开启。

门控和残差初始化不能使整个新分支永久无梯度。若用零门控，检查首步门控梯度和后续分支梯度，不要求零门控首步所有支路都有非零梯度。

不把预测框和轨迹序列化为长文本，不额外跑第二次完整VLM推理。

原scorer保持不变；仅在原系统本来使用scorer时沿用它。


## 6. 真实BEV provider：不能只交付占位实现

优先顺序：

1. 本地已有、与允许输入匹配且能验证的camera-only BEV模型。
2. BEVDet单时刻camera-only配置，独立环境加载公开检查点并适配NAVSIM。
3. 确有依赖或检查点阻塞时，完成同一provider契约及轻量校准几何BEV实现，但明确其为待训练模型，不冒充已有外部先验。

第一版可让BEV provider在独立进程／环境离线提取特征，避免旧MMCV依赖污染当前Qwen环境。

必须同时交付：
- 从允许的当前观测生成缓存的真实入口。
- 部署时的在线／服务接口。

不能让正式模型依赖无法重建的测试缓存。

冻结外部provider不等于部署可以删掉它；所有使用外部特征的实验要报告新增推理开销。

BEV应有真实的空间几何定义：
grid范围、分辨率、坐标轴、外参方向、图像裁剪与缩放后的内参。

不能把普通图像特征reshape成H×W就称为BEV。

BEV提供者必须先在NAVSIM训练数据上做当前目标感知核验；需要适配时仅用训练部分。

原nuScenes权重的域外性能不足，不可直接解释成“BEV注入无效”。

记录provider适配的额外数据和训练预算。

BEV特征加入ego坐标位置编码和可计算的观测支持信息，再由Reader提取固定数量scene／agent tokens。

无可靠置信度时不虚构confidence字段；视野掩码不等于真实遮挡可见性或校准风险概率。

随机特征／空provider只准用于测试和对照，不得作为已实现BEV功能的证据。

外部几何模型的非BEV特征可以扩展同一接口，但不在本轮另开一条大实验线。

缓存元数据至少包含：
scene token、决策时间、sensor contract、provider代码／权重标识、图像变换、坐标系、特征维度、grid定义和dtype。

缓存内容必须与增强后的输入一致；无法保证时禁用相关增强或在线重算。


## 7. 标签生成与loss

先审查原始NAVSIM日志和当前预处理pkl是否保留annotations与track ID。

缺失时从原始日志生成独立world-target缓存；不要按每帧数组下标假定同一辆车。

当前预测slots与GT目标进行一次Hungarian匹配。

未来标签沿用匹配到的track ID，在完整有效时间段追踪；不要每个未来时刻独立匹配再拼接。

未来点统一变换到决策时刻ego(t0)坐标系。

先用合成几何测试验证：
“全局静止目标在ego(t0)中位置不随未来自车运动变化”。

再用真实场景检查。

缺失标注、目标暂时消失、超出标注范围与真实空场景必须区分。

未来缺失点只mask，不填零参与回归。

当前只监督约定ROI和观测／标注覆盖区；忽略区域的预测不能错误地当成no-object惩罚。

采用近似可见性规则时明确其限制。

在当前GT数量超过slot容量时，按当前时刻规则处理并报告覆盖率，不用未来危险程度选择GT，不静默删除困难场景。

目标：

L =
  L_ego_FM
  + lambda_cls * L_cls
  + lambda_box * L_box
  + lambda_motion * L_motion

L_motion按有效实例与有效时间点归一化。

当前框、轨迹的尺度统计只来自训练集。

初始权重写入配置，根据小训练集的loss和共享梯度尺度做一次有限校准，之后冻结用于主要对照；禁止按测试分数反复调权重。

空目标batch、全部未来无效batch必须数值稳定。

各分布式rank有效目标数量不同时保证正确的全局归一化。用单卡／双卡等效梯度测试验证，不只看日志平均值。

至少导出32个真实场景可视化：
- 图像中的当前框投影。
- ego(t0)俯视当前框。
- 同一track的未来轨迹。
- 有效掩码。
- 预测slot匹配。

包含转弯、车辆交汇、静止、空目标及目标离开视野等场景；禁止用未来GT筛选正式输入。


## 8. 训练、梯度与缓存策略

默认先冻结原视觉编码器和外部BEV provider。

先训练Reader、queries、投影及bbox／motion头；随后联合现有动作头。

Qwen语言部分的小规模LoRA作为共享配置用于所有相应训练对照，不只给完整模型开放额外训练容量。

冻结Qwen参数不等于Qwen前向no_grad：
新增输入token需要穿过冻结Qwen获得梯度。

只有完全固定、且其输入不含可训练上游的特征提取才可no_grad／缓存。

新增world tokens／Reader尚在训练时，不得复用旧的最终VLM hidden-state缓存。

缓存必须验证tokenizer、prompt、位置编码、视觉配置与相关权重；不匹配就重建，不能静默兼容。

视觉解冻做成独立开关，但不强制纳入本轮全矩阵。

启用时去掉对应no_grad并停用该视觉路径的固定缓存，同时检查实际梯度和参数更新。

损失和梯度日志区分：
Reader、token参数、Qwen LoRA、视觉模块、bbox头、motion头、动作头。

被声明可训练的模块应出现在optimizer中且发生预期更新。

不要在forward里新建或迁移参数模块。

训练／推理共享token和world-memory构造代码。

Flow-Matching重复采样时，world memory、mask、action queries和GT动作必须按同一batch顺序扩展。

保存并恢复所有新增参数、配置、tokenizer、优化器、scheduler、随机状态和采样进度；明确支持的exact-resume边界。

不得靠未经检查的strict=False忽略关键缺失权重。


## 9. 有限预算内的执行顺序

先写 execution_budget.yaml。

已有用户／项目预算优先；若没有，采用以下“试运行上限”，不是论文最终训练配方：

- 真数据过拟合：64个训练场景，最多1000步。
- 首轮pilot：最多8192个训练场景，每个训练变体最多1000步。
- 初筛：1个训练种子，评估使用共同的固定采样种子。
- 仅对最有信息量的一组配对比较追加第2个训练种子。
- 超参数／收敛修复最多2轮，每轮必须提出具体故障假设并记录改动。
- 总训练预算最多12000个optimizer steps，包含过拟合、provider适配、预热、pilot、修复重跑和追加种子。
- 优先完成核心对照，不足时按依赖关系裁剪并标注未执行。
- 不自动开启全数据长训、200 epoch、多候选大规模RL或无限搜索。

执行顺序：

P0：
原基线回放、world关闭回归、数据和标签检查。

P1：
真实64场景拟合当前bbox，再加入未来轨迹，检查可学习性与梯度。

P2：
验证真实BEV provider，完成BEV→Qwen端到端接入。

P3：
运行下述配置的预算内pilot与共同开发集评估。

P4：
完成诊断、选择配对复跑、导出完整训练命令与最终报告。

训练loss下降幅度不设成任意科学结论门槛。

若小集学习不动，依次检查标签／匹配、目标尺度、头部容量、梯度和冻结范围，用“固定特征训练头”等隔离实验定位。

未收敛pilot只允许判为证据不足，不据此宣判方向无效。

工程正确性通过后，规划未提升不应阻止完成其他已计划变体和交付。

但真实泄漏、坐标错误、NaN或检查点恢复损坏必须先修复，不能带病扩大实验。


## 10. 对照配置与评测

交付以下全部配置，并按预算运行：

A0：
原检查点原路径，只评估，验证基线回放。

A1：
原动作模型继续训练，同等数据与优化步数。

A2：
新增同样数量的图像条件scene／agent tokens与Reader，但世界loss关闭，检验额外容量／token作用。

B：
无外部BEV，当前bbox监督。

C：
无外部BEV，当前bbox＋未来motion监督。

D：
有外部BEV，当前bbox监督。

E：
有外部BEV，当前bbox＋未来motion监督。

B/C及D/E各自保持相同模块、初始化、训练范围与优化预算，仅改变相应监督开关。

记录真实参数量和每步成本，不能把外部provider额外预训练当成零成本。

每个变体从同一个基础检查点及可复现的新模块初始化开始。

不得将C的训练结果继续训练成E，再宣称独立公平对比。

若使用阶段预训练，各对照使用对应匹配流程并记录全部阶段预算。

WorldToActionAdapter只在核心矩阵后做一组开／关对比，且占用既定追加预算，不形成无限组合。

优先使用项目已有的固定开发集。

若存在1696场景／16完整log的开发集，先验证manifest和与训练数据的关系，再使用；不要硬造这个划分。

按完整log避免相邻片段交叉。

记录开发集是否已被基础检查点预训练见过，不能将“增量训练未见过”误写成“基础模型完全未见过”。

不使用navtest做：
超参数选择、teacher适配、难例挖掘或选择训练种子。

最终navtest只在模型选择冻结后按项目既定协议执行；超预算则提供命令并标记NOT_RUN。

统一评测：
输入视角、时刻、规划时间跨度、候选数、采样步数、scorer、随机噪声、评估器与失败处理。

v1 PDMS和v2 EPDMS分开，不混算。

至少记录：

感知：
当前目标召回／误报、中心距离误差、朝向误差，以及距离／观测支持分组。

预测：
有效目标ADE／FDE、有效覆盖率。
端到端检测匹配覆盖与匹配目标上的motion误差分开，不能靠漏检困难车降低ADE。

规划：
PDMS与分项、零分比例、场景级配对变化。
有多候选时附候选均值、oracle和选中值，明确oracle非部署成绩。

成本：
参数量、训练峰值显存、samples/s、包含外部provider的推理时延。

开发集差值使用场景配对，并在有足够log时以log为cluster估计bootstrap区间。

单个训练种子的区间不代表跨训练种子稳定性。

关闭／屏蔽world memory、跨场景错配BEV仅作为使用诊断。它们属于潜在分布外干预，不能单独作为因果证明。


## 11. 必须实测的关键测试

1. 初始化且原参数未更新时，world关闭恢复原token序列与原预测；固定噪声，FP32比较中间输出，BF16使用预声明容差。

2. 删除、打乱或替换所有WorldTargets，固定当前输入时预测不变。

3. provider仅读取允许的相机和时间；缺少未来文件时也能推理。

4. token缺失、重复、越界直接报错；batch padding不改变样本结果。

5. 相机缩放／裁剪后的标定正确；坐标变换和静止目标测试通过。

6. 当前匹配与未来track一致；空目标／全无效未来无NaN。

7. 冻结Qwen仍允许新输入token回传；视觉解冻时确有视觉梯度，缓存不绕过训练。

8. 训练／推理条件特征一致；重复采样后条件和样本对应正确。

9. feature cache错版本／错sensor contract会被拒绝；无缓存和合法缓存输出一致。

10. 保存加载新增token、world模块和原策略后预测一致，关键missing/unexpected keys逐项核对。

11. 有设备时进行真实两进程训练、空目标rank和resume测试；无设备就准确标NOT_RUN，不拿CPU mock代替GPU实测。

12. 梯度累积、不同有效目标数、分布式归约与单卡等效性正确。

只新增与本任务相关的测试。
不要把任务扩成整个仓库的全面审计。


## 12. 交付物、状态与结束条件

交付：

- 完整模型／数据／provider实现，无假实现替代核心链路。
- A0/A1/A2/B/C/D/E配置及一组adapter配置。
- 数据目标生成、BEV缓存提取、训练、评估、resume和可视化入口。
- 参数可通过配置或CLI设置，不硬编码开发机路径。
- tests、第三方来源说明及独立依赖安装说明。
- reports/structured_world_v1/下的完整证据。
- docs/STRUCTURED_WORLD_V1_QUICKSTART.md。

报告目录至少包含：
基线manifest、数据／标签审计、架构与梯度路径说明、命令与run ledger、机器可读指标、场景级CSV、可视化和最终总结。

QUICKSTART给出已实测的最短运行命令及完整实验命令；未运行的命令明确标注。

使用CODEX_GOAL_STATE.md保存进度：
已完成事项、当前阻塞、下一步命令、实验预算余量、最近有效commit。

中断后基于真实状态恢复，不能重复已有大实验。

最终分开给出：

engineering_status =
  READY / PARTIAL / BLOCKED

research_status =
  POSITIVE / NO_CLEAR_GAIN / INCONCLUSIVE / NOT_RUN

READY要求：
真实数据端到端训练和推理、真实BEV路径及关键回归验证完成。

只有mock或只完成无BEV版本不能标完整READY。

POSITIVE必须引用配对规划结果及不确定性；只验证bbox／motion改善不能写规划有效。

NO_CLEAR_GAIN应说明训练预算和收敛情况。

未收敛、输入不匹配、provider未验证或样本不足用INCONCLUSIVE。

最终答复必须列出：

- 最终分支和commit、基线SHA／检查点。
- 实际完成的结构及其与WCog的差异。
- 哪些外部特征真实进入了Qwen和动作路径。
- 哪些模块实际训练，哪些固定，哪些缓存。
- 各变体真实结果、失败数量、未运行项。
- 工程结论、研究结论及对应证据文件。
- 一条可恢复任务／复现实验的具体命令。

代码按阶段提交。

若现有认证允许，推送仅本任务的新分支并核对远端SHA。

不自动合并，不推大权重／缓存／私有数据／密钥。

推送失败保留本地提交并报告，不宣称已上传。

现在开始执行：

先核对有效基线与工作区，建立隔离分支；
打通当前bbox＋未来轨迹的真实监督；
然后完成BEV注入和预算内对照实验。

最终目标是可复现的实现与可信证据，而不是预先指定的分数。