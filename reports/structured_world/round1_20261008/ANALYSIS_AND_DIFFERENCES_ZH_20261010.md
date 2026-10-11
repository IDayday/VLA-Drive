# 第一轮结构化场景监督：结果、配置差异与当前判断

本报告汇总已经实际完成的训练与评价，不把工程短测当作完整实验。
NAVSIM 四组及 nuScenes 两组尚未全部到达共同终点。当前证据支持
“同一模型的轨迹修正可以提高部分规划表现”，尚不能支持
“新未来辅助监督优于一般未来监督或历史最强方法”。

## 1. 新方法与历史方法的区别

主干仍为通用预训练 Qwen、原 DDP Flow Matching 动作头、当前多视图、
导航和合法自车状态。原 FM 定义、repeat8、10步单候选采样保留。
当前 DINO 为各新组共同的固定基础正则，weight1.0，不作为新贡献。

新增真实图像几何 BEV、逐物理时间的未来空间特征，以及复用成熟
可变形注意力的 FGTR 一次轨迹残差修正。W 不直接 reshape 成 BEV。
未来空间与查询位置使用模型从独立噪声生成的 q0；q0 detach，修正损失
仍通过 H_A、未来表示和 FGTR 回传。没有用 GT 未来轨迹作为默认条件。

共同几何准备从公共 ImageNet ResNet50 开始，只训练当前深度、道路和占用，
五个完整训练 epoch；正式驾驶模块使用规定的通用/随机起点。
不从历史 S0 驾驶权重续训，不加载完整 ResWorld 驾驶初始化。
旧未来 DINO、视频、GT-MAE 与暂停的旧优化轨迹研究均保留。

| 组 | 未来目标 | 辅助查询分布 | 作用 |
| --- | --- | --- | --- |
| G0 | 无未来标签，未来空间和 FGTR 仍存在 | 无未来标签监督 | 结构与规划驱动表示对照 |
| G1 | 完整未来占用 Ot | 全局均匀 K=1024 | 一般未来监督 |
| G2 | 条件进入/释放 | 全局均匀 K=1024 | 事件目标组织对照 |
| G3 | 与 G2 相同 | 50% q0完整车身邻域、25%边界/变化、25%全局 | 规划局部监督 |

2026-10-11 执行审阅补充：G3 的准确含义是 planning-local sampling for structured scene supervision。同一 `sample_queries` 同时服务道路、当前占用、未来语义与车身关系，因此 G3−G2 改变整体结构化监督分布。FGTR 读取共享 Bt latent，结构化监督塑造该表示；部署删除语义 head，预测道路/占用数值不作为硬约束输入规划。既有组名、配置和结果数字保持原样。

G1−G0、G2−G1、G3−G2 才分别回答未来标签、事件组织和查询分布的价值。
G3−历史 S0 同时包含结构、当前辅助监督与计算变化；内部 q_final−q0
也不是独立训练 baseline 的因果消融。

## 2. ego-conditioned 的含义与我们的真实实现

ego-conditioned 是让策略依赖自车当前/历史运动信息，不是一种独立算法。
本项目已经实现：

`合法历史运动 → 原MLP状态投影 → Qwen状态token → H_A → 原DDP q0 → FGTR q_final`

nuScenes 用前一时刻与当前 LiDAR 传感器定位姿态，转换到内部向前/向左坐标，
按真实过去时间间隔换算为名义0.5秒运动。四维状态为归一化 dx/dy 和
sin/cos 转角。没有用未来第一步位移估计当前速度，没有显式 CAN bus
加速度通道；它是历史运动代理，不是精确瞬时速度/加速度测量。
状态 token 位于 W/action token 之前，确实进入 Qwen 因果上下文。

未来空间还以模型自己的 q0 为条件。这与“当前自车状态条件化”是两个概念，
均不意味着部署时使用精确 GT 未来坐标。导航采用各组一致的 VAD 高层意图
协议；若意图由日志未来构造，作为协议提供的导航条件披露，不传精确终点。

公开工作的 with-ego 标记并不代表相同输入。UniDriveVLA 发布接口使用六个
CAN bus 状态通道并提供历史轨迹；DiffusionDrive 发布头预测 planning status，
不能把它直接视为和我们相同的 GT 状态输入对照。

实现入口：[nuScenes adapter](../../../starVLA/dataloader/structured_world/nuscenes_adapter.py)、
[当前编码](../../../starVLA/model/framework/DDPForesight.py)、
[完整模型](../../../starVLA/model/framework/vla_structured_fgtr.py)。

## 3. 24 epoch 的来源与实际训练量

24 epoch 来自本轮原始研究范围第17节的预登记，不是从某篇论文抄来的最佳配方。
观察点为6/12/18/24 epoch，统一延长最多32 epoch 需遵循共同规则。
满足第一轮预算不等于已经验证所有模块充分学习。

nuScenes G3 实际完成：23,230个有效 train700 样本，batch32，726 updates/epoch，
17,424次更新与557,520次有效场景曝光。五 epoch 共同几何准备另计116,150次曝光，
不训练规划器。FM repeat8 不把独立场景曝光或 optimizer updates 乘八。

实际 optimizer 为 AdamW 模式，betas0.9/0.95，weight decay0.001，clip norm1。
Qwen 视觉塔冻结、语言部分微调；原动作头/状态投影峰值 LR1e-5，
几何/未来/修正/语义模块1e-4。真实 schedule 为726步预热、按17,424共同终点
余弦衰减至各组峰值5%；最终实际 LR 约5e-7与5e-6。
不能仅凭 YAML 中的 warmup_ratio 或单一 min_lr 推断执行情况。

## 4. 同数据集的公开配方差异

| 工作 | 感知/理解准备与驾驶训练 | Global batch | 主要学习率与优化区别 |
| --- | --- | --- | --- |
| 本项目 | 5 epoch共同几何准备＋24 epoch完整驾驶训练 | 32 | 原动作头1e-5、新模块1e-4；无LoRA/EMA |
| VAD | 48 epoch感知/运动/地图、规划loss为0；再12 epoch规划监督 | 8 | AdamW2e-4；backbone乘0.1；cosine、clip35 |
| DiffusionDrive | 发布配置100 epoch SparseDrive感知准备；再10 epoch规划 | 准备64；规划48 | 准备4e-4、规划3e-4；cosine |
| ResWorld | 先加载GeoBEV感知权重，再12 epoch | 论文8；发布默认在8卡下16 | AdamW1e-4；发布step24超出12epoch，不发生该次衰减 |
| UniDriveVLA | 3 epoch混合VQA准备＋30 epoch联合训练＋15 epoch专家训练 | 发布默认128 | 联合2e-4、VLM乘0.5、LoRA/EMA；最后1e-4且冻结VLM |

主要来源：[VAD论文](https://arxiv.org/html/2303.12077v3)、
[VAD阶段配置](https://github.com/hustvl/VAD/tree/1688c4b1c3a9e2e7873ca9700ff8058170c0e3c8/projects/configs/VAD)、
[DiffusionDrive论文附录](https://arxiv.org/html/2411.15139v3)、
[DiffusionDrive感知准备配置](https://github.com/hustvl/DiffusionDrive/blob/ae54fd87b32b3762f20e63ffd0af91d343cade85/projects/configs/sparsedrive_configs/sparsedrive_small_stage1.py)、
[规划配置](https://github.com/hustvl/DiffusionDrive/blob/ae54fd87b32b3762f20e63ffd0af91d343cade85/projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py)、
[ResWorld论文](https://arxiv.org/html/2602.10884v1)与[锁定配置](https://github.com/mengtan00/ResWorld/blob/a5ae973001dd6001d95b3389020a9e263d73a56e/projects/configs/resworld/resworld_config.py)、
[UniDriveVLA论文](https://arxiv.org/html/2604.02190v1)与[发布配置](https://github.com/xiaomi-research/unidrivevla/blob/a93c175af893b35dc16618e659eca4d18bb1ec86/nuScenes/projects/configs/UniDriveVLA/unidrivevla_stage1_2b_cotraining.py)。

这些是公开配方，未取得作者实际训练日志。ResWorld batch 论文/代码不一致；
GeoBEV权重完整预训练曝光与划分尚未在本项目复现核实。
UniDriveVLA的128是32卡×每卡4的配置默认；启动8卡示例不自动证明论文实跑batch。

轮数、更新、曝光和GPU-hours不可互换。DiffusionDrive配置仅5,860次规划更新，
少于我们的17,424，但此前感知准备配置为43,900次更新。
UniDriveVLA默认30＋15阶段合计9,855次配置更新，也不比我们多。
所以不能说较好的公开结果全部来自更多规划 updates。
人口也不同：我们要求合法过去姿态和完整未来，23,230样本不能直接等同
公开配置的28,130 keyframes及部分未来mask协议。

差距的合理解释包括更成熟的感知准备、历史信息、更丰富状态和不同优化配方。
我们随机初始化的大动作头沿用1e-5尤其值得做共同优化对照，但尚无实验能
确定提高LR的收益或把分数差距定量归因给它。

## 5. nuScenes 已完成结果与外部定位

官方val150的4,969个合格样本，6个当前相机、3秒6点、FP32、单候选。
下表为前2/4/6点分别取均值，再平均三个时间窗口；碰撞为box collision。

| 模型观察点 | 平均L2 / m | 平均box collision / % |
| --- | ---: | ---: |
| G1，6epoch | 0.6652 | 0.4813 |
| G3，6epoch | 0.6906 | 0.7614 |
| G3，12epoch | 0.6157 | 0.7357 |
| G3，18epoch | 0.5836 | 0.5484 |
| G3，24epoch最终 | 0.5848 | 0.2516 |

G3最终q0为0.5877m/0.3466%，q_final为0.5848m/0.2516%。修正主要体现为
碰撞下降，L2仅改善0.00284m。18→24epoch的L2局部持平而碰撞继续改善，
不能据此宣布所有目标充分收敛，也不能保证长训补齐差距。

公开前缀平均L2：VAD-Base无显式planner状态0.72m、带状态0.37m，
DiffusionDrive0.57m，UniDriveVLA-2B带状态0.43m，ResWorld带状态0.30m。
本项目接近DiffusionDrive公开L2，尚未达到更强带状态参考的水平。
历史输入、训练准备、人口和碰撞几何差异阻止严格排名。
尤其不能把我们的前缀3秒L2=0.9330m与论文瞬时3秒端点混比；
我们的瞬时3秒L2为1.9498m。

G3最终全局有效标签上的道路MAE1.0679m、当前occupancy IoU0.5139、
未来IoU0.3930、enter recall0.3298、release recall0.5578。
同一固定GT车身查询集共89,442个查询，道路有效43,487、占用有效14,866；
invalid/unknown仍保留并计数，不当作空闲。几何关系误差不是官方碰撞/责任概率。
当前“可见性分组”是标定/ROI支持代理，不能宣称完整遮挡GT分析已完成。

完整协议、端点分表和数值来源见
[外部比较](NUSCENES_EXTERNAL_COMPARISON_20261010.md)、
[训练预算核验](NUSCENES_TRAINING_BUDGET_20261010.md)。

## 6. NAVSIM 最新50k开发评价：是否值得继续

相同锁定v1 canonical、dev1696/16 logs、单候选FP32。全部行均为50k训练步，
不是Navtest，不是每组历程最优权重混合表。

| 模型 | PDMS | NC | DAC | TTC | EP | 零分场景 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G3 q0 | 88.8949 | 99.1745 | 95.9316 | 96.2854 | 82.9244 | 82 |
| G3 q_final | 90.1987 | 99.0861 | 97.1698 | 96.1675 | 84.5026 | 62 |
| 历史S0 | 91.0018 | 99.1450 | 97.9953 | 96.4033 | 85.1029 | 48 |
| 历史S3 | 90.9723 | 99.0566 | 97.9953 | 96.0495 | 85.3506 | 48 |
| 历史V_QUERY | 91.3684 | 99.5873 | 97.8184 | 97.0519 | 85.3336 | 44 |

同一G3修正增加1.3038 PDMS点，按log聚类、保留实际场景权重的95%区间
为[0.6914,2.2614]。修正修复26个初始零分，也新增6个零分；
DAC/EP改善，但NC/TTC下降。这说明残差不是普遍安全改善，需要检查失败转移。

G3−S0为−0.8031点，区间[−2.2174,0.3259]；
G3−V_QUERY为−1.1697点，区间[−2.7545,0.4058]。
相对S0修复16个零分，却新增30个零分。当前均值落后，区间宽且跨0，
既不能宣称超越，也不能把“不显著”理解为已经证明等效。

对实际canonical逐场景分项进一步检查：G3最终62个零分中，48个为
道路合规乘法项归零、13个为责任碰撞项归零、1个同时碰撞与方向项归零。
相对q0修复的26个零分中25个来自道路；但新增4个道路零分和2个碰撞零分。
相对S0新增的30个零分中26个来自道路、4个含碰撞，而修复的16个中
只有12个道路失败。这说明目前主要缺口仍是道路失败覆盖，修正的平均道路收益
没有达到历史S0的稳定性。TTC有6个改善、8个退化，也需要单独检查。
这些是评价器结果分解，尚不是“辅助标签导致失败”的因果解释。
[逐分项失败转移数据](NAVSIM_50K_FAILURE_BREAKDOWN_20261011.json)保留全部计数。

G3 25k→50k：PDMS90.0320→90.1987，道路MAE0.8799→0.7873m，
固定车身道路余量MAE0.3023→0.2614m，占用剖面MAE0.1330→0.1215，
enter recall0.2066→0.2234。空间表示仍在学习，但辅助误差下降没有同时带来
明显的历史规划优势；动态进入召回仍低。

当前判断：值得完成已登记的同架构对照和共同终点，以定位监督价值；
尚不足以宣称方法优于历史强基线，也没有理由仅凭50k否定整个方向。
重点是G1/G0、G2/G1、G3/G2及失败转移，而不是只追加更多组件或只看IoU。
100k终点与scheduler保持原登记；最终Navtest在配置冻结后执行，不按milestone选优。
第二训练seed、额外优化控制和容量控制仍属于后续必要证据，不在此报告中伪装完成。

## 7. 本次归档、代码与复现

本次[snapshot](snapshot_20261010_v1/MANIFEST.json)收录四个已存在正式run：
NAVSIM G1/G3与nuScenes G1/G3。训练曲线截止各自可验证native checkpoint，
排除尚未保存的live tail，每100步记录并保留观察点/端点。
结果包括19条规划行、8个完整场景监督观察点、986条训练曲线，
保留原始结果身份与摘要，未上传样本token、图像、标注cache或权重。

- [训练曲线CSV](snapshot_20261010_v1/TRAINING_CURVES.csv)
- [规划结果CSV](snapshot_20261010_v1/PLANNING_RESULTS.csv)
- [完整场景指标JSON](snapshot_20261010_v1/SCENE_FIELD_METRICS.json)
- [50k成对区间与零分转移](snapshot_20261010_v1/NAVSIM_PAIRED_COMPARISONS.json)
- [可执行导出代码](../../../tools/structured_world/export_experiment_snapshot.py)

模型、adapter、标签构建、多卡训练/恢复与canonical推理代码均在同一任务分支。
NAVSIM运行源冻结为`0c54f1c49e7ed2a76b22826b654d6c72625f5b6d`，
nuScenes为`41da7c198cf0ac4f799aee2a27a0a8126de30230`；
文档和调度代码提交不改变已运行的冻结科学合同。

新增恢复helper是有明确原运行身份的路由工具，保留native模型、optimizer、
scheduler、RNG和数据位置，不是新初始化。nuScenes恢复仍按用户安排后置。
可见性helper通过两个真实标注/预测样本的track分区与空正样本检查，
尚未替代冻结主评价器或形成全验证集可见性结果。
新资源上的profile、恢复和启动状态另见部署记录；资源准备不等于正式训练完成。

有上述本地资产时可重新导出，输出目录必须不存在：

```bash
bash tools/structured_world/export_round1_snapshot_20261010.sh /mnt/project/round1-results-new-snapshot
```

该命令读取当前已保存的native边界，重新运行时可能捕获更晚训练状态；
本次提交的snapshot及其hash保持不变。

仅根据已有canonical结果重算上述失败分解：

```bash
python tools/structured_world/analyze_navsim_failure_transfer.py \
  --snapshot-manifest reports/structured_world/round1_20261008/snapshot_20261010_v1/MANIFEST.json \
  --output /mnt/project/navsim-50k-failure-breakdown-new.json
```

[归档与新增工具验证记录](ARCHIVE_VALIDATION_20261011.json)保留执行过的检查及范围。
