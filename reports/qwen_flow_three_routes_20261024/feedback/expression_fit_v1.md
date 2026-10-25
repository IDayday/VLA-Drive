# 四场景teacher可表达性probe：真实终点证据

> **验收口径更新（2026-10-25）**：本页所有DEV/VAL、TRAIN、候选搜索或拟合结果仅作诊断，不作为方法有效性、性能提升或方法排名依据。S1/S2/S3已完成完整NAVTEST，当前效力判断见[完整测试报告](full_navtest_feedback_20261025.md)；本页历史诊断不承担效力判断。原实验日期、数值和历史记录保留；详见[当前验收口径](navtest_acceptance_20261025.md)。

这是一项定向TRAIN机制拟合，不是新正式方法臂，也不是泛化或Navtest评测。
同一成熟BASE、同一480张量attention LoRA，固定四个不同logs的TRAIN场景，每scene一条先前真实canonical评分合格teacher。
纯原生正权重FM、无原GT/anchor、独立ε/t，512更新、B4、LR5e−5、8步warmup，参数全程未调。
训练/评价源码锁定`10de893a87c5daf85f985ce86785eb52ae97f846`；评分保持canonical v1.1、原shift5十步单候选ODE。

## 直接结论

**当前480-LoRA能够学到并在原ODE下复现至少这一关键道路安全修复；“该修复完全不可表达”被本探针排除。**
唯一stable-zero场景从五个输出全部PDMS0/DAC0，变为128步五个输出全部严格安全、PDMS均分99.94455；
512步五个输出全部PDMS100，NC/DAC/TTC/EP/Comfort/DDC均1。
实际模拟的未来车身越界比例由44.5%降为0；原始模拟进度约44.382m→44.877m，
所以这是越界后果被修复，不是通过普遍停驶制造高安全分。

| 更新 | 定向TRAIN4 PDMS | 严格安全输出/20 | 零分输出/20 | 到单teacher平均ADE（m） |
|---:|---:|---:|---:|---:|
| 0 | 63.13437 | 11 | 5 | 0.53987 |
| 128 | 88.07689 | 16 | 0 | 0.37628 |
| 256 | 92.23588 | 18 | 0 | 0.27561 |
| 512 | 94.48738 | 19 | 0 | 0.21422 |

**94.48738只属于这4个被训练的定向场景，不能写成Navtest达到92，也不是S3的泛化成绩。**

## 四场景的请求轨迹拟合与真实分项

误差均相对该scene唯一真实评分teacher请求，按全部五seed平均；ADE/FDE为局部坐标米。
heading为物理角差wrap到[−π,π]后的绝对均值，单位弧度；训练依然是源3维直接heading归一化，没有改成sin/cos。

| case | 更新 | PDMS | 严格安全/5 | ADE m | FDE m | heading MAE rad | NC/DAC/TTC |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 stable_zero_repair | 0 | 0.00000 | 0 | 0.56416 | 0.84284 | 0.01864 | 1.000/0.000/1.000 |
| 1 other_safety_repair | 0 | 100.00000 | 5 | 0.86191 | 1.92793 | 0.02573 | 1.000/1.000/1.000 |
| 2 other_safety_repair | 0 | 66.66667 | 1 | 0.42875 | 0.50537 | 0.07044 | 1.000/1.000/0.200 |
| 3 safe_EP_improvement | 0 | 85.87081 | 5 | 0.30464 | 0.79130 | 0.08344 | 1.000/1.000/1.000 |
| 0 stable_zero_repair | 512 | 100.00000 | 5 | 0.17048 | 0.07783 | 0.00821 | 1.000/1.000/1.000 |
| 1 other_safety_repair | 512 | 100.00000 | 5 | 0.24587 | 0.29806 | 0.01084 | 1.000/1.000/1.000 |
| 2 other_safety_repair | 512 | 91.66667 | 4 | 0.17066 | 0.13295 | 0.01277 | 1.000/1.000/0.800 |
| 3 safe_EP_improvement | 512 | 86.28285 | 5 | 0.26987 | 0.74583 | 0.04071 | 1.000/1.000/1.000 |

不能称四条teacher都已完全复制：case2到512步仍有1/5输出TTC不通过，均分91.66667，teacher为100；
case3的teacher为91.08329，学生仅85.87081→86.28285，五个输出都没有达到teacher−0.005。
case1虽被归为历史candidate0安全修复，其本次初始五seed已全部100；128步还曾降到83.33333/TTC均值0.6，
后在256/512步恢复100。分类依据历史candidate0，不等于新的五seed基线也失败，不能把它计成新增修复。

## 为什么不是S3泛化成功，也不能确定唯一原因

与正式S3同时改变了多件事：移除GT项、从最多4条softmax目标改为固定单teacher、
将每scene的teacher曝光从8次增至512次、将自然512场景/B16改为定向4场景/B4。
因此本探针无法区分GT冲突、多目标/模式混合、修复样本监督稀释或有限优化曝光哪个是主因。
它只排除当前模型/480-LoRA对这项安全修复完全无表达能力，不证明任意修复都能复制，
也没有观察未训练场景上的遗忘或新增失败。不能据此启动或宣布后续GRPO已成功。
S3自然人口与DEV的原结果仍是正式对照；本probe权重不得替换其终点或用作新基线制造涨幅。

## 终点、输入与费用审计

- 原生phase COMPLETE、512真实optimizer更新、2048 teacher FM曝光；每场景512次。480/480 LoRA张量实际改变，480个Adam状态均step512，参数及moments均FP32有限，scheduler进度512。
- 初始LoRA SHA与S1/S2/S3完全一致，B初始全零；probe0与既有BASE在同四scene/五seed的归一化动作、请求轨迹和所有原生分项逐值差均为0，没有换弱起点。
- 冻结Qwen/provider/base/reference初末stored hash一致，正式trainer已执行终点完整hash门槛。
- 执行commit的全部注册source-file hash匹配。优化路径只有teacher→原source FM→当前model.policy velocity；没有原GT loss、anchor或reference_velocity调用。reference forward次数未独立遥测，结论来自锁定执行路径审计，不能伪称另有独立计数器。
- 80条真实物理推理和80次真实评分、无额外尝试；全部四观察点保留4×5输出，模型输出没有teacher/scorer筛选。
- 训练FM前向/反向各2048；另80×10=800批ODE速度前向，总2848。Qwen prefill仅4次。update计时1038.847秒、均2.029秒；生成134.967秒、评分等待12.665秒；env worker wall1324.609秒、完整invocation1327.783秒。均为共享GPU worker区间，不是独占GPU计算时间，且不含既有搜索构建成本。

完整80条逐输出标量诊断与hash审计保留在私有`audit_extended_v1.json`；
可提交的无原始场景/轨迹数组聚合为 [expression_fit_v1_summary.json](expression_fit_v1_summary.json)。
本次分析仅读取既有checkpoint与80条记录，没有新模型前向、训练、搜索或评分。

## 2026-10-25验收与费用补充

本报告仅提供TRAIN表达能力机制证据。DEV/VAL、定向TRAIN或任何拟合均分都不承担方法有效性或性能提升判断；正式效力现见[完整NAVTEST](full_navtest_feedback_20261025.md)，不能用本probe替换S3正式终点。

新增外层operator计时为1332.998秒（exit0）。原报告保留的invocation1327.783秒、env1324.609秒及update1038.847秒属于不同嵌套计时范围，不能相加。见[费用范围补充](execution_cost_addendum_v1.json)。
