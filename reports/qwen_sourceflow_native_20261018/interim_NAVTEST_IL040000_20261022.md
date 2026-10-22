# 40k IL全量Navtest与20k配对结果（2026-10-22）

**40k IL的canonical NAVSIM v1.1单候选PDMS为85.786557，相对20k的80.370064变化+5.416493分。** 本轮为用户明确要求的新中期Navtest。它是IL检查点比较，不是RL增益；是否达到92：未达到，差6.213443分。

## 固定评价合同

- 全量12146场景/136 logs、seed2101、单候选10步source shift5 ODE。无best-of-N、搜索、scorer选轨或新增seed。
- 输入为原生Qwen当前三视图、导航与合法当前ego；源8×3 ActionDiT解码。模型、源代码和评分身份由摘要绑定。
- 固定40k里程碑在本轮结果出现前选定，便于与已完成的40k DEV对应；不依据Navtest选择IL父模型或调训练参数。
- 20k与40k使用同一评价源码`777a9bdfc6b67dc9e28f1b00deafbc5930476fe6`、同一数据、噪声、精度、后处理和评分器。实际源文件hash由配对入口逐项核对。
- 历史20k和本次40k Navtest均已观察；后续结果不能再称完全盲测。正式IL源码、100k预算和五个DEV父模型选择规则未改，正式RL尚未启动。

## 完整均值（0—100）

| 指标 | IL20k | IL40k | 变化（分/百分点） |
|---|---:|---:|---:|
| PDMS | 80.370064 | 85.786557 | +5.416493 |
| NC | 96.089247 | 97.781163 | +1.691915 |
| DAC | 90.260168 | 94.492014 | +4.231846 |
| TTC | 88.926396 | 93.018278 | +4.091882 |
| EP | 77.826447 | 80.976504 | +3.150057 |
| Comfort | 99.892969 | 100.000000 | +0.107031 |
| DDC | 96.789066 | 98.287502 | +1.498436 |

按log聚类、保留scene权重的配对bootstrap（2000次）PDMS变化95%区间：**[+4.103521, +6.682706]分**。仅覆盖当前固定训练/推理seed下的场景抽样不确定性，不包含训练seed或推理seed变化。

## 恢复、新失败与收益分解

- 零分场景：1583 → 910；非零→零332，零→非零1005。
- NC/DAC/TTC联合完全通过：9769 → 10690；原联合安全人口新增违规426，原不安全人口联合恢复1347。NC=0.5不计完全通过。
- 正向增益G+贡献9.184207个PDMS点，负向退化G−贡献3.767714点，净变化+5.416493点。
- 任意提高/下降场景：3176/5156；达到预登记0.005原始score幅度的提高/下降：2605/4318。任意负差不等于严重事故。

| 官方安全项 | 20k违规 | 40k违规 | 新增违规 | 恢复 |
|---|---:|---:|---:|---:|
| NC | 503 | 289 | 71 | 285 |
| DAC | 1183 | 669 | 283 | 797 |
| TTC | 1345 | 848 | 186 | 683 |

## 轨迹行为的对应变化

下列是同一全量场景的描述统计，不是额外训练reward。请求轨迹与模拟后果分开保存。

| 统计 | IL20k | IL40k |
|---|---:|---:|
| 请求终点前向位移（米） | 20.096383 | 19.426348 |
| 请求分段平均速度（米/秒） | 5.290431 | 5.059367 |
| 请求分段jerk RMS代理（米/秒³） | 7.009289 | 3.379868 |
| 模拟前向位移（米） | 20.219468 | 19.781219 |
| 模拟原始进度（米） | 21.089621 | 20.489663 |

请求轨迹更平滑，平均行程和速度略减，同时官方EP、NC/DAC/TTC和总PDMS提高。说明收益不能概括为单纯增加前进位移。原始米数和官方逐场景归一化EP不是同一统计；分段jerk代理也不是官方Comfort。此处是关联描述，不能隔离某一模块的因果贡献。

## 执行与成本

实际使用24个不同物理GPU，分布在vla-zt、tsinghua-zt3、tsinghua-zt4各8卡。原ReCogDrive工作保留，未占用Qwen-Flow IL训练卡。tsinghua-zt6按最新用户指示排除；rl-zt2的不可见owner、zt2/zt5无法确认归属的活跃占用未被终止或覆盖。
24个worker全部exit0，且已在三个实际主机核对24个PID/start身份均已退出；逻辑单候选数12146，包括重试的实际推理数12146。未将缺失/异常填0或从分母移除。
端到端控制任务wall（含注册、验证、派发和汇总）789.331秒；worker实际时间跨度714.550秒；累计worker wall15717.035秒，模型加载1563.709秒、前向10889.464秒、评分等待1254.135秒。累计worker时间包含共享GPU占用，不能当成独占GPU计算时间或端到端wall。
确认的tsinghua-zt3八个压力worker通过原agent的PAUSED机制释放，原始argv/environment等私有恢复信息只留在服务器；全部推理worker退出后已恢复原30分钟空闲观察策略，未创建新压力脚本或强行占用真实任务卡。

## 研究边界与接下来的既定任务

该比较直接检验同一原生架构在更多IL更新后的单候选Navtest变化。IL增量不能标作Flow-GRPO收益；当前仍不能回答新原生模型的RL净收益是否成立。DEV不能代替Navtest，也不按两者分数差外推后续成绩。
保留完整100k IL与20k/40k/60k/80k/100k DEV选择，随后同一父模型的CANONICAL与SOURCE两组50k RL。未因本次Navtest改动作结构、奖励、LR、噪声、anchor或预算，也未增加seed矩阵。

## 真实入口与证据

本次执行operator（已有收据目录拒绝重复启动）：`/mnt/project/qwen-sourceflow-native-20261018/interim_navtest_IL040000_20261022/run_evaluation.py`。每个实际worker完整命令及源码身份保存在该目录的`jobs/rank*/spec.json`。
```bash
/mnt/project/flow-safe-improvement-20261013/venv/bin/python -m scripts.analysis.qwen_sourceflow.analyze_il_milestones \
  --evaluation /mnt/project/qwen-sourceflow-native-20261018/evaluations/IL_update040000_NAVTEST_interim_20261022 \
  --reference /mnt/project/qwen-sourceflow-native-20261018/evaluations/IL_update020000_NAVTEST_interim_20261020_v2 \
  --output /absolute/path/to/new_paired_summary.json
```

完整小型证据：[40k摘要](IL040000_interim_NAVTEST.json)、[20k/40k配对结果](IL040000_vs020000_NAVTEST.json)、[评价登记](IL040000_interim_NAVTEST_registration.json)、[执行成本](IL040000_NAVTEST_execution.json)。无权重、原始数据、场景图或巨大cache上传。
