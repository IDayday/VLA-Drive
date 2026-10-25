# TEST唯一效果汇总入口资格

最新口径：**完整NAVTEST提升是方法有效性和性能提升的唯一判断依据**。已有DEV/VAL/TRAIN结果仅为调试与研究线索，旧报告中的“条件支持”等效果标签已被这一口径取代；不重写其历史数值。本入口不会读取这些人口的分数或reference strata。

新增入口：`scripts.analysis.qwen_flow_exploration.summarize_navtest`。

必需参数为`--evaluation --reference --protocol --output`。输入可以是已完成聚合目录，或者同一方法所有completed shard的父目录。后者只调用原`aggregate_sharded_evaluation.aggregate`生成派生symlink聚合，不修改原轨迹、启动推理或重新评分。根owner决定何时执行，本入口不创建GPU任务、轮询owner或选择检查点。已聚合的BASE可被多次比较复用，不重复构建。

正式硬门槛：

- split恰为NAVTEST，全12146场景、136完整logs，所有场景都有固定2101–2105五个单输出。
- qualification与policy_qualification均为false；学习臂必须为登记的完整256更新终点。
- reference必须为未改变的BASE/std1。支持冻结LF十个配置及P0–P5，同一基础模型、source_files、执行等价证据、评分、采样和部署合同。
- 缺失/重复/失败/非有限记录沿用共同验证器拒绝，不能从分母消失。

五seed主结果直接调用`common_analysis.analyze_pair(..., reference_strata=None)`，报告PDMS、全部原生分项、G+/G−、新零分/恢复、NC/DAC/TTC新增/恢复和配对log-bootstrap。seed2101用同一已验证记录单列，保持与五seed均值独立，不重推理、不挑种子。

标签只解释TEST：正差但区间包含零时明确写“测试点估计为正，稳定性不足”；正区间也只支持这个固定测试协议下的观察。0.5点实用量级独立列出，不隐藏更小的正负结果。只训练seed42一次，五个推理种子不算五次训练；log区间不含训练/推理seed不确定性。BASE/固定尺度没有额外方法训练，继承共同基础模型。

不构造NAVTEST“稳定安全”reference集合。原TRAIN512/DEV256分层不适用；若未来研究使用相同测试seed定义BASE安全cohort，也不能称其为独立reference稳定集。

## 已执行CPU smoke

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
  /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.qwen_flow_exploration.summarize_navtest \
  --self-test \
  --output /mnt/project/qwen-flow-three-routes-20261024/qualification/TEST_only_summary_CPU_v3.json
```

[资格记录](CPU_smoke_v3.json)通过8项有针对性的检查，复用真实共同算术和真实双shard聚合器。小型显式fixture为4场景/2logs；仅self-test上下文临时替换计数常量，生产CLI没有放宽人口参数，正式12146/136门禁也实际拒绝了该fixture。

检查覆盖非NAVTEST、资格结果、不同推理源码、缺失scene×seed拒绝；真实五seed均值+4.75点与seed2101−1.25点分别输出；聚合与直接统计一致；不读取旧reference strata；正点估计但区间跨零不能写成功。fixture不是实际Navtest成绩，没有运行模型、评分器或GPU。

输出保留五seed主表、2101次表、完整配对scene CSV与原完成记录成本；派生分析不覆盖旧文件。根owner提交并锁定新分析源码后，再对已封存的真实TEST数据执行。未修改运行中的345源码、actor或公共统计核心。
