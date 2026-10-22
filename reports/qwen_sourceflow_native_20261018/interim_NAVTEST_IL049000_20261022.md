# 49k IL全量Navtest与40k/20k配对结果（2026-10-22）

**49k IL的canonical NAVSIM v1.1单候选PDMS为86.810426；相对40k的85.786557变化+1.023870分，相对20k变化+6.440362分。** 尚未达到92，差5.189574分。 这是IL检查点结果，不是RL增益。

## 本次实际进度与固定协议

本次启动只读核查时间为2026-10-22 13:29 UTC：正式IL八个rank共同到49,189/100,000步，记录持续增加；最近完整恢复点为49k。按用户要求不高频查询训练日志。约半小时后的独立进度核查为2026-10-22T13:59:38.398807+00:00：八个rank共同到 **49,329/100,000**，累计scene曝光1,578,528；最近500步平均13.166秒/更新，未沿用早期4.5秒估计。正式RL仍待完整IL与五个登记DEV观察点选择共同父模型。
用户明确要求新的Navtest，故在新分数出现前固定最新完整49k恢复点。完整native文件以同文件系统hardlink原样固定，避免rolling清理；没有改权重、optimizer、元数据或milestone清单。49k仍不是正式RL父模型候选。
- 全量12146场景/136 logs，seed2101，单候选10步source shift5 ODE，canonical NAVSIM v1.1。无best-of-N、搜索、GT选轨、额外推理seed。
- 当前三视图、导航与合法ego输入；原生Qwen＋源ActionDiT8×3及原后处理。相同冻结输入、精度、官方评分器和逐候选隔离。
- 新中期Navtest不参与IL checkpoint选择、训练或超参数拟合。已有20k/40k/49k测试均已被观察，后续不能宣称完全盲测。

## 全量成绩（0—100）

| 指标 | IL20k | IL40k | IL49k | 49k−40k |
|---|---:|---:|---:|---:|
| PDMS | 80.370064 | 85.786557 | 86.810426 | +1.023870 |
| NC | 96.089247 | 97.781163 | 98.007575 | +0.226412 |
| DAC | 90.260168 | 94.492014 | 95.372962 | +0.880948 |
| TTC | 88.926396 | 93.018278 | 94.105055 | +1.086778 |
| EP | 77.826447 | 80.976504 | 81.047307 | +0.070803 |
| Comfort | 99.892969 | 100.000000 | 99.942368 | -0.057632 |
| DDC | 96.789066 | 98.287502 | 98.518031 | +0.230529 |

49k−40k按log聚类、保留scene权重的配对bootstrap（2000次）95%区间：**[+0.199636, +1.955108] PDMS点**。不包含训练seed/推理seed不确定性；bootstrap seed不是新增训练seed。

## 新失败、恢复与净收益

- 零分场景：40k的910 → 49k的781；非零→零325，零→非零454。
- NC/DAC/TTC联合完全通过：10690 → 10925；原联合安全人口新增违规404，原不安全人口联合恢复639。NC=0.5不计完全通过。
- G+贡献4.490421个PDMS点，G−贡献3.466551点，净变化+1.023870点。
- 任意提高/下降场景：2813/5656；达到0.005原始score幅度的提高/下降：2216/4696。任意负差不等于严重事故。

| 官方安全项 | 40k违规 | 49k违规 | 新增 | 恢复 |
|---|---:|---:|---:|---:|
| NC | 289 | 253 | 76 | 112 |
| DAC | 669 | 562 | 260 | 367 |
| TTC | 848 | 716 | 174 | 306 |

## 请求轨迹与模拟后果

| 描述统计 | IL40k | IL49k |
|---|---:|---:|
| 请求终点前向位移（米） | 19.426348 | 19.050653 |
| 请求分段平均速度（米/秒） | 5.059367 | 4.962538 |
| 请求分段jerk RMS代理（米/秒³） | 3.379868 | 4.740840 |
| 模拟前向位移（米） | 19.781219 | 19.453055 |
| 模拟原始进度（米） | 20.489663 | 20.056434 |

分段jerk是请求点的描述代理，不是官方Comfort；原始米数与逐场景归一化EP不是同一统计。这些是行为关联，不能单独证明某模块的因果贡献。

## 非里程碑中期评价的实现边界

旧入口仅将20k/40k/60k/80k/100k认作正式IL父模型候选。为了按用户要求测最新49k，本次增加独立、带hash的中期Navtest登记，绑定具体完整检查点、模型hash、全人口、seed、输出目录和禁止选择/调参用途；普通父模型入口仍拒绝49k。没有改milestone清单或借用qualification标记发布正式分数。
新增16项CPU边界验收通过。新旧加载/评价的数学AST在去除唯一登记参数后相同，93个其余数值依赖文件逐字节不变；在同一个49k检查点、同一真实原始图像和初始噪声上，新旧完整10步ODE输出最大差为0。该资格对照2次前向、0次评分，单独计成本；旧入口qualification开关只用于数值oracle，正式全量评价没有使用它。
20k/40k评价源码为777a9bd，本次评价快照为d3a7ce8。差异明确限于检查点资格/登记、只读分析、汇总与controller字段。配对分析默认仍要求源码相同；跨此确切源码对使用封存的兼容收据、8个允许差异文件和上述实测证据，不能掩盖actor/scorer/数据变更。
正式IL持续使用原不可变d4d0012源码，本次不修改其前向、FM loss、optimizer、调度、数据和预算，也不热改现有DEV/RL控制器。

## 资源与成本

本次使用vla-zt、tsinghua-zt3、tsinghua-zt4各8个不同物理GPU，共24卡。三台的确认压力worker经原agent的PAUSED机制释放，私有argv/environment捕获不上库；所有推理退出后恢复原30分钟空闲观察策略。真实/未知任务未终止。
tsinghua-zt2、zt6完全排除；Qwen-Flow训练主机显存不足不参与；rl-zt2的不可见owner及zt5无法确认归属的活跃GPU占用未被使用。
24个worker全部exit0；已在三个实际主机核对24个PID/start身份均退出，三个原压力监控均恢复且当时无压力worker。逻辑单候选12146，实际正式推理12146；另有资格对照2次前向、0次官方评分。worker跨度694.787秒，控制进程wall（含资格等待、检查、派发和汇总）949.729秒。
累计worker wall15785.579秒，其中模型加载1834.067秒、前向10671.573秒、评分等待1259.917秒；资格worker另为248.016秒。累计worker时间不是独占GPU核计算，也不能相加当端到端wall。
固定原生49k恢复点保留40177708013字节，初始采用hardlink、不复制已有payload；原rolling清理后该固定目录继续承担保留责任。无权重或缓存上传。

## 结论与既定后续

49k相对40k净提高1.023870分，NC/DAC/TTC均值改善、EP近乎持平，Comfort小幅下降。仍有325个新增零分和404个原联合安全场景新增违规，不能称为全面改善。净收益由4.490421点正向收益减去3.466551点退化构成。

本次回答的是更多IL更新后的真实单候选Navtest变化，不能当作Flow-GRPO收益。训练仍按100k固定计划和五个DEV父模型选择继续，之后才开展同父模型CANONICAL/SOURCE各50k RL。未依据本次Navtest换模型、调LR/噪声/reward、追加训练预算或seed。

## 真实命令与证据

实际单一operator：`/mnt/project/qwen-sourceflow-native-20261018/interim_navtest_IL049000_20261022/run_evaluation.py`；实际worker命令和源码/卡号见该目录`jobs/rank*/spec.json`。已有attempt拒绝重复启动。
```bash
/mnt/project/flow-safe-improvement-20261013/venv/bin/python -m scripts.analysis.qwen_sourceflow.analyze_il_milestones \
  --evaluation /mnt/project/qwen-sourceflow-native-20261018/evaluations/IL_update049000_NAVTEST_interim_20261022 \
  --reference /mnt/project/qwen-sourceflow-native-20261018/evaluations/IL_update040000_NAVTEST_interim_20261022 \
  --source-compatibility /mnt/project/qwen-sourceflow-native-20261018/interim_navtest_IL049000_20261022/source_compatibility.json \
  --output /absolute/path/to/new_paired_summary.json
```

证据：[完整均值](IL049000_interim_NAVTEST.json)、[49k−40k](IL049000_vs040000_NAVTEST.json)、[49k−20k](IL049000_vs020000_NAVTEST.json)、[执行成本](IL049000_NAVTEST_execution.json)、[检查点固定](IL049000_checkpoint_pin.json)、[中期登记](IL049000_interim_NAVTEST_registration.json)、[新旧数值对照](qualification_rolling_interim_source_pair.json)、[源码diff](rolling_interim_source_diff.patch)、[兼容收据](rolling_interim_source_compatibility.json)。

最新完整训练进度见[完成时进度快照](progress_NAVTEST_IL049000_20261022.json)；吞吐外推仅用于估算，不包含保存、评价和未来共享扰动。
