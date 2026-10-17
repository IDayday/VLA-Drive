# B组75k相对50k：负向退化集中在少数日志的新增DAC违规

本分析只读取已经完成的1696场景DEV、固定seed2101、两种单候选ODE记录。比较的是同一个B SFT运行的50k与75k，不是RL增量、候选oracle或新的评价。原始指标、请求轨迹和模拟状态均未改写；新增推理、仿真评分和optimizer更新为0。

**每种网格的11个新增DAC违规，分别贡献总负向退化G−的78.59%和70.87%。这些轨迹多数前进更少、平均速度更低，不能统一解释为过度前进或加速。** 该定位不改变[75k完整报告](interim_B_SFT_update075000.md)的结论：DEV均分相对50k回落、配对区间跨零，继续固定100k SFT及后续RL预算。

## 退化集中程度

G−是所有负分差绝对值之和，单位为原始0–1分数的场景求和。下表不是对净均分变化的因果分解；新增DAC包含在原安全人口的新联合违规中，两个比例不可相加。两个网格分别配对，不能把两列相加当作不同场景数。

| 项目 | 原生ODE | shift5 ODE |
|---|---:|---:|
| 总G− | 13.643613 | 15.154599 |
| 新增DAC场景 | 11 | 11 |
| 新增DAC涉及logs | 3 | 4 |
| 各log中的新增DAC数量，降序 | 8/2/1 | 6/3/1/1 |
| 新增DAC对应G− | 10.722941 | 10.740721 |
| 占总G− | **78.593119%** | **70.874331%** |
| 50k联合安全人口中新违规 | 12 | 14 |
| 这些新违规涉及logs | 3 | 5 |
| 对应G−占比 | 81.647051% | 79.126696% |

原生8/11个新增DAC集中在同一个log；shift5为6/11。完整报告继续使用按log聚类的区间，不能把这些场景当成独立训练复现。

## 新增DAC的实际运动变化

以下均为75k减50k，在同一网格、同一场景和同一root seed下配对。

| 新增DAC子集 | 原生ODE | shift5 ODE |
|---|---:|---:|
| 模拟终点前进量平均变化，米 | −0.150087 | −0.207370 |
| 前进更多/更少的场景 | 3/8 | 1/10 |
| 模拟平均段速度变化，m/s | −0.040870 | −0.054727 |
| 请求终点横向坐标平均变化，米 | −0.106314 | −0.090485 |
| 模拟终点绝对横向位移变化，米 | −0.052945 | −0.040775 |
| 请求离散jerk RMS变化，m/s³ | −0.186950 | −0.125981 |

横向量是相对初始车身坐标的位移，不是车道中心误差或道路边界余量；绝对横向位移减少不代表更符合可行驶区域。请求离散jerk也不是官方Comfort。本次没有重建道路边界，尚不能从这些均值判断具体越界位置、朝向误差或地图边界敏感性。

因此，这批B SFT退化没有呈现A20k RL新增TTC案例中普遍前进更多的模式。但A/B的训练阶段、参考点和预算不同，不能据此做架构安全性的因果排名。

同时，两策略都联合安全且PDMS至少提高0.005的场景有63/25个，来自各自8个logs；其中61/63和24/25个前进更多，平均增加0.246563/0.293512米。这些是两次SFT单候选输出的配对改善，不是当前策略候选库中的安全机会率。

## 少量TTC事件仍须分场景看

官方首次TTC违规时间来自保存的0.1秒模拟网格；它是物理时间，不是预估碰撞剩余时间或Flow生成步骤。新旧模拟初始状态逐项相同，相关cache和评分协议身份也一致。

| 新增TTC子集 | 原生ODE | shift5 ODE |
|---|---|---|
| 场景数/log数 | 2/1 | 3/2 |
| 首次违规物理时间，秒 | 3.8、4.0 | 1.3、2.6、3.1 |
| 同一时刻前进更多 | 1/2 | 2/3 |
| 前0.1秒段速度更高 | 1/2 | 2/3 |
| NC不完全通过 | 0 | 0 |

两网格新增TTC集合没有交集；75k现有TTC失败各4个，仅1个共同场景。相同TTC均值可以对应不同失败场景，不能选取每场景更好的网格形成未登记的策略。样本很少，且子集按结果选定，这些只作描述性诊断。

分析器同时保留10:5:2反事实读出：完整DEV平均变化为−0.386918/−0.354612个百分点，按1e−12数值容差未出现“该读出提高而canonical下降”的场景。**SFT实际优化时间加权FM，不使用PDMS奖励；反事实读出不是SFT更新原因。** 新增DAC在该读出下也全部更差。

## 实现、验证与证据

复用原[safety分析器](../../scripts/analysis/fm_transfer/analyze_safety_transitions.py)的指标、分层、物理事件和hash检查，只增加显式`--comparison-kind sft_milestones`。默认仍要求RL相对其精确SFT parent；新模式要求两个SFT、相同run identity/config、递增且已登记的milestones，不接受跨run、反序、越预算或混入RL的比较。输出有独立schema，标记`is_RL_increment=false`及早期SFT参考角色。

17项针对性CPU测试通过；真实完整DEV分析完成。新增/恢复计数、G+/G−及平均分差与已发布配对摘要一致，容差1e−12；输入摘要、manifest和执行期间的源码hash不变。核心分析29.185秒，完整调用31.034秒，均为CPU elapsed；没有GPU任务或新的模型前向。

源码提交：`ec271529ca1d6eeaa912a3436052a8e4df4db99e`。正在运行的SFT、后续RL及数值评价快照均未修改。[完整聚合结果](B_SFT75k_DEV_safety_vs50k.json)和[执行、输入/hash与交叉检查](B_SFT75k_safety_analysis_execution.json)已保存；不上传原始场景或图片。

已实际执行：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.fm_transfer.analyze_safety_transitions \
  --comparison-kind sft_milestones \
  --evaluation /mnt/project/fm-transfer-20261016/evaluations/B_SOURCE_ACTION_DIT_SFT_update075000_DEV \
  --reference /mnt/project/fm-transfer-20261016/evaluations/B_SOURCE_ACTION_DIT_SFT_update050000_DEV \
  --output /mnt/project/fm-transfer-20261016/analysis/B_SOURCE_ACTION_DIT_SFT_update075000_DEV_safety_vs50k.json
```

本轮继续B100k SFT及其自己的20k RL，再判断上述退化是否持续；没有据这份分析调整reward、LR、时间网格、训练清单或选择中期检查点。
