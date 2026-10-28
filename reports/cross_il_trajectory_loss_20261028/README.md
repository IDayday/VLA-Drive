# 不同IL模型的轨迹L1：统一Navtest重算

当前已完整评测的Qwen-World IL53k，在最终请求轨迹的位置L1上低于此前原生Qwen-Flow、原DDP及DDP＋DINO S0；但其PDMS仍低于后两者。因而“轨迹没有学好”不能单凭当前PDMS落后或不同定义的训练loss数值成立。

本次只读复用了五个已完成单候选评测的全部预测，CPU重算GT拟合误差；没有新模型前向、仿真评分、训练更新或检查点选择。当前仍在训练的更晚World权重未在本表冒充为已评测。

## 统一测试口径

同一Navtest **12,146场景**、同一套原始Scene GT、未来8点（名义0.5–4秒）、当前后轴局部坐标（x前/y左，heading弧度）。比较的是部署最终请求轨迹，**不是LQR/bicycle执行后的模拟轨迹**。

位置L1定义为 `mean_scene,time((abs(dx)+abs(dy))/2)`，单位米。纵向、横向分别报告；角度用环绕后的绝对误差，单独报角度单位，不能把米和弧度混成一个物理L1。ADE是XY欧氏距离平均，FDE是第8点欧氏距离。

| 模型 | x L1 m | y L1 m | 位置L1 m | ADE m | FDE m | 角度MAE ° | canonical PDMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 当前Qwen-World IL53k | 0.534737 | 0.194160 | 0.364448 | 0.618126 | 1.466449 | 1.352380 | 88.542266 |
| Qwen-World IL40k | 0.521447 | 0.201686 | 0.361566 | 0.610935 | 1.469677 | 1.390521 | 87.972382 |
| 此前原生Qwen-Flow IL98k | 0.608640 | 0.218613 | 0.413626 | 0.695420 | 1.428167 | 1.363463 | 88.004393 |
| 原DDP F-SFT100k | 0.814310 | 0.212542 | 0.513426 | 0.891980 | 1.901870 | 1.430394 | 88.911606 |
| DDP＋DINO S0 100k | 0.783089 | 0.200772 | 0.491931 | 0.855392 | 1.824821 | 1.340513 | 89.527868 |

原DDP明确指以往RL使用的F-SFT frozen_visual step100000（权重9a26685…），不是把不同DDP权重混作同一基线。约89.5分的DINO模型按已验证的**S0 100k / 89.527868**定位：原DDP动作头＋当前DINO＋全8未来帧DINO等原辅助设置，权重78f8706…。未使用那条来源未核实的外部“89.41”记录。

四个当前/此前Qwen与F-SFT读出seed2101；S0沿用历史已完成评测seed42，未为了本次比较新增seed矩阵。World与原生Qwen-Flow是shift5十步ODE，DDP/S0是各自原始uniform十步ODE；输入、架构、训练人口与预算也并非单因素匹配。因此这些是实际模型端点的同GT误差对比，不是“只更换某模块”的因果实验。

## 配对结果与可作出的判断

当前53k相对三个旧模型的位置L1降低约11.9%、29.0%、25.9%。下面按44个物理log（136个切片合并真实行车log）做2000次配对cluster bootstrap，保留场景权重；负差为当前更低。

| 当前53k减去对照 | Δ位置L1 m | 95% log-bootstrap区间 m | L1相对降低 |
|---|---:|---:|---:|
| 此前原生Qwen-Flow IL98k | -0.049178 | [-0.060512, -0.038733] | 11.89% |
| 原DDP F-SFT100k | -0.148978 | [-0.168574, -0.131766] | 29.02% |
| DDP＋DINO S0 100k | -0.127482 | [-0.147392, -0.110259] | 25.91% |
| Qwen-World IL40k | +0.002882 | [-0.001009, +0.006950] | -0.80% |

区间仅覆盖固定输出下的log采样不确定性，不覆盖训练seed、推理seed或历史检查点选择不确定性。S0与当前seed不同的限制仍存在。

平均位置拟合的改善主要来自纵向：53k的x L1为0.535m，S0为0.783m；横向0.194m与0.201m更接近。S0的角度误差1.3405°稍低于World的1.3524°。此前原生Qwen-Flow的FDE1.428m也低于World的1.466m，不能称所有几何维度全面变好。

World40k到53k，在线TRAIN轨迹L1从0.006731799下降到0.004822835，但最终Navtest位置L1反而从0.361566变为0.364448（差的区间跨零），PDMS则从87.972382升到88.542266。训练时去噪重建、部署最终GT拟合、官方安全/进度评分是三个不同读出。

因此：当前World并非在平均GT轨迹拟合上弱于DDP；PDMS还落后，不能通过继续降低训练L1就推断一定解决。更接近GT与官方仿真后安全/进度更优并不等价。本分析也没有证明任何辅助任务单独带来了收益。

## 原生训练loss为什么不能直接排名

统一使用对应检查点前500个有效optimizer updates；World已all-reduce的8个rank日志是副本，只取一次；此前Qwen-Flow日志是本rank均值，需要对8rank等权平均。

| 模型/窗口 | 原生轨迹训练目标 | 原始数值 |
|---|---|---:|
| World53k | 8×3 min/max归一化clean-x L1，乘总loss系数10之前 | 0.004822835107 |
| 原生Qwen-Flow98k | 8×3归一化、源time-weighted velocity MSE | 0.000275308119 |
| DDP＋DINO S0 100k | 8×4 z-score/heading sin-cos velocity MSE | 0.003338284071 |
| 原DDP F-SFT | 原DDP velocity MSE；本次未找到匹配的clean-x L1日志 | 不编造L1 |

后两种数值**不能作为L1与World比较大小**，也不能对MSE开平方就当L1。目标（clean轨迹/velocity）、动作维度与归一化、Flow时间分布/权重、噪声输入均不同。训练World还看到了由GT混合构成的noisy输入，不能把0.004823当作从纯先验出发的最终ODE轨迹L1。上面的物理L1是从最终请求逐点重算，没有将一个归一化scalar乘某个任意尺度。

源码入口：`starVLA/model/modules/qwen_world_postcot/flow.py:clean_x_l1`；`starVLA/model/modules/qwen_sourceflow/source_flow_policy.py:forward`；`starVLA/model/modules/action_model/GR00T_ActionHeader.py`的FM loss。聚合训练窗口与输入记录hash见`native_training_objectives.json`。

## 完整性与时间戳敏感性

原始Navtest records及manifest均通过封存hash；所有12,146个原DDP processed GT与原始Scene GT独立坐标变换逐场景核对，最大坐标差1.904976443e-06（float32舍入量级）。五组预测完整人口、token/log归属、请求/评分请求及各自hash已核对；能直接重读的PDMS与原封存summary误差小于1e-8。S0重算ADE/FDE/角度与其既有ego-fit summary在1e-5以内一致。

36场景存在真实日志未来时间戳间隔异常。主表保留全部12,146场景、采用同一官方名义帧GT，不静默删除；另在统一的12,110个各点累计时间偏差<0.06s场景复算如下，排序不变。这不是新推理或评分协议。

| 模型 | 完整人口位置L1 m | 时间戳正常人口位置L1 m |
|---|---:|---:|
| 当前Qwen-World IL53k | 0.364448 | 0.361885 |
| Qwen-World IL40k | 0.361566 | 0.359117 |
| 此前原生Qwen-Flow IL98k | 0.413626 | 0.410752 |
| 原DDP F-SFT100k | 0.513426 | 0.510735 |
| DDP＋DINO S0 100k | 0.491931 | 0.489321 |

六项小型算术验收：相同轨迹为0、坐标平均与ADE分开、角度±π环绕、拒绝8×4物理轨迹、拒绝NaN、后轴坐标旋转，见`qualification.json`。原生训练与部署源码均未修改。

## 资产与执行

输入路径/版本见`inputs.json`，聚合结果见`analysis.json`及`comparison.csv`，配对统计见`paired_L1.json`。逐场景结果仅留本地：
`/mnt/project/qwen-world-postcot-20261025/cross_il_trajectory_loss_20261028/analysis/*_scenes.csv`。不上传原始场景、权重、缓存或私有轨迹。

实际CPU执行的比较代码与仓库`tools/qwen_world_postcot/compare_saved_il_trajectories.py`字节相同，code hash见analysis；实际命令：

```bash
PY=/mnt/project/flow-safe-improvement-20261013/venv/bin/python
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" /mnt/project/qwen-world-postcot-20261025/cross_il_trajectory_loss_20261028/analyze_saved_trajectories.py --plan /mnt/project/qwen-world-postcot-20261025/cross_il_trajectory_loss_20261028/plan.json --output /mnt/project/qwen-world-postcot-20261025/cross_il_trajectory_loss_20261028/analysis --workers 4
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" -m tools.qwen_world_postcot.collect_native_il_objectives --output /mnt/project/qwen-world-postcot-20261025/cross_il_trajectory_loss_20261028/native_objectives_audited.json
```

输出目录/文件采用exclusive创建；已完成结果不可直接覆盖。重放时换一个新output路径，仓库模块入口为`python -m tools.qwen_world_postcot.compare_saved_il_trajectories --plan reports/cross_il_trajectory_loss_20261028/inputs.json --output <新的本地目录> --workers 4`。两个CLI的`--help`与语法检查均已实际通过。

共复用60,730条既有单候选请求，新增模型前向0、scorer调用0、optimizer更新0。五模型核验/比较计算阶段wall合计336.06s，另有原始GT核对、日志读取和启动成本；不冒称独占GPU时间或完整CPU核时。
