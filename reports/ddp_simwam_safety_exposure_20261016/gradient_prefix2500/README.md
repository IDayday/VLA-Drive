# 固定安全曝光：保存链的分层梯度复算

2026-10-16。完成X1与SAFETY25各1000、2000、2500更新后的三个固定批次复算，使用已有48个scene groups / 384条候选记录，**没有新采样、评分或optimizer更新**。两个正式训练的配方、20k终点和评价队列未改变。

结果支持固定安全失败场景能提供可见的PG方向，并没有被更大的global std全部消除。在本次三个SAFETY25批次中，仅两个SFT失败场景的梯度在整批PG方向上的投影系数为0.970、0.971、0.712。X1恰好含两个失败场景的2500批次也为0.966，因此这不是新优化器才具备的能力。与前一份[完整2500窗口的信号统计](../early_training2500/README.md)结合，更直接的解释是改变了安全排序信号的曝光频率。

**这些梯度不能证明单候选ODE提升，也不能据此削弱anchor或放宽clip。** 它们是每份rollout已经完成四次更新后，在保存状态上重新求得的局部loss梯度；不是当时四次Adam实际位移，更不是整个训练集的梯度占比估计。

## 范围与数学口径

预先固定[registration.json](registration.json)的六项工作，不按结果选择场景或扩大样本。每个检查点使用其紧邻的最后一个完整collection，保留八个rank各自完整G8、保存链、old probability、advantage和mask。两个变体的人口和场景顺序不同，不称为同场景配对效果比较。

每rank计算三个随机步骤的原PG及beta50均值anchor梯度，各自保留源`1/8`全rank分母；tied rank的PG仍为0，anchor保留。最终先对梯度向量取步骤均值，再计算norm/cosine，不能以三个norm的平均替代总梯度norm。

固定SFT联合安全/失败标签只用于分析分层，不是actor输入。`SFT_unsafe`不意味着该场景本批所有候选都不安全。两层贡献为各自rank梯度之和除以8，没有将有信号或失败rank重新归一化成独占整批权重。

表中“失败层投影系数”为：

`dot(g_SFT_unsafe, g_PG_total) / ||g_PG_total||²`。

两层系数相加为1，但在相互抵消时可以小于0或大于1；它不是因果安全收益百分比。原PG梯度使用源relative-clamped、dimension-mean转移代理，不是完整混合生成链的无偏概率梯度。

## 实际结果

| 变体 / 更新 | SFT安全/失败组 | raw global std | PG norm | beta50 anchor norm | PG/anchor cosine | 失败层投影系数 |
|---|---:|---:|---:|---:|---:|---:|
| X1 / 1000 | 8 / 0 | 0.0357 | 0.011767 | 0.006227 | −0.272 | 0.000 |
| X1 / 2000 | 8 / 0 | 0.0335 | 0.012464 | 0.006940 | −0.152 | 0.000 |
| X1 / 2500 | 6 / 2 | 0.2249 | 0.019058 | 0.006364 | −0.167 | 0.966 |
| SAFETY25 / 1000 | 6 / 2 | 0.2506 | 0.019743 | 0.007836 | +0.046 | 0.970 |
| SAFETY25 / 2000 | 6 / 2 | 0.2383 | 0.021612 | 0.010113 | +0.109 | 0.971 |
| SAFETY25 / 2500 | 6 / 2 | 0.3281 | 0.022439 | 0.007144 | −0.042 | 0.712 |

第一份X1全是SFT安全场景不是事后选择：检查点和collection在复算前固定。两份X1批次没有失败层，因此其投影系数为0属于人口事实，不是方差网络或梯度断开。

SAFETY25三个批次中的失败场景，其当前G8联合安全候选比例分别为37.5%/37.5%、87.5%/50%、62.5%/37.5%。这些组同时存在安全和失败候选，可提供组内排序。它们是有限候选机会，不能当成部署恢复率。

在这些批次中，beta50 anchor总norm均小于PG总norm；但末步anchor、各方向和跨步骤抵消不能用单一大小关系概括。SAFETY25总PG/anchor cosine有正有负，证据不支持“安全训练没有梯度”或“anchor在所有batch都压制PG”。同时，不排除其他批次、训练阶段或实际Adam预条件下存在冲突。

## 随机步骤差异

| 变体 / 更新 | step7 PG / anchor norm | step8 PG / anchor norm | step9 PG / anchor norm | step9 PPO clip比例 |
|---|---|---|---|---:|
| X1 / 1000 | .006191 / .000844 | .013406 / .002261 | .032045 / .016412 | 6.250% |
| X1 / 2000 | .011444 / .001404 | .018334 / .003185 | .030634 / .018199 | 29.688% |
| X1 / 2500 | .023830 / .000828 | .030505 / .002333 | .057853 / .016663 | 17.188% |
| SAFETY25 / 1000 | .025666 / .000876 | .049821 / .002814 | .021074 / .020978 | 25.000% |
| SAFETY25 / 2000 | .019731 / .000931 | .042682 / .003130 | .037864 / .027869 | 14.063% |
| SAFETY25 / 2500 | .029132 / .001199 | .039364 / .003464 | .051791 / .018816 | 28.125% |

末步anchor在六个批次中均明显大于较早步骤，但PG也随步骤变化，且step9的PG/anchor cosine在SAFETY25为−0.217、+0.222、+0.078，不能由末步norm较大推断始终对抗PG。本轮不修改非均匀dt、时间网格、锚定权重或最后一步行为。

这里的ratio是更新四次后的current对旧behavior，clipping可能是实际更新结果；它与此前首inner未更新时ratio误差为0的检查不是一回事。raw Gaussian与source clamped代理分开保留，不能将代理裁剪结果写成精确整链KL。

## 源loss与只读性验证

复用[diagnose_saved_batch.py](../../../scripts/analysis/simwam_recipe/diagnose_saved_batch.py)，增加可选登记接口及[saved_gradient_strata.py](../../../scripts/analysis/simwam_recipe/saved_gradient_strata.py)。不修改actor、sampler、loss、optimizer或部署代码。

每个rank用原`actor.forward / _policy_loss`独立重算loss和完整梯度，对照分解后的三步PG+anchor；六个最终批次全部通过。最大逐rank梯度绝对差约7.40e−10，分层再合并与原loss总梯度最大绝对差约8.49e−10。192个LoRA参数与载入值逐项相同，没有optimizer step或残留`.grad`。

每个分析job在单卡顺序读取八个原rank的batch，并显式保留原rank平均语义；这是保存数据上的单卡梯度复算，不冒称本次又做了八卡训练验收。原正式八卡更新/恢复验证仍沿用既有证据。

CPU独立autograd算例覆盖0/2/8个失败组、tied rank保留anchor、空分层、抵消时负或大于1的投影、非有限拒绝。见[cpu_qualification.json](cpu_qualification.json)及[更新后记录](cpu_qualification_v2.json)。

来源检查初次因X1旧`collection.py`文件hash与新文件不同而停止，尚未发GPU工作。实际差异仅是采样窗口断言由字面量变为注册映射；本次不调用collector，使用的`exploration_summary` AST相同。原`SceneStream`、`rollout_digest`也相同，数值actor/model/概率/loss依赖逐文件hash相同。[明确兼容记录](source_compatibility.json)保留差异及适用范围，没有删除源身份检查。

## 三次失败与修正

首次六项工作中三项完成，三项因`train_reward_mean`核验失败退出。保存的原始FP32 reward、逐候选readout、优势及链摘要未改变；原因是对同样八个值做CPU/GPU FP32归约时相差1.1920928955078125e−7，超过旧只读检查的1e−7容差。

用全部48个保存组执行[最小GPU复现](reward_mean_reproduction.json)：48/48的原CUDA FP32均值与日志逐值相同，仅三个CPU重算值超旧容差。因此这是只读统计核验误报，不能当成正式训练reward错误。

修正为GPU诊断显式使用原CUDA归约，并要求该字段与日志**完全相同**；不放宽容差、不改数据。普通CPU调用保留原保守检查，遇到同类数值边界仍可能拒绝；本补丁没有将GPU执行静默加入CPU工具。

失败目录原样保留；[补充登记](retry_registration.json)只允许重跑原三个失败job，并额外复算一个原已成功job以验证诊断修改的等价性。该批次修正前后所有逐步骤结果、PG/anchor梯度、LoRA有效更新norm、概率clamp结果和候选类别完全相同。原始源码`35b54d3`和修正源码`c84a2d0`分别保存，不热改已有快照。

另有四次SSH命令误用了不存在的20261016环境路径，在worker启动前退出127，随后改为既有20261013环境。[记录](dispatch_failure.json)明确它们没有模型加载、采样或更新；未独立采集的SSH wall不伪造成0。

## 成本与入口

最终覆盖六个独立保存批次、48组、384条记录。实际启动10个模型分析worker：7个成功（含一次重复等价性检查）、3个失败。共享worker job wall累计394.16秒，其中失败86.96秒；成功的worker主体累计291.69秒。首次开始至最终完成跨度397.73秒，包含中间诊断/源码准备，不是纯GPU计算耗时。

成功invocation合计current/reference速度前向各336次；失败进程的部分前向计数未独立保存，标为未知。峰值tensor allocated约6.88GiB/worker，不代表驱动总占用。另有最小GPU均值复现主体4.46秒，导入与SSH时间未完整计入。全部工作0新候选、0新评分、0optimizer更新。[机器可读结果、逐次身份及成本](results.json)

CPU入口已执行：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m scripts.analysis.simwam_recipe.saved_gradient_strata --output "$NEW_CPU_OUTPUT"
```

真实GPU入口已在授权Tsinghua zt执行；重验必须选登记内run/checkpoint/collection，使用新输出路径：

```bash
python -m scripts.analysis.simwam_recipe.diagnose_saved_batch \
  --run "$REGISTERED_RUN" --checkpoint "$REGISTERED_CHECKPOINT" \
  --collection "$REGISTERED_COLLECTION" \
  --gradient-strata-registration /mnt/project/simwam-safety-gradient-20261016/registration.json \
  --output "$NEW_GRADIENT_OUTPUT"
```

当前决策仍是完成SAFETY25的固定ODE观察点及20k终点，判断持续安全信号是否转化成NC/DAC/TTC保持及PDMS收益。本次不加大LR、不改变global std、不放宽clip或削弱anchor，也不新增训练组。当前最好完整Navtest仍为X1原ODE的89.3817，不能用这份局部梯度证据宣称92已实现。
