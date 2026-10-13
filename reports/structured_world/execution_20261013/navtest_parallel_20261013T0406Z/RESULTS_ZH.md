# G1 80k、G2 50k完整Navtest与并行调度（2026-10-13）

新增两组完整结果：**G1 80k PDMS 89.1511；G2 50k PDMS 88.3943**。每组均为12,146个唯一场景、136个log，q0与q_final全部评分成功。预测成员、载荷hash、CSV、汇总及队列完成身份已核验。主结果始终采用q_final。

| 权重 | q0 PDMS | q_final PDMS | q_final零分场景 | 评分失败 |
|---|---:|---:|---:|---:|
| G1 80k | 88.8408 | **89.1511** | 549 | 0 |
| G2 50k | 87.1948 | **88.3943** | 635 | 0 |

| 主结果分项，0–100 | G1 80k | G2 50k |
|---|---:|---:|
| NC | 98.0981 | 98.0652 |
| DAC | 97.2748 | 96.5915 |
| EP | 83.2357 | 82.5461 |
| TTC | 95.0436 | 94.8213 |
| 舒适度 | 100.0000 | 100.0000 |
| DDC | 98.4522 | 98.4028 |

## 学习变化与50k共同观察点

G1 80k较70k提高0.1437分，按log聚类、保留真实场景权重的20,000次bootstrap 95%区间为[-0.0772, 0.3713]；较60k低0.1312分，区间[-0.5190, 0.2324]。当前更接近平台波动，尚未超过60k的89.2823。同模型修正提高0.3102分，区间[0.1754, 0.4608]，修复51个q0零分场景、新增15个。

G2 50k较45k的89.3858下降**0.9915分**，区间[-1.3393, -0.6350]。零分场景508→635：修复86个旧零分，同时引入213个新零分。NC下降0.3746、DAC下降0.6998、TTC下降0.4775、EP下降0.9410，各自区间均低于零；退化涉及道路、碰撞和进度。

G2的q0也从45k的88.4864下降到87.1948。50k修正仍带来+1.1994分，区间[0.8630, 1.5555]，修复199个q0零分、新增60个。因此跌幅已出现在原始提案，修正有所缓解，但未补回整体退化。这不能仅凭规划分数定位为某个优化器、表征或损失的单一原因。

| 50k共同观察点 | q0 PDMS | q_final PDMS | q_final零分 |
|---|---:|---:|---:|
| G0_NO_FUTURE_LABEL | 87.3220 | 88.8899 | 588 |
| G1_FULL_UNIFORM | 87.4397 | 88.9409 | 567 |
| G2_EVENT_UNIFORM | 87.1948 | 88.3943 | 635 |
| G3_EVENT_LOCAL | 86.6842 | 88.5195 | 625 |

| 对照，q_final | PDMS差值 | log聚类95%区间 | 本观察点支持的判断 |
|---|---:|---:|---|
| G1−G0 | +0.0510 | [-0.3166, 0.4168] | 尚无未来标签增加规划收益的明确证据 |
| G2−G1 | −0.5467 | [-0.9095, -0.1996] | 本seed的50k事件全局监督低于完整未来监督 |
| G3−G2 | +0.1253 | [-0.2865, 0.5114] | 局部采样优势尚不明确 |

G2 45k仍是目前已完成观察点中分数最高的权重；不能用它与其他组不同步长的最佳值替代上表。本轮只有一个训练seed，场景bootstrap不能代替训练复跑，50k也不是100k共同终点。用户要求中间观察点直接测Navtest，因此这些结果不再来自完全未使用的最终留出集。本次未停止或改变训练，后续仍执行登记的观察计划；不能依据这一次退化自动宣布整个方向无效。

算法表述保持准确：G3改变road、current occupancy、future semantics和body relations共用的结构化监督查询分布。语义监督塑造Bt，FGTR读取Bt latent；部署不将预测道路/占用数值作为硬约束。q0来自原10步采样并detach，修正梯度仍经H_A/Bt/FGTR回传。此次调度没有修改这些定义。

## 实际并行与耗时

两组正式推理均采用以下动态核验出的26张独立物理GPU；卡数是本次资源快照，不是skill固定配额。

| 容器 | GPU数量 | 分配 |
|---|---:|---|
| training-vlawm-tsinghua-zt | 8 | GPU0–7 |
| training-vlawm-tsinghua-zt2 | 7 | GPU1–7；GPU0保留真实任务 |
| training-vlawm-zt2 | 3 | GPU3、4、7，与自身G1训练共享 |
| training-vlawm-zt4 | 8 | GPU0–7，与自身G2训练共享 |
| training-rl-zt2 | CPU | 32个canonical评分进程，与GPU推理同步 |

推理占比1.0、主动限速等待0秒，采用小块动态领取任务，快卡承担更多场景。没有停止真实训练。vla-zt/vla-zt2已有其他任务保留；隐藏或无法准确归属的GPU占用未被当作压力脚本清理；旧9小时临时授权未延长。具体排除证据见RESOURCE_SUMMARY.json。

G2原生权重准备在04:11 UTC启动，实际与G1全量推理重叠。其原生FP32恢复、导出和冻结原始入口对照耗时较长，04:22通过首次核验，04:26进入全量推理。**两组全量GPU波次没有同时运行**；实际重叠的是G2准备与G1推理/收尾，不能把这一点报告成两个完整波次同时推理。

| 已执行成本 | G1 80k | G2 50k |
|---|---:|---:|
| 正式分片注册至完整评分校验，分钟 | 17.637 | 17.557 |
| 工作进程推理时间累计，GPU-hours | 5.726 | 5.670 |
| 工作进程驻留时间累计，含加载，GPU-hours | 6.139 | 6.137 |
| 单进程峰值allocated，GiB | 12.548 | 12.548 |
| 单进程峰值reserved，GiB | 13.078 | 13.078 |

这些成本不包含此前的原生恢复/profile准备，准备时间另见G2_PREPARATION.json；共享GPU的进程驻留时间不是独占设备账单，也不能与训练GPU-hours简单相加。不据此宣称经过隔离A/B验证的固定加速倍数。

G2满速推理期间，G1/G2最近32步中位时间约7.718/9.348秒；结束后恢复约3.168/3.217秒，G0约3.114秒。04:43 UTC训练分别到G0 58,743、G1 81,088、G2 50,740步，近期FP32 master更新验证通过；G3已完成100,000步。相关原始聚合见TRAINING_DURING_G2.json和TRAINING_AFTER_G2.json。

## 调度修改、验证与存储

`run_navsim_navtest.py`新增严格登记推理进程识别及并行入口；`navtest_parallel_controller.py`以一个队列锁管理多个独立权重任务。每组独立保存场景身份、worker和评分输出。GPU在CPU加载阶段预留容量，完整执行一场景后才交由实际显存读数进行后续共享准入；CPU预算扣除其他活动评分池。

04:19仅交接CPU队列控制器，G1既有26个GPU worker和原评分器继续运行，未发送信号给它们或训练进程。新控制器核验了完整worker报告、远端PID/start退出和全部结果。由于父进程交接，旧父进程wait退出码不可恢复；这一局限在收尾记录中保留，没有伪造为重新wait成功。

完整结果结束后，04:44在空闲边界切换到含加载预留/联合CPU预算的新控制器，PID 507036，未热改GPU源码。它继续按当前授权资源处理50k后每10k的登记请求，无新请求时保持空闲。保留默认串行入口作回退；恢复时必须确认现有唯一CPU控制器已退出，不能启动竞争队列。

86项CPU检查通过，包括进程身份绑定、容量预留、评分预算、cadence、恢复/扩容和压力监控。首次测试使用默认conda解释器时缺少pidfd API，产生CPU fixture测试错误；改用系统`/usr/bin/python3`后通过，原问题记录在SOURCE_VERSIONS.json，未改训练环境。四台主机各用同一组4个真实场景与原始冻结入口核验，q0/q_final/out_of_range逐位一致；这不是另外重复一次全量原始入口推理。

G1临时native pin的17个硬链接和临时FP32导出已释放；G2临时FP32导出释放12.487 GiB，原50k训练检查点保留。压力监控仍按600秒确认空闲规则运行：归档时Tsinghua-zt正在计时，Tsinghua-zt2因真实任务继续让出资源，未谎报恢复完成。代码、skill和产物均在`/mnt/project`，仅Python环境可保留容器本地安装。

## 身份与可执行核验

- 训练及原始模型/推理源码：`0c54f1c49e7ed2a76b22826b654d6c72625f5b6d`，冻结worktree干净。
- 本轮并行调度与G2实际worker来源：`a56632e9da14d8f811b55484b185f696f06f8793`。
- 后续调度的容量预留/CPU预算、项目skill：`5366c82776dc807c2c39c80850f1636c94cda06e`。
- G1 worker保持前一冻结路由版本；具体每个worker及两版控制器文件hash分别保存在WORKER_AGGREGATES和SOURCE_VERSIONS文件。
- FP32 master严格恢复、FP32推理、TF32关闭、原10步FM、单候选、scene绑定seed42；场景和canonical评价未变。
- skill位于`/mnt/project/skill/navtest-parallel-eval/SKILL.md`，仓库内版本在`skills/navtest-parallel-eval/`，格式校验通过。

在training-rl-zt2可以复核已有结果，不加载模型、不派发GPU：

```bash
/root/miniconda3/envs/navsim/bin/python -B /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_parallel_20261013T0406Z/frozen_runtime/run_navsim_navtest.py --verify-complete --request-key G1_FULL_UNIFORM_0080000 --output /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225
/root/miniconda3/envs/navsim/bin/python -B /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_parallel_20261013T0406Z/frozen_runtime/run_navsim_navtest.py --verify-complete --request-key G2_EVENT_UNIFORM_0050000 --output /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225
```

完整当前控制器命令见CONTROLLER_RESERVED_LAUNCH.json；分析、评分和验证实际命令分别见对应`*_COMMAND.json`和`*_SCORER_LAUNCH.json`。LEARNING_CURVES.csv收录17个已完成检查点、两套轨迹的全部聚合分项。只上传代码及汇总数据，没有上传权重、缓存、私有图片、逐场景记录或密钥。
