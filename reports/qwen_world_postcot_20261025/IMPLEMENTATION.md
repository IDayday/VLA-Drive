# 完整 Qwen-Flow 表征与行动评价框架

起点为 `425eb5420ce1b592786f455ceea9bd6d124d2c54`，独立分支
`feature/qwen-flow-world-postcot-500m-20261025`。用户参考 `6fbdd59` 是其祖先，
两者间是报告更新。旧正式 IL 实际运行源码是 `d4d0012`，旧正式 RL 是
`434cba8`；这些运行工作区、checkpoint 和任务均未热修改。

本轮训练代码已提交并封存为 `d54617b3ff02d714c067ce809a50f50cd19bceb6`，
正式源码为 `/mnt/project/qwen-world-postcot-20261025/snapshots/formal_main_v2`。
其 `SOURCE_BINDING.json` 和外部工程验收逐文件绑定50个代码／配置文件。
后续报告提交不改变该实际训练源码身份。

## 已实现的计算路径

当前原生三相机、导航和8维ego → 官方Qwen3-VL-2B → 第8/16/28层与同次
visual merger → 144空间+16全局memory → 32预测memory → 20层clean-x动作头。
Qwen输入不增加token，部署输出原生8×3轨迹，0.5–4秒、rear-axle、弧度。

动作头为1024宽、16头、4096 FFN，每层self/cross attention和6d AdaLN。
动作本体实际467,119,107参数，不是估算500M。每层192-token K/V一次投影，
主动作和精确候选重放共享有梯度context，部署跨K10复用。

训练同时执行五项任务：clean-x轨迹L1、当前DINO密集MSE、当前对象监督、
WA-JEPA未来特征MSE、行动分项评价。主权重10/1/0.1/0.5/0.2；后置任务
为factor BCE+0.25同场景NC/DAC边界pair loss，软标签保留。

后置分支将缓存的**精确物理候选**编码到sigma=0，共享actor trunk和context，
不执行ODE、不经过最终action projection、不将评分最大化当作主策略loss。
坏候选不进入轨迹模仿目标。训练forward的当前Qwen只调用一次。

部署删除DINO/future/object/post readouts，保留学生predictive memory；不加载
教师、候选库、GT或真实评分。clean-x损失、K10 ODE以及RL正sigma处的
一次x→velocity转换分别实现，sigma=0 encode-only不作速度转换。

## 正式训练定义

阶段一是“轨迹模仿学习＋表征辅助监督＋行动评价辅助监督”：
`uses_metric_supervision=true, uses_rl=false`，不是完全没有PDM监督的纯IL。
global32，默认micro4×8卡，必要时micro2×accum2；100k更新包含1000接口
warm start，原100k cosine时间轴不缩短。接口阶段Qwen LR=0，联合阶段
仅初始化刚解冻Qwen的Adam状态，保留其余历史和全局scheduler。

阶段二复用真实canonical PDMS Flow-GRPO，G8/K10/inner4、attention LoRA16、
永久IL reference、50k更新。Qwen和全部学生memory冻结，教师及辅助readouts
不运行。正式RL等待完整IL和DEV选择；不要求各辅助模块先独立证明涨分。

## 数据边界和已修复问题

旧TRAIN/DEV目录不重叠，但共享15个物理日志。本轮固定DEV1696和NAVTEST12146，
在TRAIN中预先排除这些物理日志：90,890场景、147物理日志；DEV15物理日志；
NAVTEST44物理日志。RL原hard-list交集15,171场景、90物理日志。
旧Base可能见过DEV同源日志；公开WA预训练也可能覆盖内部DEV，不能称完全未见。

625个TRAIN场景有真实时间gap，不把错时刻pose冒充固定0.5秒waypoint：
轨迹有效mask逐点检查实际累计时间，其他监督保留。教师12帧缺失或非2Hz时
仅屏蔽相应future目标，不捏造有效零特征。

真实生产另发现project图像镜像缺失/损坏，而canonical挂载完整。当前DINO
严格用与Qwen相同的row.images；未来通过raw log、token、offset解析互补的
trainval_all与trainval_sensor_blobs两处物理图像库，核验真实时间及实际文件SHA。
单独使用其中一处会错误屏蔽合法未来监督，因此旧v2仅留作失败/性能诊断，正式
配置只接纳resolved_v3。旧失败收据和partial cache保留，不修改用户镜像或数据清单。

## 依据与复用

- [AUGUstus](https://arxiv.org/html/2610.16985v1)：行动生成与后置理解训练的参考；
  不将增加一个post head本身作为已验证新颖性。
- [DrivoR](https://arxiv.org/html/2601.05083v2)：行动评分参考；本轮不加载其权重，
  不使用其多候选部署选择代替单轨主结果。
- [WA-JEPA官方实现](https://github.com/AFARI-Research/WA-JEPA)、
  [公开权重](https://huggingface.co/AFARI-Research/WA-JEPA)：严格加载已训练target
  encoder和projector，版本与真实测试见TEACHER_AUDIT.md。
- [DINOv3](https://github.com/facebookresearch/dinov3)、
  [SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)、
  [ZeRO](https://www.deepspeed.ai/tutorials/zero/)：对应特征、attention和分片实现。

工程验收、正在运行的生产任务与科学结果分别记录；目前不得据代码接通宣称PDMS改善。
