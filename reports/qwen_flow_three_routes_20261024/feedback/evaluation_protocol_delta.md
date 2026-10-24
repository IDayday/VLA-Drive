# 学生辅助读出与统一主评价的边界

2026-10-24审阅发现，immutable `97a66c10743651767ab11c0ec8bc6cbd29619430` 内
`feedback.student.evaluate_student` 使用 `three_route_ODE_eval` 的scene-bound
noise namespace，并一次建立五个root，由公共runner以chunk4生成。此前统一的原生部署使用
`native_qwen_sourceflow_ODE` namespace、每个推理seed单独batch1。
二者即使都标为seed2101，也不是相同root或相同模型batch布局。

正在运行的S1训练不热修改、不重启。immutable97附带的自动读出保留为**辅助评价**，
其轨迹、评分、推理与评分费用全部保留，不能重标或复用于主配对。该差异不进入FM训练、
teacher筛选、优化器或模型权重；发现时还没有产生新学生正式效果结论。

可变开发版本只将helper默认namespace和生成布局与原生部署对齐，补充协议、评分器、
sampler、精度、seed、人口和batch身份。仍然不允许把旧缓存按新身份读取。
三组学生的正式主比较统一交由公共plain merged adapter evaluator执行：
不构造teacher/reference、不使用存档或评分器选轨，固定每seed单候选shift5十步ODE。
helper与该正式评价的文件schema不同，不因root对齐就自动合并两份结果。

本修正是评价工程合同修正，不改变已登记S1/S2/S3训练配方，也不据新结果调整超参数。
