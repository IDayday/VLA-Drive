# 数据、时间与坐标

实际共同训练 101592 场景/1176 logs；开发 1696 场景/16 logs。log 与 token 交集均为 0。64 场景审计各取不同训练 log，选择在结果之前固定；容量实验使用其中 48/16 个训练 logs，学生 tiny-fit 使用全部 64。没有拿 Navtest 拟合统计、阈值、压缩器或损失权重。

统一变换记法 T_A_from_B，列向量。未来标签变换为 inverse(T_W_from_E0) @ T_W_from_Etau @ T_Etau_from_Ltau。真实时间戳用于匹配，试验性 1.0 秒请求在这 64 场景的最大误差 0.002742 秒。这是明确的新适配任务候选，并非已经核验的 SGDrive native horizon。

实际读取了二进制 PCD；原 text 文件读法失败后改用既有 LidarPointCloud.from_buffer。相应失败日志保留。原始合并点云含五个 sensor_id，坐标 lidar2ego 为 identity，但这不意味着五个发射器具有相同物理原点。现有 NAVSIM 资产中未核实每传感器原点、首回波/去畸变来源；原发布 occupancy 路径在这批样本也不存在。因此仅保存 hit-positive 证据，无回波、未覆盖、遮挡后方均为 unknown，没有伪造 observed-free。

静止世界点的正反变换代数误差 <1e-8m，证明公式/矩阵约定正确；这不是完整真实点云配准质量验证。LiDAR 高度与 ViGT native 地面/相机原点之间的变换仍缺独立证明，不能在看到教师结果后加一个经验 z 偏移就宣布正确。

当前能够继续的是原生教师一致性、压缩损失及教师协议下的小规模迁移诊断。独立空间能力、时间变化能力和动态 persistence 优势尚不具备验收证据。修复需要有来源的传感器外参与回波语义，或可靠的独立 occupancy 数据产品；不需要再增加一个教师模型。
