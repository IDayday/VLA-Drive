# 本轮验证与证据索引

本表按研究请求中的25项约束对应实际代码和执行证据，不把20个单元测试称为25项完整链路验证，也不把调试权重当正式终点。`evidence/TESTS_ROUND1_FINAL.log`记录本轮相关20项测试通过；原失败和修正记录保留。正式R_FUTURE_DIFF seed42的最后在线验证仍在等待其共同24epoch终点。

| 请求项 | 实际验证及边界 |
|---|---|
| 1. 原S0行为不变 | 原生100k checkpoint及来源核验；真实样本缓存q0与原生predict逐位比较；缓存续接每worker比较既有样本，S0 buffers不变。`FULL_POPULATION_RECEIPT.json`、`CACHE_REDISTRIBUTION_CHECK.json`、`CAMERA_ONLY_KINV2.json` |
| 2–3. 标签隔离 | `ModelInputs.from_mapping`拒绝多余GT字段；`ResidualRefiner.camera_only`拒绝`OracleRelations`；真实图像入口文件访问guard禁止地图、框、未来日志、ego GT与feature cache；单独oracle入口。Python guard不是OS沙箱 |
| 4. q0/扰动不读GT未来 | 原生scene-bound10步proposal；`trajectory_queries`只收q0/当前速度/config；完整查询bank的`contains_GT=false`；真实在线不打开GT标签 |
| 5–7. 路径法向、时间域与停车 | 6/8步测试、斜向路径法向、固定端点且严格单调warp、真实环境时间不warp、停车heading、heading_transition_v2合法性；改变validity版本不改变pose字节 |
| 8. 完整车身/后轴 | body_points前后轴极值及90度旋转测试；Shapely直线、拐角、孔洞、旋转与跨边界误差。`GEOMETRY_APPROXIMATION.json`是有限样例误差，不是任意地图上界 |
| 9–10. 当前/CV/未来同坐标、合法速度 | CV原点旋转与平移测试：当前lidar速度旋转到ego后按实际时间平移；缺速度零速回退；无未来frame输入，附加虚假未来字段不改变结果；输入框与速度不被原地修改。真实标签统计保留fallback计数 |
| 11. 距离/unknown | SDF符号、空障碍、裁剪、地图外未知、车身跨边界和旋转测试；静态地面支持延续不覆盖未知实例/缺帧，原严格版本保留 |
| 12–13. 配对有效性/交换/相同查询 | 两端有效性取交集；缺anchor时差异loss为0，含NaN标签不污染梯度；交换两端差值反号、SmoothL1误差保持；相同几何无候选ID差异 |
| 14. 模糊区 | 物理阈值在训练侧冻结：<0.5米位移模糊区、道路±0.25米带、障碍0/1米带；评价保存各组支持数和模糊数，不依据PDMS设阈值 |
| 15–16. ABS/DIFF匹配 | 参数初始化hash、数据/查询/曝光/绝对loss相同；六个anchor-variant对均值归约，复制相同pair不机械放大loss。`ACTUAL_FORMAL_COMMON_INITIALIZATION.json`及完整训练identity |
| 17–18. E/R对应输入独立训练 | E五种实际输入各自训练；R从首步读取冻结的对应P预测，checkpoint记录predictor SHA；不以GT训练后在部署换成预测。`HEAD_TASKS.json`与实际train.jsonl/COMPLETE |
| 19. 冻结/eval | 原模型全部参数requires_grad=false/eval；缓存生产前后buffers逐位比较；小头优化器不含S0/P参数。固定Qwen视觉/语言与proposal真实模块调用次数另实测 |
| 20. cache/online | 调试与正式已完成模型逐项核对W/H_A、q0、查询和q_final；正式E_NONE补充执行hook的4场景q0/q_final差异为0；FP16与不量化读出差异单列，不称其逐位无损 |
| 21. 单一S0 | 原S0执行路径直接调用visual和language子模块，因此旧outer hook断言失败已保存；修正为实际子模块hook后，4场景均为visual=1、language=1、proposal=1；量化诊断的额外前向独立计数且不参与选轨 |
| 22. 无GT选轨 | 部署仅一次连续残差输出；七条query不评分选优；主表固定q_final；q0独立诊断，不按官方分数择优 |
| 23. 原生恢复 | E_NONE短链路逐位相等；P_DIFF原严格失败保留，参数/moments/输出与连续A/A误差并列；RNG/数据位置/顺序/曝光/scheduler精确。真实RAM续训在optimizer边界、原源码严格恢复，日志update连续 |
| 24. 真数据前后向 | 真实64场景全部输入模式可学习性、4096校准/读取对照、完整101592小头训练；原图通过冻结模型真实编码，小头反向在相同紧凑表示上执行，未声称联合反传到冻结S0 |
| 25. 结果分表 | `summarize_round1.py`校验oracle/camera模式、同q0、同场景及canonical版本后分别输出；历史G只读、特权E、camera-only R和后续从头计划分开 |

完整正式部署还核验原生模型恢复、当前输入身份、TF32关闭和标签文件不可访问。派生场景地图只保留在共享受控artifact目录，Git只收聚合结果、数值统计和案例选择元数据。

数值边界：FP16紧凑W/H_A相对L2约2e-4，正式E_NONE四场景与不量化读出的最大最终坐标差约6.01e-5米；它不是所有场景规划分数等效的证明。canonical导出使用与在线注册入口相同的FP16往返，头部FP32；冻结q0保持原FP32。GPU恢复存在原生浮点累加差异，未反复抬高阈值掩盖旧失败。
