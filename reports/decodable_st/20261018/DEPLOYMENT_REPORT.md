# 去教师部署与成本

两个实际版本均在新进程验证：

| 导出 | 训练来源 | 独立当前输入误差 | 说明 |
|---|---|---:|---|
| engineering_export_v2 | 表征2步 + reader2步 | max abs 0 | 完整student→reader→原DiT链；未证明能力 |
| representation_export1000_v1 | RS表征1000步 | max abs 0 | reader关闭，供遗忘/表征诊断 |

进程禁用teacher imports、教师权重/cache读取，HF离线；加载teacher模块数0。固定真实当前三图、导航、合法ego及相同noise。未使用GT map/框/LiDAR/未来位姿。原训练辅助head剥离后保留W、位置、Qwen、LoRA、新queries/readout/reader和动作模块。

第二份导出复用同一12GB只读基础bytes的hardlink，仅增加训练delta，避免再复制大模型。基础SHA256 `4b32b9bc95d09fc91383b63b38f09eba8dfeded3d5e3e45f4045ebfe19bb9c3a`；overlay的SHA在RESULTS及私有manifest。恢复时strict delta inventory与基础hash都校验。导出为可信本地、锁定Torch/Transformers版本的research artifact，不能当不可信pickle加载器。

第一次导出因Transformers用于训练的局部input-gradient hook不可序列化失败。修复仅移除该明确hook，原失败日志保留；未删除任意forward hook，也未关闭教师拒绝检查。

同A800、同真实场景、FP32、TF32关闭、单候选Euler10，交替6个窗口、首轮舍弃：

| 端到端预测 | 5个计时窗口中位 |
|---|---:|
| 原S0 | 1.04735秒 |
| 空间query/readout/reader工程模型 | 1.11291秒 |
| 相对开销 | +6.26% |

独立1000步无reader导出中位约1.11192秒。这个协议只有一个场景，不是跨场景P90；没有把BF16速度和FP32分数混用。该无reader诊断路径仍计算空间表征，不能把它解释成原S0最快执行。

独立进程只验证一个固定当前输入；完整部署人口、正式P3权重、时间学生与跨环境portable导出尚未完成。PASS范围是已运行的teacher-free执行与数值一致性，不能替代teacher/representation/planning PASS。
