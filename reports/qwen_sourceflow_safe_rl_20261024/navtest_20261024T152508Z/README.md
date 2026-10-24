# 用户指定新权重Navtest：2026-10-24 15:25 UTC登记

已固定并校验五个完整检查点，使用同一原始观测、seed2101、单候选10步shift5 ODE及canonical NAVSIM v1.1，各12146场景/136logs。用户要求本次中期Navtest观察，正式训练预算、DEV选择和永久IL53k reference不变；这些新观察将如实披露，不用于当前run调参或重新选择IL父模型。

|模型|固定更新|状态|
|---|---:|---|
|IL|65000|已固定；独立IL进度观察|
|原CANONICAL|50000|已固定；原正式终点|
|原SOURCE|45500|已固定；中期观察|
|WLA_ADV|2000|已固定；早期安全方法观察|
|SCADV3|1300|已固定；早期安全方法观察|

当前登记不含任何新PDMS数字；结果完成后另存聚合摘要与配对安全分析。更新预算不一致，不能把本表作为匹配终点方法排名。

模型推理、LoRA导出和原始图片ODE对照全部调用不可变`434cba8`快照。完整性、聚合及配对分析调用不可变`ee200bc`；本次新增operator只复用旧管线、扩展固定权重数与资源路由，不改actor/采样/评分。登记与实际operator身份见[registration.json](registration.json)、[launch.json](launch.json)。

优先使用vlawm-zt 0–7、tsinghua-zt6 1–7共15个空闲GPU；仅对授权且实测显存允许的vla-zt卡共享推理。zt6 GPU0及其它任务保留。共享设备最多两个模型进程（包括既有训练），CPU并行评分同步进行；工作进程累计时间与端到端wall分别报告。

实际控制目录：`/mnt/project/qwen-sourceflow-native-20261018/latest_NAVTEST_20261024T152508Z`。已有单owner，勿重复启动。已执行入口：

```bash
/usr/bin/python3 -u /mnt/project/qwen-sourceflow-native-20261018/latest_NAVTEST_20261024T152508Z/owner.py \
  --registration /mnt/project/qwen-sourceflow-native-20261018/latest_NAVTEST_20261024T152508Z/registration.json
```

公开operator为[scripts/analysis/qwen_sourceflow/evaluate_registered_navtest.py](../../../scripts/analysis/qwen_sourceflow/evaluate_registered_navtest.py)。完整权重/原始图像/逐场景chain不上传Git。早期结果不能代替两组各50k的固定终点与DEV选择后的最终确认。
