# 优化轨迹标签 Qwen-World IL：中间权重 Navtest

同一通用初始化、100k预算的优化标签IL；13k/15k在新增评分前固定，10k结果直接复用。完整12146场景、canonical v1.1、单候选clean-x十步shift5 ODE、seed2101。不重复原controller已登记的20k及后续里程碑，不根据Test重选训练起点。Test历史已被观察。

| Checkpoint | Updates | PDMS | Change vs optimized IL10k (points) | Change vs previous optimized IL checkpoint (points) |
|---|---:|---:|---:|---:|
| OPTIMIZED_IL_013000 | 13000 | 66.369502 | -13.534773 | -13.534773 |
| OPTIMIZED_IL_015000 | 15000 | 81.359718 | +1.455444 | +14.990216 |

同一优化标签IL的10k历史完整结果作为共同参考，13k/15k逐点比较另列。模型还在完整100k训练的早期，不能用这些中间点判断优化标签的最终价值。区间按物理log聚类，保留场景权重；不含训练或推理seed不确定性。逻辑新增24192条场景评测，额外2次native/export一致性；共享worker费用见资产根的逐调用收据。