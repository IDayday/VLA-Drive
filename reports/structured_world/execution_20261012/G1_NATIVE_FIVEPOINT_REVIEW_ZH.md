G1五点真实原生CPU比较已完成5/5；完成于2026-10-12 02:57:22 UTC，本次只读复核既有结果，没有重启或重算。

每点A/AA/B加同一25k origin，完整校验17个原生文件、8个optimizer分片与8份RNG。覆盖1146个可训练参数张量、2633678885个参数元素、7个optimizer组以及模型、BN和Adam一/二阶矩。15个capture bank与1个origin共272个不同payload文件；五点原comparator重复校验origin，总340次文件校验。

五点所有A/A、A/B、AA/B的E1均通过；原始bitwise strict和固定atol=1e-6、rtol=1e-4门限均失败，五次exit2完整保留。全部浮点类别没有nonfinite元素。

|继续步数|Master A/A relL2|Master A/B relL2|BN A/A relL2|BN A/B relL2|group6二阶矩 A/B÷A/A差异范数|
|---:|---:|---:|---:|---:|---:|
|1|7.7001859e-08|7.8285997e-08|0|0|1.058210|
|32|7.8200546e-05|7.8258282e-05|0.00049692645|0.00049153089|0.777974|
|100|0.00046887921|0.00046384529|0.0033566062|0.0034435219|1.905879|
|250|0.0013219407|0.0013132987|0.0081719881|0.0080445931|1.896255|
|500|0.0024118843|0.002390672|0.012219812|0.011162978|4.601859|

500步整体master A/B relL2为0.002390672，A/A为0.0024118843。但group6（geometry/future_space/refiner/scene_semantics）Adam二阶矩的A/B差异为A/A的4.601859倍；不能用整体接近A/A掩盖分组扩大。A/A自身也发生训练漂移，当前结果不能单独证明全部差异由P0造成，也不能登记候选无损。

五个CPU进程串行墙钟合计2.778087小时；原native三角色训练进程成本12.290808 GPU小时，两者分别登记。当前payload存在性复核为272/272；这里未再次大规模读取payload，完整SHA验证来自未修改的原comparator实际结果。

注意：冻结工具在每个pair下的origin_relative_updates实际都使用A/B误差与更新cosine。本摘要只借用其中共享的A/AA/B更新范数，再用各pair真实difference_norm重算更新相对误差；原报告未改写。P50/P90/P99是逻辑张量最大误差的分位数，固定门限违例总数含不同类别重复检查，不能当作独立参数数量。

本证据是权重/optimizer/BN五点演化，不是车辆q0/Bt/q_final轨迹或PDMS比较。仍缺本证据之外的G3/early、恢复、配对canonical dev及完整source/receipt准入链；已有fullgrad负例保持。passed=false、formal_authorized=false；未改trainer/guard/阈值，未启动GPU、复制/生成/删除权重或commit。
