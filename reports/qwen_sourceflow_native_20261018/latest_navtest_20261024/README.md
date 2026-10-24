# 最新IL与两条RL的完整Navtest（2026-10-24）

三组均覆盖12146场景、136logs；训练seed42、评价seed2101；单候选、source shift5十步ODE、canonical NAVSIM v1.1。没有best-of-N、随机SDE部署或scorer。权重在05:45UTC锁定，本批是用户要求的中途观察，不改变IL100k、RL各50k终点、53k永久reference或原DEV选择规则。

|模型|optimizer update|PDMS|相对自身IL53k|配对95%区间（PDMS点）|
|---|---:|---:|---:|---|
|自身IL父模型|53000|87.016025|0|—|
|IL|59787|87.106111|+0.090086|[-0.534245, +0.639913]|
|canonical|35800|90.077079|+3.061054|[+2.340550, +3.715691]|
|source|30800|90.264806|+3.248780|[+2.528781, +3.870139]|

IL59,787行仅衡量IL继续学习，不是RL收益。RL父模型始终是53k，且其Navtest已被用户看过后用于明确选择，`Navtest_used=true`。区间为按log聚类、保留scene权重的2000次bootstrap，不含训练seed或推理seed不确定性。

两条RL相对自身IL仍有约3.1–3.25个PDMS点的净收益，均未达到92。相对已发布20k：CANONICAL由90.284539变为90.077079（−0.207459）；SOURCE由90.233689变为90.264806（+0.031117）。继续训练没有显示单调增益。此处两条RL步数不同，不作为训练reward优劣的匹配因果比较，也不按本次Navtest修改训练或挑选checkpoint。

## 安全分项与代价

以下为0–100分项均值，NC=0.5不会被统计为完全通过。

|模型|NC|DAC|TTC|EP|Comfort|DDC|
|---|---:|---:|---:|---:|---:|---:|
|IL53k|98.0899|95.2742|94.0557|81.7227|99.9506|98.4110|
|IL|98.0693|95.7929|94.1874|81.2839|99.9588|98.2916|
|canonical|98.7650|97.8511|96.7067|82.8733|99.9671|98.1105|
|source|98.2916|97.8923|95.2330|84.9990|99.8765|97.8429|

两个RL相对IL53k的NC/DAC/TTC均值均提高；仍存在新增失败，DDC下降，SOURCE的Comfort下降。总分提高不代表每个原安全scene保持安全。

|相对固定IL53k|CANONICAL35.8k|SOURCE30.8k|
|---|---:|---:|
|非零→零|107|136|
|零→非零|494|468|
|固定IL联合安全10905场景中的新增联合违规|143|226|
|no_at_fault_collisions新增/恢复|33/119|57/85|
|drivable_area_compliance新增/恢复|79/392|83/401|
|time_to_collision_within_bound新增/恢复|68/390|146/289|

|收益分解（PDMS点）|CANONICAL35.8k|SOURCE30.8k|
|---|---:|---:|
|正向G+|4.957410|4.923656|
|负向G−|1.896356|1.674876|

## 本批资源与迁移

原24个worker分布在vla-zt、vla-zt3、tsinghua-zt4，与两个RL共享部分GPU。用户新增允许空闲vlawm-zt与tsinghua-zt6的1–7号卡后，实际迁移10个未结束分片：vlawm-zt的0–7，tsinghua-zt6的1、2。其余分片已完成，不为使用GPU重复推理。zt6的0号卡有其他任务，未使用。旧IL/RL没有被发送停止信号。

每个模型的rank/world8划分、模型/源码/精度/后处理、绑定scene的noise及canonical独立评分均未改变。先以精确PID身份向原evaluation发送SIGINT，等待其finally写入中断及成本、释放rank锁，再在新主机以新attempt重启同一rank。已存在scene通过原seal校验后复用；旧失败/中断记录没有覆盖为成功，actor源码没有热改。所有successor退出0，完整人口与请求/评分一致性校验通过。

当前有限legacy资源lease已关闭，用户的后续Navtest任务级授权保留。zt6授权仍仅1–7，其他项目及旧暂停实验没有恢复。资源与文件hash摘要见[resource_migration.json](resource_migration.json)。

## 物理计算成本

|模型|逻辑完整scene|已记录完成推理|恢复检查复用计数|累计worker wall秒|本批端到端跨度秒|
|---|---:|---:|---:|---:|---:|
|IL|12146|12146|2700|15030.35|2164.60|
|canonical|12146|12146|7110|15738.49|2178.55|
|source|12146|12146|4045|15464.31|2187.12|

复用计数是恢复时验证已有文件的次数，不是新增候选，也不应叠加到scene数。已记录完整推理各12146次；中断可能发生在forward计数递增之前，因此未完成的部分forward次数不能精确回溯，上限为每个被中断worker1次（本批10个）。其时间包含在中断worker wall中。摘要v3中的`physical_inference_including_retries`在此应理解为已记录完成推理，不能据它声称没有任何中途重算开销。多个worker wall之和不是端到端wall，也不是独占GPU计算时间。模型加载、forward和评分等待见各summary，迁移成本未删去。

## 资产与复核

评测原始资产：`/mnt/project/qwen-sourceflow-native-20261018/latest_NAVTEST_20261024T054539Z/`；迁移/中断/新attempt：`/mnt/project/qwen-sourceflow-native-20261018/navtest_resource_move_20261024/`。报告只包含聚合摘要，不上传原始场景、图像、轨迹、权重、optimizer或cache。

同目录IL/canonical/source的summary和paired JSON均为实际完整校验产物。对应正式推理源码434cba8，分析源码70a68db；各JSON包含其自身seal与identity。已有20k结果保留，历史Navtest多次观察事实不变。
