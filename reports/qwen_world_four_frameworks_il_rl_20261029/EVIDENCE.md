# 四架构共用的原生候选证据库

## 固定来源

候选只读来源为 `qwen-world-postcot-20261025/candidates/full_train_initial_v1`，90890个TRAIN场景，每场景16条精确去重物理轨迹，共1,454,240条。bank hash为 `356d24a0e8e9e6095a9dd1f8d2d96fe39394ca51408124f6d51b62cd99f95b7f`，manifest文件SHA256为 `76ab503f36d89bf6790394a501434e642fd3ba1ef32b1cebd319d3834399e5d3`。

本轮不刷新候选，也不按各架构分数重新选择成员。原候选包含真实记录轨迹、公共policy输出和已真实评分的平滑coherent扰动。新建库只补齐精确候选的原生时序证据，不生成外部on-policy概率。

评分协议SHA256仍为 `2dee107a4e8358bc873530c0eb2ee16d8761e471eab1d80acc76022e789654e1`。复用 `batched_canonical_score.score_scene`；仿真和原始分项可以批量，最终progress归一化保持每条候选独立的 `[固定官方reference, 一条候选]`。不会让G条候选共同改变EP分母。

## 事件合同

每条轨迹重新执行原生simulator/scorer，一次执行同时取得六分项、canonical总分和事件证据：

- NC：官方 `time_to_at_fault_collision()`，保留NC=0.5的合法中间分。
- DAC：官方0.1秒网格上的 `NON_DRIVABLE_AREA` 车身mask，取第一次true。
- TTC：官方 `time_to_ttc_infraction()`；它是首次规则违规时间，不是最小TTC值。

8个半秒区间使用 `[0,.5],(.5,1],...,(3.5,4]`，t0映射第一类；边界容差1e-8秒。已知全窗口无事件为第8类，缺失反馈则报错，不能填成none。

每条记录绑定旧record_id、exact FP32轨迹/hash、观测/hash、归一化/坐标版本、metric-cache SHA、原evaluator及本次原生源码hash、六分项与valid、总分与valid、三事件首次时间/类别/valid。保留完整41步官方DAC布尔mask；大规模simulated_states/body仅保存shape/dtype/字节hash，不伪装成完整时序数组。原始请求轨迹完整保留。

## 事务和费用

`EvidenceWriter`每个scene一个SQLite事务。只有所有候选的请求身份、cache、旧分项/总分和事件合同一致，才提交整场景。部分采样、失败scene或缺少任一part时都不发布完成标记。断点恢复复用已提交scene；尚未提交的完整scene重新评分。

独立attempt账本记录提交的候选数、实际返回分数数、已知评分wall、simulator batch数。进程中断且无法知道实际评分进度时，保留 `INTERRUPTED_UNKNOWN_PARTIAL`，不将其成本当作0。全局合并只生成record_id→part索引，不复制所有大payload。读取者只允许封存完整库；`get(old_record_id)`与`scene_records(token)`均重新验证record seal。

CPU owner和每个part worker保存PID/start_ticks。`owner.lock`与part writer锁防重复。SIGUSR1/SIGTERM只建立STOP_REQUESTED，worker在完整scene边界停止；新的task授权每scene检查。失败时其他part也在scene边界停止，不扩展GPU或修改未知任务。

## 已执行的验证

1. `test_evidence.py`：20项CPU测试通过，含物理事件边界、NC=.5、missing≠none、错轨迹/cache/score拒绝、完整scene提交、事务中第二条insert故障回滚、未完成拒绝发布、resume及成本、parts合并和reader完整性。
2. `qualification/evidence_native32_v2`：两个固定TRAIN场景×16候选。官方逐候选入口与chunk1、chunk4逆序、chunk16逐项对照；总分/全部分项/执行状态/事件时间最大误差均0。资格开销128次候选评分、74个simulator batch。首32里有4条DAC事件。
3. `qualification/evidence_native_failure_probe_v1`：另按固定TRAIN次序找到一条实际NC/TTC失败候选，只新增2次评分。官方与批量首次事件均为NC3.6s、DAC0.7s、TTC2.7s，对应[7,1,5]。定向选择只用于接口验收，不是性能评估。
4. `qualification/evidence_build2_v1`：真实2worker构建、scene提交、parts合并、Reader.get/scene_records，全部32条请求hash和分数与原库严格一致。两scene合计已知评分wall3.1999秒；总SQLite229376字节。该小样本不能代表复杂全人口的准确耗时。

v2资格绑定当前核心源文件：`evidence.py=28debee579a731c93a46101eb75018d19e85aea63c63c130d128557d63f1348c`，CLI=`c2654f6c26c59ab80e74ca7d2c4ae77bc4cdbc3d952120db70af6ebe51d87072`。v1资格和build2是前序资格记录，保留各自独立source identity；没有覆盖它们伪装成同一执行版本。

## 实际入口

```bash
python -m tools.qwen_world_compare.build_evidence_bank validate --bank <sealed_fixed_bank> --output <new_qualification_root>
python -m tools.qwen_world_compare.build_evidence_bank build --bank <sealed_fixed_bank> --output <evidence_root> --workers 8 --resume
python -m tools.qwen_world_compare.build_evidence_bank finalize --bank <sealed_fixed_bank> --output <evidence_root>
```

训练机器、CPU并发和路径必须由当前grant及实占解析；上述不是绕过task授权的通用启动器。完整库要在本轮immutable数据源码快照上启动；此文不把资格32条当作完整1,454,240条建设完成。原候选库历史构建成本和新时序证据重评分成本分别保留，候选没有被当成免费数据。

## 完整建设已实际启动

按主负责人统一资源安排，正式CPU并发锁定为24（不是上面入口示例的8）。2026-10-29在`training-rl-zt2`启动唯一owner PID147711/start_ticks1465467559，实际源码为 `runtime/data_7c59ce9`，提交`7c59ce9c2cf747adf7f435f1fd288d1477f7a8f5`；工作树干净且五个科学依赖文件逐hash与native32_v2资格一致。正式输出为实验ROOT的 `evidence`。当时128可用CPU、load1约10、MemAvailable约737GiB，未占用GPU。

24个part worker已逐一校验真实PID/start_ticks、parent PID、存活状态及`CUDA_VISIBLE_DEVICES=''`。启动约两分钟时，以每part只读事务一致地观察到8385场景/134160条证据已提交，所有scene均16条，未发生部分scene发布。该数据是进度快照，不是全库完成。

独立小库 `qualification/evidence_first32` 已完成32场景/512条，bank hash=`bd085be359a1d9c3c5203901f933ed5525ee3ca68b07e7a36da0aef84d5435a5`，仅供真实IL资格验收，不代替正式全人口库。其8worker owner PID147712/start_ticks1465467561已完成。此处原生32候选数学验收、32场景小库、完整90890场景库是三个不同预算。

实际启动命令：

```bash
python -u -m tools.qwen_world_compare.build_evidence_bank build \
  --bank /mnt/project/qwen-world-postcot-20261025/candidates/full_train_initial_v1 \
  --output /mnt/project/qwen-world-four-frameworks-20261029/evidence \
  --workers 24 --resume
```

实际launcher同时固定单线程OpenBLAS/OMP/MKL、当前task grant及immutable PYTHONPATH；LAUNCH、PLAN、各part WORKER、STARTUP_PROCESS_PROOF和STARTUP_COMMIT_COUNTS保存在正式输出目录。不要因看到这些命令就再次启动第二个owner。原先两scene冷启动1.60秒/scene不能准确外推完整任务；24worker实际批量处理已显著快于该冷启动估计，ETA应采用后续实测窗口。
