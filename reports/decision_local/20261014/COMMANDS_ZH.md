# 已执行入口与正式任务登记

运行资产根目录为 `/mnt/project/decision-local-consequence-artifacts/20261014`。依赖环境是该目录下 `envs/qwen/bin/python`；canonical CPU评分沿用 `/root/miniconda3/envs/navsim/bin/python`。代码、输入缓存和训练产物都在共享 `/mnt/project`，没有升级其它任务的环境。

## 精确命令的实际收据

| 工作 | 命令、源码和状态的收据 |
|---|---|
| 21卡续接S0训练缓存 | `resources/cache_redistribution_21/cache_21_rank_00.json` 至 `20.json`；逐worker源码与续接数值证据在 `anchor_train_full_v1/workers/` |
| 最初15个正式小模型 | `HEAD_TASKS.json` 每个job的 `command`/`training_source_sha`/`requires`；实际启动与返回码在 `head_queue_runtime/claims/`、`results/` |
| 40次开发规划观察 | `development_observations_v1/REGISTRY.json` 中 `commands.export` 与 `commands.score`；原canonical scorer不修改 |
| 15项预测器终点关系检查 | `relation_endpoints_v1/REGISTRY.json` 中每个job的 `command`；包括固定训练4096、完整开发和固定新扰动幅度 |
| 真实在线部署 | `online_deploy_smoke4_kinv2/` 是已执行4图链路；`CAMERA_ONLY_KINV2.json` 与 `CAMERA_ONLY_E_NONE_DIAGNOSTIC.json` 是在线/缓存比较 |
| 恢复验证 | `resume_smoke/RESUME_NUMERICAL_ASSESSMENT.json`、`resume_smoke/OUTPUT_ASSESSMENT.json` 及仓库 `evidence/RESUME_AA_AB.json`、`RESUME_OUTPUTS.json` |

登记不等于已训练完成。完整人口、对应预测器24epoch终点等依赖未齐的job保持待启动，不能把队列名称当结果。正式训练源码固定为 `/mnt/project/VLA-Drive-decision-local-execution-frozen-20261014` 的 `3c08240a…`；后续报告提交不会改变后台训练器。

以下读取方式可打印一条实际登记的完整命令，不重复启动已有工作：

```bash
python - <<'PY'
import json, shlex
path = '/mnt/project/decision-local-consequence-artifacts/20261014/HEAD_TASKS.json'
jobs = json.load(open(path))['jobs']
job = next(j for j in jobs if j['id'] == 'formal_seed42_P_DIFF')
print(shlex.join(job['command']))
PY
```

## 原生暂停与恢复

训练器每次optimizer边界检查本run目录的 `STOP_REQUESTED`。受控暂停将完整模型、optimizer、scheduler、Python/NumPy/CPU/CUDA及shuffle RNG、epoch/offset/order和累计曝光写入原子 `latest.pt`，并写 `PAUSED.json`。必须确认原PID已经退出、checkpoint完整后，才能在合格资源上用**同一个登记command增加 `--resume`**恢复；保持原源码、输出路径、数据身份和所有科学参数。停止标记先记录归档，再移出run目录，否则恢复会立即再次暂停。不要运行两份同一run。

`--max-seconds` 是该次执行的边界预算，不重置scheduler或曝光；到期也走相同保存路径。原E_NONE短恢复逐位一致；P_DIFF的原严格逐位失败仍保留，E1状态一致，完整参数/moments/输出差异与连续A/A比较后单列。正式24epoch完成的run无需恢复，不应为测试再次训练或改成32epoch。

## 本轮新增分析的真实执行示例

以下工具已经在真实开发/训练缓存上运行。输出为不可变目录或文件，复跑时使用新的输出路径，不覆盖既有结果。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python \
  tools/decision_local/audit_information_content.py \
  --labels /mnt/project/decision-local-consequence-artifacts/20261014/labels_dev_current_ground_kinv2 \
  --output /mnt/project/decision-local-consequence-artifacts/20261014/INFORMATION_CONTENT_DEV.json

/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python \
  tools/decision_local/audit_current_frustum.py \
  --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_dev_full_v1 \
  --queries /mnt/project/decision-local-consequence-artifacts/20261014/queries_dev_full_kinv2 \
  --output /mnt/project/decision-local-consequence-artifacts/20261014/current_frustum_dev_v1 \
  --max-seconds 900
```

`analyze_planning.py` 已对真实canonical调试结果执行失败分组；`visualize_failure_cases.py` 已生成并检查道路/障碍/完整车身图。病例按固定类别、四种零/非零转移和差值10/50/90%分位同时抽取，空分层不补成功例。派生地图只在受控共享目录保存，不提交原图或场景地图到Git。

Navtest的新缓存、导出均要求 `FINAL_METHOD_LOCK.json`。本轮不会为每个epoch启动Navtest，也不会由开发观察器自动选择“最好”checkpoint。最终方法与共同终点已确认，实际最终命令收据如下。


## 本轮最终登记与可执行命令

下列目录均位于上述artifact根目录。它们记录的是已经启动/执行的argv、源码、设备和输出身份，不是待办模板；完整结果状态以对应COMPLETE为准。运行已完成的训练命令会拒绝覆盖，不能把这些展示命令直接当新的实验提交。

| 阶段 | 实际命令/进程收据 | 完整状态/结果 |
|---|---|---|
| 17个正式头 | `HEAD_TASKS.json` + `information_seed43_v1/HEAD_TASKS.json` | `PRIMARY_HEADS_COMPLETION.json`，每个原生24epoch状态与SHA |
| 两折ABS/DIFF | `conditional_oof_v1/HEAD_TASKS.json`、`EVALUATIONS.json` | 各run的COMPLETE及`pooled_heldout_v1` |
| 48次开发观察 | `development_observations_v1/REGISTRY.json` + `information_seed43_v1/EVALUATIONS.json` | `final_common_analysis_v1/COMPLETE.json` |
| 六个正式在线一致性检查 | `formal_deployment_v1/REGISTRY_v2.json`、`REGISTRY_RL45.json` | 六份`seed*_*.json`，均通过 |
| 真实图像64场景 | `formal_deployment_v1/REGISTRY_TSG5.json` | `online_DIFF_64/COMPLETE.json`；与缓存最大差0 |
| 实际Qwen调用/量化诊断 | `instrumented_deployment_v1/REGISTRY_V2.json`、`REGISTRY_TSG5.json` | E_NONE/R_DIFF各4场景COMPLETE及parity |
| 固定最终六组Navtest | `final_navtest_v1/REGISTRY_VLA.json`、`REGISTRY_RL.json`、`REGISTRY_TSG5.json` | `THREE_HOST_ROUTING.json`最终分配；旧两机登记保留，移出任务已被原子claim |
| 原生checkpoint终点核验 | `tools/decision_local/receipt_completed_heads.py` | `PRIMARY_HEADS_COMPLETION.json` |
| 失败及案例 | `endpoint_case_analyses_v1/JOBS.json`、`endpoint_cases_v1/*_RUN.json` | 固定分组与双向失败，私有派生地图不进Git |

E/P/R由同一个`train_small_model.py --kind ...`入口执行，相应类型和预测器SHA严格检查。查询与标签由`build_query_trajectories.py`、`build_consequence_labels.py`构建；G只读由`evaluate_g_relations.py`读取；`evaluate_relations.py`与`compare_relations.py`汇总能力；没有复制一套独立训练系统。

已执行的完整64场景部署命令（标签/特征cache文件访问被guard拒绝）：

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u \
  /mnt/project/VLA-Drive-decision-local-deployment-v2-frozen-20261014/tools/decision_local/predict_current.py \
  --anchor /mnt/project/decision-local-consequence-artifacts/20261014/ANCHOR.json \
  --refiner /mnt/project/decision-local-consequence-artifacts/20261014/formal_seed42/R_FUTURE_DIFF/epoch_24.pt \
  --predictor /mnt/project/decision-local-consequence-artifacts/20261014/formal_seed42/P_DIFF/epoch_24.pt \
  --output /mnt/project/decision-local-consequence-artifacts/20261014/formal_deployment_v1/online_DIFF_64 \
  --index /mnt/project/decision-local-consequence-artifacts/20261014/formal_deployment_v1/fixed_dev64.json \
  --split dev --max-seconds 1200
```

打印某个已登记Navtest的原始完整导出/评分命令：

```bash
python - <<'PYCODE'
import json, shlex
root = '/mnt/project/decision-local-consequence-artifacts/20261014/final_navtest_v1/'
job = next(j for j in json.load(open(root+'REGISTRY_TSG5.json'))['jobs']
           if j['id'] == 'seed43_R_FUTURE_DIFF_epoch24')
for phase in ('export', 'score'):
    print(shlex.join(job['commands'][phase]))
PYCODE
```

导出分片的实际`--rank/--shards`和GPU绑定在对应`evaluation_runtime_*/claims/<id>/EXPORT_*.json`；CPU命令仍使用锁定canonical实现。动态路由不改变scene seed、batch1、头FP32、TF32关闭或q_final主输出。已完成的分片/分数只可在身份与哈希一致时按原生机制复用。

原生恢复的精确argv见每个run的`PRIMARY_HEADS_COMPLETION.json.runs[].command`，在已经退出的暂停run上追加`--resume`；RAM续训同时保留`ram_continuation_tsg5_v1`等输入逐字段/逐字节一致性记录。当前所有正式头已完成，不需再次续训。


最后的汇总/原生ego误差/聚合图实际argv保存在`final_navtest_v1/trajectory_errors_launch/COMMAND.json`和`FINAL_ANALYSIS_COMMANDS.json`；全部返回成功。描述性ego误差只在模型轨迹导出后读取未来GT，使用native Scene转换，不参与部署或选轨。初版错误地假设export有index.json而失败，修正为绑定已有feature身份的population元数据后通过64场景GT对照；失败收据保留，未改变模型或canonical评分。

`COST_FINAL_GPU.json`、`FINAL_RESOURCE_INVENTORY.json`、`final_navtest_v1/FINAL_COMPLETION.json`记录实际完成成本、消费者退出与完整场景数。最终只推`feature/decision-local-consequence-learning-20261014`，不合并。推送后的远端SHA在共享目录`FINAL_DELIVERY_RECEIPT.json`核对；它是交付提交，训练源码仍以各run identity为准。
