# 已执行入口与正式任务登记

运行资产根目录为 `/mnt/project/decision-local-consequence-artifacts/20261014`。依赖环境是该目录下 `envs/qwen/bin/python`；canonical CPU评分沿用 `/root/miniconda3/envs/navsim/bin/python`。代码、输入缓存和训练产物都在共享 `/mnt/project`，没有升级其它任务的环境。

## 精确命令的实际收据

| 工作 | 命令、源码和状态的收据 |
|---|---|
| 21卡续接S0训练缓存 | `resources/cache_redistribution_21/cache_21_rank_00.json` 至 `20.json`；逐worker源码与续接数值证据在 `anchor_train_full_v1/workers/` |
| 15个正式小模型 | `HEAD_TASKS.json` 每个job的 `command`/`training_source_sha`/`requires`；实际启动与返回码在 `head_queue_runtime/claims/`、`results/` |
| 40次开发规划观察 | `development_observations_v1/REGISTRY.json` 中 `commands.export` 与 `commands.score`；原canonical scorer不修改 |
| 15项预测器终点关系检查 | `relation_endpoints_v1/REGISTRY.json` 中每个job的 `command`；包括固定训练4096、完整开发和固定新扰动幅度 |
| 真实在线部署 | `online_deploy_smoke4_kinv2/` 是已执行4图链路；`CAMERA_ONLY_KINV2.json` 与 `CAMERA_ONLY_E_NONE_DIAGNOSTIC.json` 是在线/缓存比较 |
| 恢复验证 | `RESUME_NUMERICAL_ASSESSMENT.json`、`OUTPUT_ASSESSMENT.json` 及仓库 `evidence/RESUME_AA_AB.json`、`RESUME_OUTPUTS.json` |

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

Navtest的新缓存、导出均要求 `FINAL_METHOD_LOCK.json`。本轮不会为每个epoch启动Navtest，也不会由开发观察器自动选择“最好”checkpoint。最终方法与共同终点确认后，再为实际最终命令生成收据。
