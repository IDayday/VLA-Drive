# NAVSIM 部署与恢复命令

G0/G3已有活动controller。以下命令用于核对和checkpoint边界之后的恢复，
不要在同一个run仍有训练进程时重复运行。训练实现和评价器均使用冻结源
`0c54f1c49e7ed2a76b22826b654d6c72625f5b6d`；本分支新增工具只负责资源路由。

## G0 当前执行与证据

目标服务器：`training-rl-zt2`，完整8卡。实际执行的controller命令为：

```bash
/tmp/structured-world-round1/envs/qwen/bin/python -B -u \
  /mnt/project/VLA-Drive-structured-world-fgtr-round1-20261008/tools/structured_world/deploy_navsim_g0.py \
  --qualify-and-start
```

此入口已完成16步profile和8→16步恢复验证，再以共同规定初始化启动独立正式run。
它是一次部署入口，不能用重复运行它替代正式run的native恢复。

实际正式run、配置、数据和调度见
[部署记录](NAVSIM_RESOURCE_DEPLOYMENT_20261011.json)。
完整worker argv在该JSON的`G0.formal_command`中。

已保存native checkpoint并在原controller停止后的G0恢复命令，在`rl-zt2`上执行：

```bash
training_source=/mnt/project/VLA-Drive-structured-world-fgtr-io-formal-frozen-20261009
asset_root=/var/tmp/structured-world-fgtr-round1-20261008/navsim_g0_deploy_20261010T2355
artifact_root=/mnt/project/structured-world-fgtr-round1-artifacts/20261008
run_root=$artifact_root/navsim_g0_rlzt2_deploy_20261010T2355/formal/G0_NO_FUTURE_LABEL
cd "$training_source"
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  MAX_JOBS=4 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  /tmp/structured-world-round1/envs/qwen/bin/python -B -u -m torch.distributed.run \
  --standalone --nproc-per-node=8 -- tools/structured_world/train_vla.py \
  --config configs/structured_world/navsim_G0_NO_FUTURE_LABEL_seed42.yaml \
  --cache "$asset_root/navsim_train_full_label_v5_centered" \
  --output "$run_root" --scope formal --micro-batch 4 \
  --geometry "$artifact_root/navsim_shared_geometry_5epoch_seed42_v1/shared_geometry.pt" \
  --geometry-identity e07c1a00d8961e10f88d71799957bcd647bc4a58009cd6cfbae22f42dfa9fecd \
  --dino-root "$asset_root/navsim_current_C1_only_v1" \
  --dino-identity bda28c73a00fa2d47df2720dd03007f09d006f34482e406b0418a2112ae6924e \
  --save-every 5000 --execution-mode io_preserving_v1 \
  --execution-acceptance "$artifact_root/IO_EXECUTION_ACCEPTANCE_v1.json" \
  --stop-after 25000 --resume
```

`--stop-after`指定下一个已登记观察点，不改变100k scheduler horizon。
到达25k之后使用50k、75k、100k；完成每个milestone的评价后再推进下一段。
原controller已经自动执行这些阶段，正常运行无需人工重复命令。

## G0 固定开发集评价

同一服务器、同一冻结环境；使用上面相同路径变量。达到25k checkpoint后：

```bash
/tmp/structured-world-round1/envs/qwen/bin/python -B -u -m torch.distributed.run \
  --standalone --nproc-per-node=8 -- tools/structured_world/infer_checkpoint.py \
  --run "$run_root" --tag milestone_0025000 \
  --inputs "$asset_root/navsim_dev_current_v1" \
  --output "$run_root/dev_25000_seed42" --sampling-seed 42 --scene-fields

/root/miniconda3/envs/navsim/bin/python -B -u tools/structured_world/score_navsim.py \
  --predictions "$run_root/dev_25000_seed42" \
  --index /mnt/project/local-interaction-mask-v2-artifacts/20260927/official_dev_metric_v2/cache_index.json \
  --devkit /mnt/project/local-interaction-mask-v2-artifacts/20260927/reference_sources/navsim-v1.1 \
  --output "$run_root/dev_25000_canonical" --population dev --workers 12

/tmp/structured-world-round1/envs/qwen/bin/python -B -u tools/structured_world/score_scene_fields.py \
  --predictions "$run_root/dev_25000_seed42" \
  --labels "$asset_root/navsim_dev_label_v1_centered" \
  --output "$run_root/dev_25000_scene_metrics"
```

50k/75k/100k同步替换checkpoint tag与对应输出目录。部署输入只有当前图像、
导航、合法ego状态和标定；后两条CPU评分才读取离线标签/metric cache。

## G1 路由恢复与原队列

[resume_navsim_group.py](../../../tools/structured_world/resume_navsim_group.py)
从原`EXECUTED_CONTROLLER.json`提取实际train/infer/score/scene命令，
不会自动恢复旧G0队列。默认只读；CPU准备和实际执行为独立入口。
每次准备/执行均要求新的control输出目录，且检查checkpoint、来源、资产、
GPU锁和现有停止请求。实际已执行命令、最终接替服务器和恢复验证在G1独立
部署记录中保存。不能把已有NU资格标记改名为另一台服务器的资格。

G1已经在`training-vlawm-zt2`实际恢复。
[成功恢复记录](NAVSIM_G1_RECOVERY_20261011.json)中的`controller_launch`
保留本次执行argv；`verified_native_boundary`和`eight_rank_probe`提供真实验证。
其controller自动推进50k/75k/100k和开发评价。已验证数据准备可通过
`--reuse-prepared`复用，仍重新检查运行环境、原身份和metadata；未来CUDA更新
不承诺bitwise相同。

恢复控制测试复现，仅运行CPU/mock，不派发GPU worker：

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  /tmp/structured-world-round1/envs/qwen/bin/python -B \
  tools/structured_world/validate_recovery_routing.py \
  --output /mnt/project/structured-world-routing-validation-new
```

输出目录必须不存在且位于源码、原实验资产目录之外。

G3/G2的原controller和source不变。G3当前继续到100k，G2在其登记队列中等待。
nuScenes恢复helper本轮保留为后续工具，按用户安排不执行。
