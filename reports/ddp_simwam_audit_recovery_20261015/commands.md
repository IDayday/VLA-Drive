# 实际执行入口

以下环境已执行验证；没有新的算法训练配置。Python为
`/mnt/project/flow-safe-improvement-20261013/venv/bin/python`。
CPU审计工作目录为 `/mnt/project/VLA-Drive-simwam-audit-recovery-20261015`。
真实训练资格在不可变快照
`/mnt/project/simwam-audit-recovery-20261015/source/73dd0f7c4ea799c5d81420c168943187baa83c86`。
所有输出文件/目录已存在，重做验收须使用新输出身份，不能覆盖证据。

## CPU与只读复核

CPU验收显式设置 `CUDA_VISIBLE_DEVICES=""`，不初始化本地GPU。

```bash
python -m scripts.analysis.simwam_recipe.qualify_transactions \
  --output /mnt/project/simwam-audit-recovery-20261015/qualification/cpu_transactions_v4
python -m scripts.analysis.simwam_recipe.qualify_accounting \
  --output /mnt/project/simwam-audit-recovery-20261015/qualification/accounting_v3.json
python -m scripts.analysis.simwam_recipe.audit_checkpoints \
  --run /mnt/project/simwam-recipe-20261015/runs/R_SOURCE_seed42_v1 \
  --output /mnt/project/simwam-audit-recovery-20261015/R_SOURCE_endpoint_audit.json
python -m scripts.analysis.simwam_recipe.summarize_training \
  --run /mnt/project/simwam-recipe-20261015/runs/R_SOURCE_seed42_v1 \
  --window-updates 5000 \
  --output /mnt/project/simwam-audit-recovery-20261015/R_SOURCE_training_v2.json
python -m scripts.analysis.simwam_recipe.audit_evaluations \
  --root /mnt/project/simwam-recipe-20261015/evaluations \
  --output /mnt/project/simwam-audit-recovery-20261015/evaluation_v2
```

R_CANONICAL也执行相同审计/汇总命令，替换run与输出名。`audit_evaluations`只验证已有轨迹/评分，保留旧摘要；中断后只复用已验证的新摘要，不运行模型。`audit_checkpoints --checkpoint <目录>`是单点只读分类入口，不等于授权恢复。

## 八卡资格与恢复

在当前授权、空闲的Tsinghua zt2执行，环境 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 NO_ALBUMENTATIONS_UPDATE=1`。下列命令实际分别以stop8连续、stop2、从inner2到stop3、从inner3到stop8运行：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  --module scripts.analysis.simwam_recipe.train \
  --config configs/flow_rl/simwam_source_ddp.yaml \
  --manifest /mnt/project/simwam-recipe-20261015/qualification/real_TRAIN_tokens.json \
  --qualification --save-every 100 --stop-after-update 8 \
  --output /mnt/project/simwam-audit-recovery-20261015/qualification/gpu8_recovery_v2/resumed \
  --resume /mnt/project/simwam-audit-recovery-20261015/qualification/gpu8_recovery_v2/resumed/checkpoints/update000003_inner3

python -m scripts.analysis.simwam_recipe.qualify_recovery \
  --continuous /mnt/project/simwam-audit-recovery-20261015/qualification/gpu8_recovery_v2/continuous \
  --resumed /mnt/project/simwam-audit-recovery-20261015/qualification/gpu8_recovery_v2/resumed \
  --interrupted-after 2 3 \
  --output /mnt/project/simwam-audit-recovery-20261015/qualification/gpu8_recovery_v2/result.json
```

两卡故障资格使用 `--nproc_per_node=2 --qualification`，分别注入
`--fault before_collect --fault-rank 0`、`--fault after_collect --fault-rank 1`、`--fault before_finalize --fault-rank 1`。
输出分别在 `qualification/gpu2_faults_v1/`。故障入口只在qualification允许；不用于正式训练。

## 显式跨源码迁移（仅隔离副本执行过）

```bash
python -m scripts.analysis.simwam_recipe.make_migration_receipt \
  --checkpoint /mnt/project/simwam-audit-recovery-20261015/qualification/legacy_migration_v1/checkpoints/update000001_inner1 \
  --new-identity /mnt/project/simwam-audit-recovery-20261015/qualification/gpu8_recovery_v2/continuous/identity.json \
  --parity-evidence /mnt/project/simwam-audit-recovery-20261015/qualification/normal_path_parity_v2.json \
  --recovery-evidence /mnt/project/simwam-audit-recovery-20261015/qualification/gpu8_recovery_v2/result.json \
  --output /mnt/project/simwam-audit-recovery-20261015/qualification/legacy_migration_v1/migration_receipt.json
```

实际恢复使用上面的八卡train入口、相同资格manifest，加上：

```text
--output /mnt/project/simwam-audit-recovery-20261015/qualification/legacy_migration_v1
--resume /mnt/project/simwam-audit-recovery-20261015/qualification/legacy_migration_v1/checkpoints/update000001_inner1
--migration-receipt /mnt/project/simwam-audit-recovery-20261015/qualification/legacy_migration_v1/migration_receipt.json
--qualification --stop-after-update 8
```

旧run identity/source_files保留，新execution与receipt另外存档。receipt不适用于不同源码目录或未验证的进一步改动。正式两组已完成，不应重新执行正式train或将这些资格权重作为正式起点。

## 实际采样检查和已有batch诊断

```bash
python -m scripts.analysis.simwam_recipe.qualify_sample_recompute \
  --checkpoint /mnt/project/simwam-recipe-20261015/runs/R_SOURCE_seed42_v1/checkpoints/update020000_inner0 \
  --output /mnt/project/simwam-audit-recovery-20261015/qualification/sample_recompute_v1.json
python -m scripts.analysis.simwam_recipe.diagnose_saved_batch \
  --run /mnt/project/simwam-recipe-20261015/runs/R_SOURCE_seed42_v1 \
  --checkpoint /mnt/project/simwam-recipe-20261015/runs/R_SOURCE_seed42_v1/checkpoints/update020000_inner0 \
  --collection 5000 \
  --output /mnt/project/simwam-audit-recovery-20261015/gradient_diagnostics/R_SOURCE_update020000.json
```

相同诊断执行了两组10k/15k/20k，collection分别2500/3750/5000；5k复用旧输出。无新评分或搜索。

## 原始观测评价账本smoke

下面命令实际运行两次。第一次1个场景新推理，第二次完整复用；这不是新增正式seed/人口评价。

```bash
python -m scripts.analysis.simwam_recipe.evaluate_policy \
  --export /mnt/project/DriveDreamer-Policy-flow-grpo/artifacts/action-only-checkpoints-v1/frozen_visual \
  --export-sha256 9a26685aa3838a2e1b89ab2d92997664afde4b259fed6683851f42781ad16eb4 \
  --cohort DEV --shard 0 --shards 8 --max-scenes 1 \
  --output /mnt/project/simwam-audit-recovery-20261015/qualification/evaluation_ledger_v1
```

主结果继续引用原锁定的21份完整评价，不用本smoke替代。新评价代码hash发生变化，不能拿旧Navtest锁直接启动新代码；本轮没有这么做。
