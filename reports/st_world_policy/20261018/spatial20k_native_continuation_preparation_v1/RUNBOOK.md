# SP2/SP3 5k→共同20k：原生状态续训准备

仅准备，未启动。既定shortlist保留SP2/SP3两臂，不能只延长SP3。源为现有干净运行树 `/mnt/project/VLA-Drive-st-official-runtime-v2-20261016`，固定 `fdf25f99b702754787904215bf1c2a2b7515ea75`；复用已有 `budget_extension.py`，无训练器修改。

两个 `*_CONFIG_PROPOSAL.json` 已生成并通过原 `validate_config` 和原 `_recipe` 相等检查。除累计 `stop_after` 从5000到20000和显式continuation父绑定外，所有训练配置保持。`max_updates=20000`、warmup1000、cosine/AdamW、global32/world4/micro4/accumulation2、BF16执行+FP32master、TF32关闭、当前DINO和原输出/训练精度保持。SP2原PCA v1与SP3原PCA v2路径分别保留，不能为了路径整齐改依赖。

旧config内 `adaptation_reason` 的文字描述仍记录最初stop5000筛选来源，不改变该文本以满足原 `_recipe` 的严格同配方检查；实际新执行上限是已核对的 `execution.stop_after=20000`。

| 组 | 唯一父tag | receipt SHA256 | 新运行名提案 |
|---|---|---|---|
| SP2 | step_005000_1792222283615035511 | 9e8b7694996130c1c6148519c27c9d562df106b46553357b4218e68245637183 | SP2_fulltrain_seed42_official_20k_v1 |
| SP3 | step_005000_1792224403643960209 | d4f5225f84f6d2410f91ed79357258cff68c0a0e84e93f7d696520ac9ee5db93 | SP3_fulltrain_seed42_official_20k_v1 |

父完整路径、registration、哈希、游标见 `PREPARATION.json`。两臂均step5000、exposure159992、epoch1/offset1825。预期下一次optimizer update为5001，最多各新增15000步。底层原生恢复严格加载完整模型/FP32master/Adam/scheduler，核对DeepSpeed global_steps及receipt，再恢复各rank RNG和生成器；不是仅加载推理权重。

## 必须正确区分两种入口

1. **第一次20k身份运行**：新run registration+approval，config包含continuation，命令**不加 `--resume`**。原 `resolve_extension` 核对父完整receipt与载荷，加载父原生状态；写新lineage且保留旧run。
2. **新20k身份随后中断**：同一新registration+approval加 `--resume`，自动选择该新run最新合法checkpoint。不能再以首次continuation入口回到5000。

旧5k config加 `--resume` 不能作为20k续训，因为`stop_after`是累计阈值。不得改写旧config、registration、approval或checkpoint。新运行身份是原训练状态的预算延长，不是新模型或重新warmup。

## Lead执行时待绑定的实际输入

- 使用上述已生成config创建新不可变registration，保留原source fingerprint、训练/teacher/Base/environment/target绑定，执行stop_after20000；新run_id使用表中名字或另一明确新名字。两臂原数学正确性证据和shortlist决定直接继承。
- 新实际四卡placement/UUID、当前允许host/policy、租约和剩余预算。旧5k lease只授权5000阶段且有历史GPU状态，不能沿用来声称20k已准入。
- 当前checkpoint承诺与磁盘容量：每完整native checkpoint SP2为37,366,035,746B、SP3为37,371,024,866B。6k–20k各15个正常周期点，两臂合计1,121,055,909,180B，另计实际wall暂停的边界checkpoint和原子写临时空间。不能自动删除5k或其他旧checkpoint。
- 保持原每次28800s训练墙钟与外部29400s+180s退出界限。到时安全暂停，再从新run resume；不要为一次跑到20k而悄悄扩大原config时限。
- 核查新输出没有重复活跃作业；原训练器 `RUN.lock`/独立启动台账仍使用。实际调用前原 `check_formal_approval` 和严格native恢复各自检查真实依赖。

本报告没有生成虚假的已批准registration/lease，也没有增加新算法门槛。下面变量必须由Lead填写为**实际已创建文件**，不是虚构路径。每次选择一臂，两个任务可以按真实资源独立安排；不自动串行重试。

## 实际CLI命令

`TRAIN_WORLD_HELP.txt` 是本轮实际 `--help` 输出，接口只有registration、approval、resume与acknowledge-stop；没有 `--stop-after` CLI参数。

```bash
cd /mnt/project/VLA-Drive-st-official-runtime-v2-20261016

# 由 Lead 设定实际本臂的新 registration、approval 和获准4卡可见集。
: "${SPATIAL20K_REGISTRATION:?set actual new registration path}"
: "${SPATIAL20K_APPROVAL:?set actual approval path}"
: "${SPATIAL20K_VISIBLE_GPUS:?set approved four-GPU set}"

PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python -B \
  -m tools.st_world_policy.check_formal_approval \
  --registration "$SPATIAL20K_REGISTRATION" \
  --approval "$SPATIAL20K_APPROVAL" --repo "$PWD"

# 首次新20k身份：从config.continuation的固定5k原生父端点加载。
CUDA_VISIBLE_DEVICES="$SPATIAL20K_VISIBLE_GPUS" \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
OPENBLAS_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
timeout --signal=TERM --kill-after=180s 29400s \
  /root/miniconda3/envs/ddp/bin/python -B -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=4 \
  -m tools.st_world_policy.train_world \
  --registration "$SPATIAL20K_REGISTRATION" --approval "$SPATIAL20K_APPROVAL"
```

新20k身份已经产生完整checkpoint后的恢复：

```bash
cd /mnt/project/VLA-Drive-st-official-runtime-v2-20261016
: "${SPATIAL20K_REGISTRATION:?actual same new registration}"
: "${SPATIAL20K_APPROVAL:?actual same new approval}"
: "${SPATIAL20K_VISIBLE_GPUS:?approved four-GPU set}"
CUDA_VISIBLE_DEVICES="$SPATIAL20K_VISIBLE_GPUS" \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
OPENBLAS_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
timeout --signal=TERM --kill-after=180s 29400s \
  /root/miniconda3/envs/ddp/bin/python -B -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=4 \
  -m tools.st_world_policy.train_world \
  --registration "$SPATIAL20K_REGISTRATION" --approval "$SPATIAL20K_APPROVAL" \
  --resume
```

只有确有原operator `STOP_REQUESTED` 且被明确确认继续时才附加原 `--acknowledge-stop`；本准备不创建或删除停止标志。完整20k原生终态不应增加新update。

后续评估比较相同20k端点、原同1696dev/协议与原seed；不拿SP3@20k减SP2@5k。此处未启动评估，也未准备navtest。所有数据角色、单轨迹参考上下文与既有发布边界保持。
