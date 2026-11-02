# Training configurations and executable commands

所有主算法固定真实D44 parent、fresh Adam；FP32 parameters/optimizer/action-state/Gaussian，BF16 forward；seed42。部署seed2101，B1/K10/shift5，无R/scorer/CoT。

P0正式登记：D0=4epochs，D2=6epochs，D5=6epochs；B64不同scene/G16/M2，单机8卡、每rank累积8个scene。DDP按全局有效scene分母归约；padding不贡献梯度，也不计有效曝光。真实两rankGloo与显式全batch梯度对照已通过，包含部分rank安全等分与padding。

N=103288，每epoch1614 fresh global collections、3228 optimizer updates、103296 raw exposures（8 padding）。D0四epoch12912updates，D2/D5六epoch各19368updates。四次或两次inner不能记成新候选。

训练LR由scene-epoch计时；text LoRA起点2e-6，warmup0.08epoch，hold0.60，cosine到4.0，min ratio0.05。D5额外FiLM1e-5、I/O3e-6、position1e-6；独立Adam参数组，clip1。学习率不得按观测Navtest无记录修改。

新开发入口可直接执行：

```bash
python -m pytest -q tests/d44_native_rl
python -m tools.d44_native_rl.campaign register --source /ABS/FROZEN_SOURCE
python -m tools.d44_native_rl.campaign run --interval 1800
python -m tools.d44_native_rl.campaign status
python -m tools.d44_native_rl.report --output /ABS/STATUS.json
```

以上调度命令必须在本任务资源环境中执行，CPU owner也设：

```bash
export WORLD_COMPARE_ROOT=/mnt/project/d44-native-rl-20261103
export WORLD_COMPARE_TASK=d44_native_rl_20261103
```

原生训练/恢复入口由授权host上的UUID/lease worker启动：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m tools.d44_native_rl.train --config /ABS/REGISTERED_CONFIG.yaml \
  --output /ABS/RUN
# 同一命令增加 --resume /ABS/LEGAL_CHECKPOINT
```

正式第一次100updates保存退出后，调度器用同一源码/配置/Adam/RNG/old-material恢复继续，不从资格权重初始化。collection失败只保存failure evidence，latest不推进。inner中断继续原链剩余inner。

完整评价入口：

```bash
python -m tools.d44_native_rl.evaluate --checkpoint /ABS/CP \
  --output /ABS/EVALUATION/evaluation --rank 0 --world 8
# 所有8个独立shard完成后：
python -m tools.d44_native_rl.evaluate --summarize \
  --output /ABS/EVALUATION/evaluation --baseline /ABS/D44_BASELINE/evaluation
```

`--limit`只能用于资格，结果带qualification标记。正式评测不得以小样本结果替代12146。0.5/1/2/3/4/6/8/12/20epoch生成原子评价请求；0.25epoch属于训练资格观察。所有独立算法的Adam都重新创建；恢复同一run才保留Adam。

D6未完成生命周期时，读配置明确抛出NotImplementedError。尚未验收的路线不得混入已登记P0正式run。
