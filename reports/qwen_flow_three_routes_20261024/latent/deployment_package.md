# 独立潜噪声策略部署包

这是部署工程补全，不改变actor特征、网络、超参数、采样器或任何已运行训练。原97、9ecd及正在运行的CENTER7ec快照不修改。新共享分片评价入口可以加载独立q包，不再从训练checkpoint取optimizer、旧rollout或外部normalizer文件。

完整冻结Qwen-Flow基础模型仍是独立不可变资产，调用方须实际核对其hash；q加载器核对包内声明与共同协议的base hash。只有小策略及CENTER的固定均值在q包内。

## API与内容

```python
from starVLA.rl.qwen_flow_exploration.latent.deployment import (
    export_q_package, load_q_package,
)

metadata = export_q_package(checkpoint, protocol, output_directory)
policy, metadata = load_q_package(package_directory, protocol, device=base_device)
```

返回的元数据字段固定包含`arm`、`training_seed`、`qualification`、`policy_sha256`、`checkpoint_state`、`base_model_sha256`、`protocol_sha256`、`policy_identity`。`policy_sha256`覆盖全部q state，包括CENTER均值buffer，可以直接用于行为/评价身份。原训练身份和源码映射以hash保留。

目录严格只有：

- `q_weights.pt`：有限FP32 q张量；不含基础模型、optimizer、scheduler、数据游标或候选库。
- `identity.json`：sealed配置与来源；CENTER统计身份内联，无原normalizer路径依赖。
- `COMPLETE`：sealed原子发布记录，绑定身份和权重文件hash。

加载器检查文件与tensor hash、严格state keys/shape、7168维合法特征接口、8×3 root、std边界、原生CPU噪声namespace、K10/shift5、base/common身份和完整预算边界。CENTER均值直接从`feature_center` tensor恢复，核对TRAIN-manifest及均值hash。它不调用训练normalizer加载函数。返回策略处于eval且全部参数冻结；加载过程不消耗全局CPU RNG。

未完成目录及额外文件/张量被拒绝。资格权重默认拒绝导出/加载为正式结果；只有工程测试可以显式使用`allow_qualification=True`。该标记不会被改成正式策略身份。

推理仍使用已有condition和原ODE，例如：

```python
import torch
from starVLA.rl.qwen_flow_exploration.latent.policy import condition_features
from starVLA.rl.qwen_flow_exploration.latent.gaussian import sample_roots
from starVLA.rl.safe_flow_improvement.contracts import seed_for

# Run q outside Qwen's BF16 autocast scope: q and its Gaussian parameters stay FP32.
with torch.no_grad():
    parameters = policy(condition_features(condition)).detached()
    rng = torch.Generator(device="cpu").manual_seed(
        seed_for("native_qwen_sourceflow_ODE", inference_seed, scene_token)
    )
    root = sample_roots(parameters, 1, rng)[:, 0]
    action = frozen_base.policy.sample_ode(condition, root)
```

没有scorer、teacher或候选选择进入actor。确定性均值root读出继续单独命名，不替代随机主读出。

## 已执行的导出与验收

真实L1正式包已导出到`/mnt/project/qwen-flow-three-routes-20261024/deployment/q_L1_seed42_v1`，policy hash为`3145cce82090ca0e46a914cd6d48a036d3f1260340bbb3b8e1a1937fc06f3c41`。CENTER资格包明确位于`qualification/deployment_packages_v1/CENTER_qualification`，不得用于正式评价。元数据见[包索引](pure_q_deployment_index_v1.json)；权重不提交git。

执行过的CLI：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
  /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.qwen_flow_exploration.latent_export \
  --protocol /mnt/project/qwen-flow-three-routes-20261024/common_protocol.json \
  --checkpoint /mnt/project/qwen-flow-three-routes-20261024/runs/L1_seed42 \
  --output /mnt/project/qwen-flow-three-routes-20261024/deployment/q_L1_seed42_v1

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
  /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.qwen_flow_exploration.latent_deployment_validate \
  --output /mnt/project/qwen-flow-three-routes-20261024/qualification/pure_q_deployment_CPU_v1.json
```

[CPU验收](pure_q_deployment_CPU_v1.json)使用实际L1和实际CENTER资格权重，通过五项检查：

1. 导出后删除临时父checkpoint和normalizer副本；原实验资产不删除、不改名。
2. 显式禁止包外数据读取，loader只读取上述三个文件，原normalizer加载函数不可用时仍成功。
3. 与原训练构造器逐值比较输出、B1/B16、相同CPU创新下root及完整联合logprob；真实tiny source ActionDiT的10步shift5 ODE及物理解码完全一致。
4. 部署没有autograd，CPU RNG不变；资格默认仍被正式加载拒绝。
5. 即使人为更新全部seal使文件hash自洽，额外非法q tensor key仍被严格拒绝。

这证明小q包的独立加载和输出一致性，不是新的规划分数或完整基础模型GPU评价。没有新增GPU、rollout、评分或训练。
