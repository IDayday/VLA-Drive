# 20k终点证据整理：已执行检查，不是新的实验结果

2026-10-17 14:58 UTC，B正式训练已封存16,000/20,000的READY恢复点，原进程继续运行。34/36阶段评价完成，B20k及最终Navtest仍未完成。本次补齐终点公开证据的整理入口，未改训练、采样、损失、评分或现有控制器。

[`package_rl_endpoint.py`](../../scripts/analysis/fm_transfer/package_rl_endpoint.py)复用已经完成的汇总、人口、安全转移、保存批次和退出审计。它要求固定20k COMPLETE、5000份fresh rollout、40000场景曝光、320000新候选、完整Adam/scheduler证据和关闭的成本账本；两个网格、两个评价人口与自己的SFT和15k分别核对，不挑最高点。独立汇总在均分、G+/G−和安全事件上必须一致。

输出是新的人工复核目录，最后才写`COMPLETE.json`；拒绝覆盖既有目录。复制文件保留原hash，人口和梯度证据只移除单个私有scene记录，保留原文件hash及省略数量。没有调用模型、评分器、训练器或调度器。此入口不是另一个实验平台，也不自行提交或推送结果。

## 实际验证

使用已经结束的A组真实20k资产，完成验证及独立目录打包；没有重新运行A。12项检查全部通过，记录见[验证结果](endpoint_publication_validation.json)和[实际资格入口](reproduce_endpoint_publication_qualification.py)：

- A真实终点的四个人口/网格交叉核对，以及完整文件复制/hash/最后发布标记。
- 私有逐scene记录省略，已有输出拒绝覆盖且hash不变。
- 内存负例：错误预算、开放成本账本、READY冒充COMPLETE、错误自身SFT、G+不一致、rank评价身份不一致均拒绝，输出目录不创建；真实源文件未修改。
- 当前尚未结束的B真实输入被拒绝，没有产生B终点包。
- 已提交的B完成核验脚本与实际等待执行的脚本hash相同，后续不能以覆盖不同版本的方式替换它。

A早期退出审计没有记录历史verifier/manifest文件hash；兼容时明确保留这个缺项，并重新核对实际rank身份、评价manifest及独立安全汇总，不补写虚构历史hash。B新审计必须具有并匹配这两个hash。此差异只涉及旧证据schema，不放宽B终点要求。

## 已执行及后续入口

已执行的只读验证：

```bash
python -m scripts.analysis.fm_transfer.package_rl_endpoint \
  --variant A_DDP_TIME --validate-only
```

资格入口还实际创建了独立A证据包，并核对其每个文件hash；详细路径记录在资格脚本中，不能再次覆盖运行。

B既有20k分析owner完成后，使用新输出目录：

```bash
python -m scripts.analysis.fm_transfer.package_rl_endpoint \
  --variant B_SOURCE_ACTION_DIT \
  --output /mnt/project/fm-transfer-20261016/qualification/B_RL20k_publication_review_v1
```

截至本记录，该B成功入口尚未执行，不能由A通过推断B已完成。发布到仓库前仍需审阅实际终点分数、图表、限制与成本；若目标文件已经存在，必须逐项hash相同才可复用，禁止覆盖。尤其`reproduce_B_RL20k_completion.py`已经在仓库存在。

最终Navtest继续由已有唯一owner在两组固定终点及阶段证据齐全后执行。此工具既不提前打开Navtest，也不以DEV超过92替代最终测试。当前16k恢复点不是新增效果评价点，不增加方法、seed、rollout或更新预算。
