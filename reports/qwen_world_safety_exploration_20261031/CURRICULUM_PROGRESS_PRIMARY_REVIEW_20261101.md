当前 ACTIVE 已真正激活：完整1k审计为2000次场景曝光，其中1520次uniform、480次priority；unique为1520，control为2000，同rank/collection仅88/2000场景相同。它改变了训练分布，不是同一场景上的算法对照。更低的训练reward也不能直接解释为能力下降。

完整同预算NAV对照中，500更新差为+0.2294分，区间跨0；1k差为−0.1973分，已有44log区间[−0.3945,−0.0021]。1k修zero82、新增25，净少57个，但固定control高分群体损失贡献−0.6644，超过零分修复贡献+0.5461。这里的NAV仅供研究判断，绝不把其ID整理成TRAIN队列。

三篇最相关的区别：[PAC](https://arxiv.org/html/2608.30528v1)把优势潜力与实际reward趋势分开，再做不确定性分配；[ClosedLoop Countdown](https://arxiv.org/html/2610.39056v1)用固定probe分别控制进度和复习覆盖；[AceGRPO](https://arxiv.org/html/2602.07906v5)更接近当前方差×余量启发式，还带演化buffer。[CoDaPO](https://proceedings.mlr.press/v306/zhou26ah.html)另含梯度重加权，不能把当前clamped动作logp直接当其置信度。Ace作者链接当前仅见README等，不能称已核完整实现。

两个实验必须分开、现在均不启动：

- A：原eligible、75%uniform、次数/TTL全部不变，只乘固定TRAIN log/category级进度因子。饱和rawA0与全fail仍进不了priority，不能声称解决遗忘。
- B：独立新增“曾严格高分→当前固定root退化”的TRAIN rehearsal admission，在总25%priority内限制份额。用小轮换面板验证同类别其它TRAIN样本，避免只记住固定128token。新admission要有正的配额/权重，不能再乘rawA0使其实际不激活；但全fail/tie/PG0时重采仍未必有用，dense或endpoint支持应另做实验。

建议固定256 TRAIN反馈面板，按实际物理log/category聚合，0/500/1500/2500/3500/4500更新做相同seed2101的单ODE；IL70基线兼容时复用。固定IL70高分队列保持，另从TRAIN4096额外场景预封1024确认面板，其结果不喂控制器。这仍是TRAIN确认，不是训练外heldout。反馈新增1280个B1候选相当于5k训练候选的1.6%，却可能多12.8%的Qwen观测prefill，不能说只多1.6%GPU成本。

变化batch的reward斜率混合了数据选择、采样噪声与theta变化。固定面板只消除重新抽root的因素，不能识别单样本重放的因果收益；稀疏/区间不明确时因子保持中性。负进度是检查遗忘的警报，不自动等于更大学习价值。先完成endpoint第8臂，再决定上述独立数据实验，不改initial_v1。
