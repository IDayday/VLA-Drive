# 同人口梯度与官方 FPO 优势尺度核验

状态：`SOURCE_AND_EXISTING_RECEIPT_REVIEW_ONLY`。本补充只读源码、小回执与既存数学结果，没有执行新梯度、模型、评分或训练。原 `PLAN.md` / `analyze_existing.py` / `RESULTS.json` 保持不变；其中等待梯度的字段是原始执行时状态，以下是随后收到的证据。

结论：**官方选定 FPO 实现有优势标准差归一化，本项目明确注册了不除标准差的 scene 内中心化适配。两者的优势尺度并不一致。** CFM 坐标 mean、MC mean、完整候选平均与BC repeat8的mean均有明确源码，不支持把本次范数差直接判为漏乘4、8、32或50。真实差距已在同一batch测到，但其因果份额、全程代表性和可用训练系数尚未确定。

## 固定官方源码与论文

官方仓库固定 SHA `418c2554f7cd22d52e14c07d951280929d73bf2f`，文件的本地工作副本、本地git对象和官方固定SHA raw内容逐字节一致，hash见 `PRIMARY_SOURCE_PINS.json`。

| 来源 / 符号 | 核实语义 |
|---|---|
| [gridworld/main.py::main](https://github.com/akanazawa/fpo/blob/418c2554f7cd22d52e14c07d951280929d73bf2f/gridworld/main.py#L120) | 默认 `positive_advantage=False`，`num_fpo_samples=50`，`timesteps_per_batch=2048` |
| [gridworld/models/fpo.py::FPO.__init__](https://github.com/akanazawa/fpo/blob/418c2554f7cd22d52e14c07d951280929d73bf2f/gridworld/models/fpo.py#L35) | 缺省positive_advantage也是False |
| [FPO.learn L158–166](https://github.com/akanazawa/fpo/blob/418c2554f7cd22d52e14c07d951280929d73bf2f/gridworld/models/fpo.py#L158) | 先算整次rollout的GAE；默认分支使用 `(A_k-A_k.mean())/(A_k.std()+1e-10)`；可选True分支是softplus，不能当成本项目采用的默认 |
| 同函数 L169–192 | 标准化在shuffle/minibatch之前；`mini_adv=A_k[idx]`。这是整个收集batch的GAE标准化，不是每scene四候选分别除std |
| 同函数 L215–224 | 每MC差值clamp±3、对MC取mean后exp，clip/min目标对动作取mean；此选定GridWorld actor没有本项目额外的expert BC锚点 |
| [DiffusionPolicy.compute_cfm_loss](https://github.com/akanazawa/fpo/blob/418c2554f7cd22d52e14c07d951280929d73bf2f/gridworld/models/diffusion_policy.py#L119) | `mse_loss(..., reduction='none').mean(dim=1)`，按动作坐标平均 |

[论文 v2](https://arxiv.org/html/2507.21053v2) Eq.5是优势加权clip surrogate，Eq.6–7是CFM差的exp与MC平均；Eq.8写平方L2和，选定官方代码实际取坐标mean。Appendix A.1.4讨论优势处理，Table A.2/A.3记录Normalize advantage=True。论文兼容不同优势估计，不意味着导航center-only与官方GAE标准化相同；论文也未给出“原BC锚点＋该导航surrogate”的权重1:1充分性保证。

## 本次实际训练适配

本批所属run的 `identity.json` SHA256为 `15535618f98c13c9b5a5929607a67012fbc5bc88f394ab353f70ebec37655930`，记录训练source `6714b1e9fb2e0c8059b90b8b4d48cd40796fe345`。下述五个源文件与本报告基线 `336e41cd` 逐字节相同，raw hash补入 `PRIMARY_SOURCE_PINS.json`。本核验没有把梯度receipt内的语义fingerprint误当作raw文件hash。

| 本地文件 / 符号 | 实际语义及尺度边界 |
|---|---|
| `starVLA/model/modules/flow_posttraining/fpo_objective.py::FPOConfig` | advantage锁定 `within_scene_center_only`，更改语义会被拒绝并要求另注册适配 |
| 同文件 `group_centered_advantages` | `A_bg=float32(r_bg)-mean_g(float32(r_bg))`，G=4；不除std，无critic/GAE。全常数组保留为零优势 |
| 同文件 `cfm_coordinate_mean` / `fpo_ratio` | H×C=8×4=32坐标取mean，MC50取mean，和选定官方代码的归约形式相同。相对论文Eq.8的sum确有1/32尺度，但这是显式选择，不是此处新发现的漏乘；改sum还会改变exp温度与clamp，不能无条件化成一个固定学习率修补 |
| 同文件 `fpo_surrogate`；`tools/st_world_policy/train_flow_policy.py::FlowTrainingModule.forward` | 本地动作和乘world/global_action_count，全球分母4B_scene；DDP平均后为全部原候选均值。等候选数下不是把scene均值额外除4。累积microbatch后不再除一次累积数 |
| `starVLA/model/modules/flow_posttraining/policy_snapshot.py::DiTFlowPolicy.bc_anchor`；`action_model/GR00T_ActionHeader.py::FlowmatchingActionHead.forward` | 原Beta(1.5,1)、s=.999、重复8次；最终loss是所有元素mean，再按本地scene数/global_scene_count缩放。因此repeat8不是BC损失乘8；BC与policy的time/dropout/targets本来不同 |
| `tools/st_world_policy/flow_training_config.py::validate_config`；`FlowTrainingModule.forward` | BC系数固定1，policy系数显式配置；真实C1 run二者1:1，目标是 `BC + policy`。这是导航适配，选定官方GridWorld actor无此相对权重 |

从源码只能得出：官方std normalization存在而本适配未采用；原始奖励幅值被保留，这可能使很小的EP差异对应很小的A。固定3584已保存的A平方均值是 `4.6467667008665003e-7`（128候选，包含64个零A），但它不是参数梯度大小的完整解释。MC平均是在估计Eq.7期望，不是应当补乘50的训练错误；坐标/动作平均也不能靠范数比反推为错误。

若在完全固定的A、ratio、参数与运算下仅将A乘某个正标量，clip surrogate和其梯度会按该标量缩放；加入BC、Adam预处理、gradient clipping和多次更新后，不能从单batch范数比推出实际更新或收益。每scene单独除std又会改变各组相对权重，尤其极小差异与稀有事件组；它不是把官方整批GAE标准化“原样接回”。本报告不指定epsilon、作用域、系数或新reward。

## 已收到的同人口真实梯度

仅读取 `/mnt/project/st-world-policy-20261016/audits/consequence/gradient_balance_boundary3584_v3/RESULT.json`，298,025B，SHA256 `7ffd159b3dda244b40dec03ad09c3142f47bc50c395d1be70bda9127c8203a2b`。该文件状态 `DIAGNOSTIC_COMPLETE`，数学source由Lead给定为 `83679937fe9702be39f0184154fdf9ebdc4eb98a`。官方COMPLETE的SHA `e1d40b12cd35e022be2ecd309b7f7195c5b53133bf66c724f13a362e850e863f` 与本分析完全相同，8个BC和8个policy microbatch，原系数1:1。

| 参数集合 | BC L2 | policy L2 | policy/BC | cosine |
|---|---:|---:|---:|---:|
| 全部可训练参数 | 0.1620324978 | 0.0000993533753 | 0.0613169% | −0.0130164 |
| DiT | 0.0545407061 | 0.0000408923735 | 0.0749759% | +0.0525445 |
| qwen_proj | 0.0387364569 | 0.0000298905222 | 0.0771638% | −0.1449560 |

因此“此batch的policy参数梯度尺度远小于BC”有真实依据，且此batch的区分信号已独立确认为EP-only。整体cosine接近正交而稍负，不能笼统说所有模块方向冲突，DiT项为正。文件中一阶Euclidean组合下降对policy surrogate的变化为 `+1.9967268779621644e-7` 每单位学习率，也只描述该局部数学量。

文件报告独立BF16 backward后FP32 CPU累加、float64内积；caller RNG和原权重不变，优化更新/新评分均0，明确 `exact_ZeRO2_update_claim=false`。不能将该分解称为原ZeRO2实际Adam更新，也不能推广全4096步。Lead另报告：子程序exit0后外部supervisor的live RSS采样发生退出竞态，监督状态保留 `STOPPED_NO_AUTO_RETRY`，cleanup live_owned=0。数学文件原子完整与运行监督是否合格是两项判断；本Agent不把监督状态改成PASS，也不批准为修监控而重跑，最终可用性由独立receipt审阅界定。

唯一下一项仍是 `NEXT_PROPOSAL.md` 的完整C1既存TRAIN账本只读审计，增加原优势幅值统计即可。先判断稀有安全/异时信息的实际支持及EP-only代表性；不从0.0613%自动产生1600倍系数、G8、C5或DFQL训练。
