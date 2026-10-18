# 唯一候选：scene center + rollout-global scale normalization

状态：**`NOT_APPROVED`**。仅为独立设计提案；没有实现、运行单元测试、读取全量反馈、加载模型或开始训练。只比较同一C1-PDMS目标的legacy与这一项新适配，不增加per-scene std、fixed-scale、batch/G大小或系数搜索。原合同§12.2波次2C允许有限稳定性扩展、最多两种更新法；它不替代新variant的独立正确性、实测profile和最终批准。

固定3584已知16/32组非零且全部EP-only；同人口policy/BC梯度范数比0.0613169%。这使“原始奖励幅值与BC相对尺度是否值得标准化对照”成为有依据的问题，但既不能诊断原适配为bug，也不能推出1600倍权重、全程梯度关系或规划收益。完整TRAIN账本幅值结果尚未交付，以下定义在见到该结果前锁定。

## 1. 官方依据与明确的新适配

官方FPO固定SHA `418c2554f7cd22d52e14c07d951280929d73bf2f`：

- [gridworld/models/fpo.py::FPO.learn](https://github.com/akanazawa/fpo/blob/418c2554f7cd22d52e14c07d951280929d73bf2f/gridworld/models/fpo.py#L158) 对完整收集rollout的GAE使用 `(A-A.mean())/(A.std()+1e-10)`，之后才shuffle/minibatch。`main.py` 默认positive_advantage=False、timesteps_per_batch=2048；实际population是该次收集的全部transition，不把配置2048误当每次必定精确2048。
- [gridworld/models/ppo.py::PPO.calculate_gae](https://github.com/akanazawa/fpo/blob/418c2554f7cd22d52e14c07d951280929d73bf2f/gridworld/models/ppo.py#L197) 返回float32 GAE。[requirements.txt](https://github.com/akanazawa/fpo/blob/418c2554f7cd22d52e14c07d951280929d73bf2f/gridworld/requirements.txt) 固定torch==2.2.2。两文件已本地工作文件、固定git对象、官方raw逐字节核验；新hash见同名JSON。
- PyTorch v2.2.2 tag解析为commit `39901f229520a5256505ec24782f716ee7ddc843`。其[官方torch.std文档源](https://github.com/pytorch/pytorch/blob/39901f229520a5256505ec24782f716ee7ddc843/torch/_torch_docs.py) 明确默认 `correction=1`，即N−1分母。已通过官方GitHub API解析tag并有界读取固定commit源码；未声称在本地安装/执行了torch2.2.2。
- [FPO论文](https://arxiv.org/html/2507.21053v2) Eq.5–8提供优势加权surrogate及CFM平均，A.1.4讨论优势处理，Table A.2/A.3记录归一化设置。它不证明导航group优势、BC锚点和本候选的收益或安全性。

本项目训练source `6714b1e9fb2e0c8059b90b8b4d48cd40796fe345` 的 `FPOConfig` 明确注册 `within_scene_center_only`，`group_centered_advantages` 保留原float32奖励幅值。官方使用约2048个MDP transition的GAE，本项目通常只有32个scene×4条完整轨迹=128个contextual-bandit action，且每四条共享scene。两者的采样依赖、数量、baseline、critic、reward单位与BC项均不同。本候选只借鉴整批标准差缩放，**并不恢复完整官方训练配方**；N−1也不让相关候选成为独立样本。

## 2. 唯一定义与固定常数

方法名 `scene_center_rollout_std_v1`；对照 `legacy_scene_center_v1`。两者只用原C1 canonical PDMS，policy_weight=1、BC_weight=1，G=4，MC=50，原10步Flow与冻结Qwen。

令u为一次old-policy optimizer边界，b为本次实际scene，g∈{0,1,2,3}。先完整复用原运算：

\[
a_{ubg}=\mathrm{FP32}(r_{ubg})-
\operatorname{mean}_{g}^{\mathrm{FP32}}\mathrm{FP32}(r_{ubg}).
\]

标准化population \(\mathcal I_u\) 是**同一old-policy identity、同一完整optimizer rollout、全部DDP rank的所有已完整有效G4候选**。包含零优势组、合法零分组和有效全失败组；不按事件、优势符号、非零与否、log或微批筛选。没有跨update/epoch历史、running/EMA统计、全TRAIN拟合的固定std，也没有CFM Monte Carlo维度、8个轨迹点或10个Flow步进入N。

\[
N_u=|\mathcal I_u|=4B_u,\quad
\bar a_u=N_u^{-1}\sum_{i\in\mathcal I_u}a_{ui},\quad
s_u=\sqrt{\frac{\sum_{i\in\mathcal I_u}(a_{ui}-\bar a_u)^2}{N_u-1}},
\]
\[
\widehat a_{ui}=\mathrm{FP32}\left(\frac{\mathrm{FP64}(a_{ui})}{s_u+10^{-10}}\right).
\]

常数与语义在此锁定：correction=1和epsilon=1e−10来自所选官方源码；epsilon放在sqrt后的std外，不改为variance epsilon或clamp_min，不通过TRAIN/DEV选择它。分母为centered advantage的std，而非raw reward的std，避免不同scene的平均难度进入尺度。**分子不再减全局均值。** 所有统计、分母和新优势均detach；输出回FP32，以保持现有FPO loss的FP32归约。FP64统计是本候选新增的跨rank数值实现边界，并非声称逐bit复制官方float32 CUDA std。

保留原全常数组为精确零：其原a全0，除任意有限正分母仍为0；有效全rollout的a全0时s=0，结果仍全0，原BC继续。这不制造安全区分，也不修复稀疏支持。若N<2、无完整有效scene、任何输入/统计非有限或出现非零a但s=0，判该rollout不可用并在backward前集体失败，不用默认分母掩盖错误。

再次全局center的解析反例：float32 reward组 `[1,1,1,1−2^−24]` 在round-to-nearest-even求四项mean时得到1，原a为 `[0,0,0,−2^−24]`；另一个全常数组a为四个0。全部8项均值为−2^−27。若分子再减该均值，原零组的每项会变成+2^−27，随后还可能被小std放大。这里只给解析反例，未新增执行测试；未来实现必须用实际目标dtype把它作为反例测试。分母计算std所需的减均值不会写回分子，因而不产生该伪信号。已有非零组中的float32中心化残差也可能被放大，仍需完整TRAIN精度证据，不能认为本规则消除了全部舍入风险。

## 3. DDP、微批、mask与恢复合同

建议唯一统计协议为：在完整feedback/freshness guard通过后、原 `train_flow_policy.py` 的 `group_centered_advantages` 之后、任意训练microbatch切片之前，汇总detach后的完整本地a。第一阶段在每rank只做非collective的本地检查/准备，用try捕获失败而不提前抛出；所有rank参加同一个失败投票，任一失败则全体在统计collectives前终止。不能仅把一个可能先抛错、又包含all_reduce的helper包进现有collective_call：否则其他rank可能已进入不同序collective。身份对齐与每个可能失败的阶段也遵循先准备、共同确认、再下一collective的顺序。

通过上述共同确认后，all-reduce整数N和FP64 sum得到全局均值；第二次归约FP64的局部 `sum((a−global_mean)^2)`，得共同s。使用中心矩的两遍归约，避免 `Q−S²/N` 的消减误差。FP64 all-reduce/归约次序的数值误差必须实测，不能声称更换rank拓扑后bitwise相等。每个rank同意并保存同一统计identity，之后所有微批复用同一个分母。

| 边界 | 候选的唯一处理 |
|---|---|
| 完整32scene | N=128，std分母127；含所有零a，不能仅用64个非零候选 |
| 原尾batch24scene | N=96，std分母95；policy动作均值分母96，BC scene均值分母24，不补齐到32/128 |
| rank本地无条目 / padding mask全空 | 统计helper贡献N=0、sum=0、M2=0，仍参加同序collective；policy图连接零。该数学支持不自动批准原trainer未支持的空scene rank执行；当前正式拓扑继续满足原rank/尾batch合同，否则先拒绝 |
| 一个实际候选反馈无效、缺失或非有限 | 保持原require_complete：全rollout在backward前失败，不能从N中删去它、部分group重算mean或仅做该更新的BC |
| 全局valid mask为空 | 失败，不产生std NaN后填零，也不提交optimizer更新 |
| 全部有效但a全零 | N仍计全部候选，s=0、新a全零；原BC照常，原有效scene与曝光不变 |
| 跨微批 / 梯度累积 | 标准化一次；不能每microbatch重算std、平均各rank std或再除accumulation_steps |
| MC50 / mc_chunk | 一个候选只有一个a和一个population计数；MC50仍仅估计CFM差的均值，分块不改变该均值或全局统计 |
| DDP loss | 保留本地surrogate和×world/N再由DDP平均；全局std已归约，不能额外乘/除world。BC原local_scene/global_scene缩放不变 |
| 异常与恢复 | collective错误先同步；不能让某rank跳过collective。每边界记录method、N、sum、M2、s、epsilon、zero/nonzero数、原a及新a hash、policy/rollout/reward identities；未消费feedback恢复时全rank重建并核对同一统计，不复用别的update的scale |

原 `fpo_ratio`、逐MC差clamp±3、MC50均值、ratio clip0.2、GT BC Beta/repeat8、optimizer/gradient clipping及其精度均不变。零a不触发新的CFM/BC计算跳过或RNG消耗更改。现有configuration、numeric/target合同、receipt与批准hash都必须新增variant身份；不得让旧配置的缺省值静默切换到新方法。此处没有给出实现patch。

## 4. 完整TRAIN结果何时足以支持有限验证

full-ledger实现 `dd413aea` 已保存每boundary的 `centered_A_float32`、原G4排序、分项支持、舍入残差上界和身份。最终聚合的全程abs(A)分位数**不能替代每update的s分布**：不同update不共享normalizer。若这些紧凑输出完整，可从它们做有界只读补充；本提案不自行读取或追加计算。

进入实现/CPU正确性审阅前，应提交以下固定证据，全部保留原重复曝光、唯一scene、唯一log和版本窗口 `[0,1024)`、`[1024,2048)`、`[2048,3072)`、`[3072,4096)`：

1. 4096个消费边界及全部payload/receipt通过独立核验，131,064 scene呈现、524,256候选呈现与原C1 80,933非零组交叉核对。错误、尾batch、空事件和合法零分不得用改分母消掉。唯一scene/log不是独立冻结策略事件概率。
2. 每update的N、zero/nonzero group数、sum(a)、sum(a²)、s、max(abs(a))；对全部update与非零update分别给s和来源固定的 `1/(s+1e−10)` 分布。分位点沿用现分析预登记的0/.25/.5/.75/.9/.99/1，不按3584、最大gain或DEV另选分位/批次。全零batch单列，不把其数值1/epsilon当作实际policy放大量。
3. 精度支持：列出s=0、0<s≤epsilon、非零a但s=0、非有限、全常数组非零等全部异常；保留原score float64→float32、中心化残差与canonical重建误差。用现有每boundary最大舍入残差δ给出 `N*δ²` 对 `sum(a²)` 的保守比较；若不能排除舍入主导，标为UNRESOLVED，不把界过松误说成已证明舍入主导，也不临时调大epsilon来通过。需要的更细核验必须另获有界只读授权。
4. 安全风险分层先锁定义：`safe_EP_only_update`要求至少一个非零组、全部非零组均EP-only且D/C/T事件全空；另独立记录`D/C/T_event_present_update`及`D/C/T_metric_discriminative_update`（至少一个group的对应终局分项变化）。这些flag允许重叠，逐项给实际分母，不能把有安全事件但该组零a说成已有policy监督。比较这些population的s、scale、原优势能量、unique scene/log与四时间窗覆盖，明确EP-only批次是否得到更大scale。

足以考虑**有限profile**的证据是：有效非零信号在四个既定窗口均存在、3584不是唯一来源，数值/归约问题已解释；按update而非只按全程RMS确认原优势幅值及其变化，并已完整呈报EP-only与安全支持之间的不同缩放风险。幅值统计只回答代表性、数值可信度和成本，不证明梯度比例或收益。没有从官方推导出的“安全gain上限”，所以不编造一个阈值来自动批准，也不按结果添加gain cap、换epsilon或选固定倍数。

若只有全空/数值残差信号、数据身份不完整，或上述按update/安全分层仍缺失，则停止在设计阶段。安全分层分母为0时，作用只能暂限于已有进度信号；该证据不能直接放行正式安全/规划收益对照。任何通过都只是值得交独立正确性和效率审阅，尚不是正式训练批准。

## 5. 可以预期的数学变化与风险

冻结同一rollout的a与统计时，令c_u=1/(s_u+epsilon)>0。忽略最终FP32舍入，单个update的policy surrogate与梯度都严格乘c_u，clip分支、优势符号及同batch相对gap不变；old=current ratio不受影响，BC梯度不变，policy与BC的cosine也不因此改善。小例可验证这点，无需把“放大梯度”本身包装成新效果证据。

**不同update的c_u不同。** EP-only的小方差update可能被放大得比出现稀有责任碰撞的大方差update更多；这改变它们相对BC和彼此的训练权重。它不保持全训练分布的绝对误选代价，不确保安全优先，也不保证更高PDMS。即使同batch相对gap保持，跨batch安全/进度权衡、Adam矩估计、gradient clipping与后续on-policy采样都会改变。c_u依赖该次采样的奖励，因此不能称无偏policy-gradient修复。

将零a组纳入population使少量非零候选在较稀疏batch中可能获得更大的归一化系数；surrogate仍按全部N平均，不能进一步推出不同batch之间的整体policy梯度必然更大，还依赖活跃占比、参数Jacobian与方向抵消。这与冻结同一batch时前述正标量关系不同。它不增加事件信息，反而可能放大微小但真实或数值性的进度差。官方较大transition rollout与导航128条相关候选的std稳定性不同；correction=1并不校正相关性。保持s+epsilon规则时，scale可很大，但centered标准差约束与归一化后各候选的有限幅度不是“训练安全”保证，尤其需检查残余全局均值与小方差、稀有事件造成的heavy tail。

当前old policy每optimizer boundary刷新、maximum_reuse=1；old/current在同一更新前权重和保存CFM样本上以eval计算，初始ratio约为1。本次唯一optimizer step之前，clip0.2通常不激活；同一批没有之后的surrogate更新替它约束step后的偏移。因此保留clip0.2不提供实际更新后的trust-region、KL或安全保证。候选不增加epochs、reuse或额外采样来改变这个既有定义；profile的更新前ratio/clip统计也不能被当作更新后偏移小的证据。

## 6. 唯一对照与后续审阅，均未启动

仅两种更新法：同C1 reward的legacy center-only与本候选。共同S0、同seed/训练人口、原可训练参数、BC曝光、batch/micro/累积、4096既定探索时钟、warmup、优化器、MC50/G4/Euler10及最终评价合同；精确执行预算和源SHA由Lead在独立profile后冻结。随着参数分化，各自产生自己的on-policy轨迹/评分；不能把完整历史C1账本当新variant的永久on-policy训练数据。已完成C1能否作为严格legacy控制须独立核验等价性，不能因为存在就默认可复用，也不默认重复训练它。

未来必要实现级正确性检查（未由本提案作者执行；独立数学审阅另出receipt，不能替代真实实现验收）：官方torch2.2.2 correction=1/epsilon的单批参考；分子再center反例；全零/合法零分/部分失败/全空mask；单rank在本地准备阶段失败的共同退出与collective顺序；尾24与不均rank分片；全batch对分片DDP及不同micro/MC chunk的loss和梯度等价；固定正scale的surrogate梯度关系；新A无reward/统计autograd；恢复统计身份与新旧配置隔离。FP64统计容差须由独立参考的数值界和固定dtype决定，不按真实训练表现选。

正确性通过后，另登记有界SMOKE/PROFILE、权重弃用；原合同上限≤256更新，具体采用现有完整窗口协议由Systems冻结，不新增profile超参搜索。profile实测完整rollout→评分→更新链的通信/吞吐/RSS/GPU成本，验证全零行为、tail、normalizer与raw/weighted policy/BC范数、clip比例、nonfinite和恢复；没有批准就不执行。最终正式批准仍绑定整合SHA、method配置、数据身份、精度、拓扑和资源租约。

正式对照风险指标预先保留：全人口PDMS、NC/DAC/TTC、EP/Comfort、合法零分、有效性/失败、原评价合同的双向退化/改善及停车低进度行为，按配对scene与log报告；不能只看归一化loss或总体PDMS。训练中保留上述按update的安全/EP-only scale分层、原优势和反馈支持、policy/BC尺度、ratio/optimizer梯度clip及非有限计数。缺少安全事件分母不是无安全退化证据。按共同预登记终点统一评价，不根据DEV选择epsilon、系数、epoch或分支，不为早期分数延长一方。资源监督、身份或数值失败立即按原停止合同处理，保留native状态；不自动调参重试。

本提案不修改任何reward、trainer或模型，也没有签署训练批准。它只登记一个可以被完整TRAIN证据否决、并可交独立审阅的尺度适配候选。
