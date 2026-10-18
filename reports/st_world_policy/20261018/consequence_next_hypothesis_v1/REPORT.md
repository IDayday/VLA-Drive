# C1 boundary3584：真实优势来源与过程信息支持

**固定32 TRAIN场景的16个非零C1优势组全部由EP差异产生；128候选没有D/C/T事件。** 这是这一个实际batch的事实，不能推广成全TRAIN安全信号或过程监督没有价值。

状态：`READ_ONLY_ANALYSIS_COMPLETE_NEEDS_INDEPENDENT_REVIEW`。独立证据Agent完成方案先登记、代码实现与一次CPU运行；没有正式训练批准，也没有训练奖励变更。

| 固定分母 | 实际结果 |
|---|---:|
| 原始rank / scene / log / candidate | 4 / 32 / 32 / 128 |
| 有效 / 无效候选 | 128 / 0 |
| C1非零优势scene / candidate | 16/32 / 64/128 |
| 非零优势组中EP-only | 16/16 |
| 同人口C2 / C3 / C4非零优势scene | 0 / 0 / 0 |
| NC / DAC / TTC / Comfort等于1的候选 | 各128/128 |
| D / C / T事件候选 | 0 / 0 / 0 |
| 无序候选对 | 192（每scene6对） |
| C1实际float32终局tie | 96/192，来自16scene |
| C1原float64终局tie | 96/192，来自相同16scene |
| 安全终局向量tie | 192/192，来自32scene |
| 各tie条件内的非空事件对 | 全部0 |
| 条件于非空事件的时序差异比例 | undefined，分母0 |

其他16scene是全零优势组；未删去。不存在合法零分或全零乘法门控组。未观察到“同终局但不同事件时间”的正例，是因为这个人口没有D/C/T事件，不能解释为时间字段无信息。32scene/192pair内部有相关性，不把pair当192个独立scene，也不构造全TRAIN稀有事件置信结论。

官方分数按原权重精确重建：`R = NC*DAC*(5*EP+5*TTC+2*Comfort)/12`。执行原float32 reward中心化，另将三个带门控的float64分项中心化并保留舍入残差；完整数值见 `RESULTS.json.advantage_value_decomposition`。这个分解刻画奖励值的来源，不是参数梯度份额，也不是实际Adam更新。仅在NC/DAC/TTC/Comfort全部不随候选变化的本批次，才能明确说C1优势仅由进度区分。

这能与正在独立执行的同一C1 boundary3584梯度分解配对解释：其policy项在本批次接受的是进度区分信号。梯度结果尚未由Lead交付，不能据此声称BC强于policy、存在冲突或需要改变权重。道路方向诊断来自boundary4095的另一批32scene，不能混作同一人口。

数据身份完整核验：`REQUESTS.json`128项与结果逐项匹配；四rank rollout_id/slot/candidate顺序、版本3584、policy参数hash、原numeric执行hash、固定参考input、轨迹hash、保存float32 reward均匹配。128个原记录全部保留41点道路flags、8段D/C/T归约、碰撞/TTC原事件、raw与参考进度。原账本3,441,888 bytes，SHA256 `8019dd29b17efa79c1fd1a7211bdf09a0b5be93931db75cd4db95ad9a3fb122f`。

9项合成控制通过：同终局道路事件异时、全空事件、先0.5后0碰撞严重性、TTC projection offset不等于physical time、右闭分箱、常数奖励零优势、float32量化tie、带抵消的协方差份额、物理时刻与Flow步非一一对应。最后一个可解反例让十个Flow步共同写两个物理轨迹点，4秒事件依赖两个点；全部十个参数的解析导数和数值差分都为0.5。只分给一个Flow步会遗漏九条非零作用路径，最近轨迹点也会遗漏前序点。这只否定无条件一一映射，不冒充实际NAVSIM执行导数。

一次实际执行：CPU **1.143397s**，wall **1.109983s**，peak RSS **423,440,384B**；任务输入逻辑读取 **3,703,826B**，JSON输出 **30,506B**。CUDA未初始化，线程1；新增评分/采样/模型加载/真实梯度/优化更新均0。120 CPU秒、32MiB任务输入、4MiB输出的上限均未触及。当前脚本的精确读取账本写入结果；方案前的路径/字段核查另外仅读有界小回执和同一3.44MB反馈，不读取完整训练manifest或metriccache。

唯一后续建议：先预算化分析**C1已完成全TRAIN既存反馈账本**，分别统计重复曝光、唯一scene、policy版本与事件对的分母，补足本批次没有的稀有事件支持。先做读取/CPU/分片预算及身份计划，尚未执行，不新增G8/评分/模型/训练。若全TRAIN显示有充分的终局同分但事件异时样本，才有依据提出过程方法的局部执行方向诊断；目前不从这32scene登记C5或DFQL训练。

代码与重现入口：`analyze_existing.py`、预先落盘的 `PLAN.md`、不可覆盖的 `RESULTS.json`。本提交交Lead安排独立审阅。
