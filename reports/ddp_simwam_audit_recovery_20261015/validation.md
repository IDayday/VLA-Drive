# 实现合同、故障与数值验收

## 源码边界

已对正式git archive的3,481个文件逐一比较真实12b8e0d目录，全部匹配；两组identity绑定的21个source_files及run/source副本逐一匹配。完整私有hash清单在 `/mnt/project/simwam-audit-recovery-20261015/source_audit/audit.json`，仓库保存核心hash摘要和实际diff。

正式进程已经结束，不能事后伪造其运行时 `module.__file__`。旧执行路径由实际dispatch cwd、完整archive、run副本与identity交叉核实。新资格运行另外保存每rank实际导入模块的绝对路径和文件hash，位于 `executions/<execution_sha>/runtime_modules_rank*.json`。

| 文件类别 | 12b8e0d→ae8ac5a | 本轮改动性质 |
|---|---|---|
| A actor/sampler/loss/optimizer/data | 核心实现未变；build_hard_manifest完善完整筛选检查，不改变已锁定清单 | actor、vendor、SDE、anchor、优势、optimizer数学、配置均未变；train入口增加事务与账本 |
| B checkpoint/resume | 未变 | 显式阶段、跨rank检查、失败证据隔离、独立attempt、严格迁移 |
| C 导出/推理/评分 | 实际推理与scorer未变 | evaluate_policy仅加invocation/耗时计数和早期worker锁；真实输出对照通过 |
| D 调度/日志/统计 | controller路由、分析和汇总改善 | 回滚归属、成本分母、缺失成本标注、审核CLI |
| E 文档/结果 | 中期快照 | 终点分析与审计凭据，不改历史结果 |

build_hard_manifest的变化不是中途改变数据人口；两组实际hard_manifest hash固定为 `66d9eccf82ecfbdde64e4b678f8af70ddff4525124f374c5de0f18d7f61e391f`。

## 核心合同核对

| 合同 | 状态与依据 |
|---|---|
| 固定F-SFT、冻结Qwen/history与原拓扑 | 原封装、权重hash与既有原始观测验收复用；本轮未动模型文件 |
| 仅96 attention投影、192 LoRA A/B | 原parameter_contract未变；终点192 Adam状态；r16/alpha32/dropout0 |
| qwen_proj/encoder/decoder/FFN/norm/base attention冻结 | 原闭包与八卡资格的冻结hash验证保留 |
| G8/K10、shift5、随机[7,8,9] | vendor数学与正式YAML不变；实际sample/recompute新增直接检查 |
| rigorous SDE/min_std、innovation±5 | 上游分支不变；没有旧rho或learned noise |
| 每维relative logprob clamp后H/D mean、PPO .005 | 全程第1 inner ratio误差0；后续clipping为实际更新，不更改clip |
| 组内中心化、全rank未筛选raw reward std | source_advantages未变；提交阶段用全rank真实reward独立验算 |
| tied只mask PG，source_rank_mean | 源loss未变，无动态补样，无分项优势 |
| x均值MSE anchor50，[.2,1,.5,.5] | 原mean/std/velocity诊断保留；heading-pair average不是角度MSE |
| 一份rollout四次实际AdamW更新 | 八rank连续20k日志、5000collection及192 Adam计数一致 |
| old chain/probability/advantage/reference固定 | 保存摘要，inner1/2/3恢复和下一fresh链精确对照 |
| SOURCE 10:5:2 / CANONICAL 5:5:2 | 完整resolved配置hash锁定；测试只允许variant/progress_weight两项不同 |
| 无BC、GT replay、dual、learned noise | vendor训练活跃分支与配置锁不变；未引入新loss |
| effective gradient clipping=0 | 保留源有效行为，未启用torch clip1 |
| DDP find_unused=True/static_graph=False/bucket25MB | 保留；新资格无bucket rebuild，正常路径逐项精确对齐 |
| 永久SFT reference、原/shift5双读出 | 不刷新reference；最终两种网格均报告，原入口合并导出不变 |
| 同seed/预算/人口/固定20k终点 | 两组终点、数据流与7次padding均核实，无新seed矩阵 |

源配置中的grpo.eval/vis字段没有独立启动源视频评测或可视化；实际0/5k/10k/15k/20k评测由锁定milestone controller执行。

旧64项上游数学对照、LoRA覆盖/merge、原始观测部署、固定bucket与旧inner1恢复均复用，详见[原validation](../ddp_simwam_recipe_20261015/validation.md)；没有重做全仓审核。

## 事务与恢复

- `READY`：没有待更新rollout，update为4的整数倍。
- `COLLECTING`：不可恢复、不得发布COMPLETE/latest。失败保留本地evidence及可获得的partial payload，从之前合法checkpoint统一重放。
- `ROLLOUT_COMMITTED`：所有rank完成评分和全局优势，collection/cursor已提交，next_inner=0。
- `INNER_COMMITTED`：inner1/2/3已完成，保存同一behavior链与固定优势；恢复剩余inner。
- `COMPLETE`：只用于20000终点；文件名COMPLETE仍是发布seal，不能单凭该文件判定阶段合法。

发布前核对run、阶段、update/collection/cursor/inner、rollout字段、行为版本、old概率/advantage摘要、全局std、Adam/scheduler进度。CPU control group在健康启动时创建。CUDA/collective致命错误走本地证据，不尝试新的NCCL全局保存。未实际制造硬件损坏；真实故障验收是可控软件异常。

写入使用唯一incomplete attempt，失败不推进latest、不覆盖旧checkpoint。已有COMPLETE还需文件hash及阶段合同；历史failure标签拒绝自动恢复。扫描的旧SimWAM formal/qualification范围未发现failure目录，零数量照实记录；既有bucket恢复失败证据和本轮失败资格运行都保留。

跨源码恢复必须提供 `make_migration_receipt` 生成的receipt：绑定旧/新完整identity、source_files diff、父checkpoint seal、科学身份与真实正常路径/恢复等价证据。只允许本轮审核过的工程文件变化，不能借新身份跳过旧source_files。

## 已执行测试

全部具体输出及hash见[evidence/qualification.json](evidence/qualification.json)。

| 测试 | 结果 |
|---|---|
| CPU两rank/Gloo：rank0采样前、非rank0部分评分失败、优势提交前 | 拒绝恢复发布，latest不变 |
| 未finalize却伪造committed phase | 拒绝 |
| rank写失败、COMPLETE前失败、遗留incomplete、重复发布 | 无覆盖、无latest推进；新attempt可重试，两rank共同返回 |
| CPU inner1/2/3恢复与日志回滚 | Adam/RNG恢复；重复有效update仍拒绝；原日志和aborted后缀凭据保留 |
| 真实两GPU：before_collect rank0、after_collect rank1、before_finalize rank1 | 均保留update0；实际评分后的局部完整链也不被误发布 |
| 两GPU失败collection统一重放 | 最终8步的Adam/权重/RNG/后续链/loss/gradient norm与连续运行精确一致 |
| 真实八GPU inner2→inner3→8恢复 | 同上精确一致；旧inner1资格不重复重训 |
| 八GPU新正常路径 vs 正式源码已验收基准 | 数学结果及RNG逐项精确；两基准source_files相同，时间/元数据除外 |
| 旧inner1 checkpoint隔离副本→新代码显式迁移 | 真实八卡运行至8步，与旧连续基准精确一致；正式run没迁移 |
| 实际采样mean/std，零LoRA与20k非零LoRA | 7/8/9步mean最大误差2.384e−7，std最大1.542e−9；保存的mean+std×clamped创新精确重建实际x_next；记录不改变RNG或chain |
| 配置与成本CPU测试 | 23项通过：完整字段锁、重复update拒绝、本段delta吞吐、失败成本保留、评价new/reused/loading/worker/job区分、旧缺失值未知 |
| 新评价账本实际原始观测smoke+复用 | 1场景两种网格动作及score与旧SFT精确相同；第二次new0/reused1；两次加载和耗时均保留 |
| 正式结果复核 | 21份完整评价payload/hash/人口/分项/配对结果全部一致；不重新生成 |

采样mean/std记录通过资格包装器捕获实际 `_sde_mean_std` 返回值，并记录原创新；不是仅检查old/current ratio=1。原Gaussian density、源clamped代理、joint与dimension-mean分别命名。实际抽样未出现±5尾部命中，尾部边界沿用原64项小张量oracle；不声称这批真实样本观测了稀有尾部事件。

### 本轮保留的失败

1. 第一次私有资格controller在快照归档尚未完成时启动失败，未加载模型或训练；日志保留，完成归档后才重新派发。
2. `gpu8_recovery_v1`的新检查器错误要求CPU mean与GPU保存值bitwise相等，实际差异1.192e−7～2.384e−7，拒绝了恢复，没有继续更新。改为语义检查绝对容差5e−7，同时保留保存payload的逐bit hash；`gpu8_recovery_v2`重测通过。未改训练logprob、ratio、clip或原概率。
3. 成本单测发现新评价invocation目录需先创建，修复后执行真实原始观测与复用验证。旧正式评价不走新账本代码，不受影响。

4. CPU/mock初版的通用RNG保存探测并初始化了本地CUDA RNG上下文，没有运行GPU模型/梯度；进程均已退出。CPU入口现强制屏蔽CUDA，v4重新通过并记录CUDA_visible=false。

这些失败都是隔离工程验收记录，不能掩盖，也不是旧RL效果差的证据。

## 统计边界

旧metrics不能倒推出所有加载/保存/失败成本。新账本分别报告逻辑有效更新/候选与实际执行尝试；失败invocation包含的全部wall不是其内部纯回滚wall，后者缺少阶段证据时未知。共享GPU worker相加不等于端到端job wall或独占GPU时间。初始化或硬终止发生在可写账本之前的残余开销仍只能记为不可追溯，不能填零。

旧训练progress末尾RUNNING被八rank COMPLETE、原生optimizer及日志终点纠正，但未重写历史文件。已有结果有效性与提分/安全是否成立分别判断。
