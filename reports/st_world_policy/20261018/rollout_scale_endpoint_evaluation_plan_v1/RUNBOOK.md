# 唯一4096终点的DEV并行评价准备

状态：**`WAITING_CHECKPOINT` / `PROPOSED_EXECUTION_NOT_APPROVED`**。新组注册已存在，但本次检查时正式run目录及4096模型尚不存在；`ENDPOINT_STATE.json` 保留实际观察，checkpoint tag/hash为null。不得以最新指针、SMOKE/PROFILE权重或早期checkpoint替代。训练runtime仍为 `751fcfa2b41549ae16f7b3204aef8dd18b912016`，没有源码修改、GPU推理、评分或新实验。

本准备只服务 `C1_rollout_std_fulltrain_seed42_official_4096_v1@4096` 减历史 `C1_fulltrain_seed42_official_4096_v1@4096`。固定DEV1696 scenes/16 logs、原单轨迹Euler10、4秒/8点、FP32 master eval/TF32 off、scene seed42、原canonical reference/context；统计、阈值和科学问题沿用已封存 `COMPARISON`。旧C1不重跑。

## 已实际核实的入口与资产

在runtime-v2目录使用真实Python3.10环境调用了 `export_world --help`、`score_world --help`、`evaluate_campaign --help`，均exit0，完整输出与receipt已保存。三次共10.7145 child CPU秒、10.9967 wall秒，限制2核，最大采样进程树RSS647,544,832B，小于2GiB；CUDA不可见。参数均来自实际CLI，不猜测merge或分片参数。

| 资产 | 本次结果 |
|---|---|
| DEV `SELECTED_SCENES.json` | SHA `4ce40893281f3cf246989cbbac6bbd80c40c06ec937a4405a5a7c353c6dd0573`；502,548B；原顺序完整1696唯一token/16logs |
| 固定 `DEV_METRIC_INDEX_PINNED_v1.json` | SHA `60fc773ab05c442db33ab1f2237c9ee8cf57a0de7464a2b76232444325b65266`；619,043B；1696条与manifest逐token/log一对一 |
| 旧C1 score COMPLETE | 当前小receipt hash与原记录相同，requested=valid=1696，invalid=unresolved=0；score identity `82992f77f75b2c52a006e9bf6ea63c49414daacfa5590454c55b293367c67742` |
| 旧C1 CSV | 原COMPLETE绑定SHA `7c611333a2b4d3b53ac363008843bf78190a3b80c501e34afcbaa579fdcd468d`；本次不读逐scene指标，最终成熟analyzer仍重新核对真实CSV/ledger/输入身份 |
| exporter/scorer/analyzer | runtime751与原6714三文件完全未变；strict Flow checkpoint reader已包含显式normalizer身份，未来仍须核验实际native资产 |

`FIXED_INPUTS.json` 保存metadata核验，不复制scene名单、标签、图像或cache payload。context文件和地图沿用原路径；不新建metric cache，不把训练G4的EP参考用于部署单候选。

## 并行执行提案与路由

拟议 **4个独立单GPU exporter rank**，各处理原manifest的 `rows[rank::4]`，每rank424 scenes、每scene仍batch1/一条轨迹。不是训练DDP，也不是四候选择优。原 `scene_noise(token,42)` 仅由token/seed生成CPU FP32噪声；rank不进入种子。所有rank共享同一output root、注册、真实checkpoint tag、源码、环境与world_size=4。

早先COMPARISON记载export world1，本次明确提出**仅执行拓扑amendment：world1→world4**，不改封存原文。`ANALYSIS_PLAN_DRAFT.json` 显式标注未批准。需Lead及独立正确性/资源审阅确认身份、单scene路径、共享写入、CPU/RAM/IO与资源预算后才可使用；不能据源码支持分片就声称真实四GPU成本或输出等价已测。不能把旧world1的C1 identity改写成world4。

当前策略与本campaign授权都允许：vla-zt、rl-zt2、Tsinghua zt/zt2/zt3/zt4/zt5/zt6，仍须保护实际工作及所有已撤销主机。Systems已封存的 `active_writers_v1/ACTIVE_WRITERS.json`（agent `479c8fcd` / Lead `211dbcac`，SHA `4fd0abb12861cdd8d10b817ecacd85d765e876e30b647eae3e7dbcbe8d518eb2`）在18:08 UTC观察到Tsinghua zt6八卡显存/利用率均为0、无可见GPU进程。本提案据此具体填入 **SSH `training-vlawm-tsinghua-zt6`、GPU0/1/2/3对应rank0/1/2/3**，实际GPU UUID和lease仍为null，最终checkpoint到达时按fresh admission绑定；快照不是预约。Tsinghua zt上的8卡既有训练和zt5的GPU0–3 PROFILE均为受保护工作，共享挂载的剩余磁盘不能跨host累加。`RESOURCE_SNAPSHOT.json`只保存小型汇总与来源pin，不读取或提交原始PID inventory。

若启动时该host不满足准入，可由Lead在允许host重绑同一W=4的四rank，先由Systems确认每host环境/路径/IO一致。正确SSH别名不带 `-worker-0`；初次worker DNS失败只说明连接别名错误，不说明host不可用。本次不重复现场核验，也不占用任何GPU。

4rank是一项执行提案，不搜索W=1/2/4/8速度矩阵。若准入时4rank无合适资源，Lead可继续使用原world1执行合同；第一次导出前只能封存一个world_size/output身份，开始后不得换W或混合分片。配置变更不得触发新推理seed、额外候选或旧C1重跑。

历史同Flow FP32 exporter在16scene工程运行中的peak reserved为13,285,457,920B、加载319.57秒；这些只是成本参考，不能当新四rank的保证。每个rank会重复父/native/图像完整性检查和模型加载；多GPU减少scene串行部分，也增加重复IO/RAM。当前实际空卡、GPU UUID、全树RSS上限、并发加载安排须由Systems最终冻结，不能按“4卡”承诺4倍速度。

## 终点出现后的具体顺序

1. **绑定真实终点。** Lead核对正式 `RUN_APPROVAL`、真实run identity及注册，找到精确 `completed=4096` 的完整native目录和 `COMPLETE.json`；保存实际tag、receipt SHA/identity、全部native文件hash/size及source751。不能只按目录名称猜测。成熟Flow reader在真正export时仍严格验证payload并从FP32 masters恢复，不将模型另存为新checkpoint。
2. **封存evaluation registration与route。** 绑定同一注册、新终点、固定DEV/context/metric index、共同score protocol及本执行amendment；按当前资源选4个GPU UUID、每rank至少2 CPU线程、输出命名空间和超时/调用上限。fresh policy、source clean、环境完全一致、可见设备与未知占用、RAM/IO/disk均由独立资源准入检查。本报告不替代该批准。
3. **把 `COMMANDS.json` 的占位符替换为真实绑定。** `__ACTUAL_VERIFIED_COMPLETED_4096_NATIVE_TAG__`、每rankhost/GPU、最终analysis plan均必须实填。4个 `exports` 条目即独立并行命令，cwd统一runtime-v2，`--rank 0/1/2/3 --world-size 4 --max-scenes 1696`。每rank只见分配的一张GPU并用 `--device cuda:0`，保持 `CUBLAS_WORKSPACE_CONFIG=:4096:8`、全部BLAS线程1。不从本准备文件自动调度。
4. **等待四份真实完成回执，随后一次CPU评分。** 4rank均写 `COMPLETE_rank{r}.json` 后，释放export GPU，再执行唯一 `score` 命令：CUDA不可见、4个CPU scorer worker、一个coordinator、同一ledger/output。成熟 `read_complete_export` 校验每rank的token集合、数量、共享identity及预测receipt hash；无须也不可手写cat/CSV拼接。不能对未完整shard先评分后伪造全量COMPLETE。
5. **复用旧C1并统一分析。** 在 `ANALYSIS_PLAN_DRAFT.json` 中仅补新组实际approval/run identity/launch的path与SHA，原C1 binding保留；不要保留null占位。发布新的sealed analysis plan，不覆盖旧plan。新score COMPLETE要求1696 requested/valid、0 invalid/unresolved，随后执行 `analysis` 条目，comparison固定 `C1_ROLLOUT_STD_minus_C1`、stage `COMMON_ENDPOINT`。成熟analyzer会验证两臂真实prediction/CSV/ledger、同observations/environment及每scene相同cache/context/reference。任何不匹配先报错，不为通过而重标旧结果或自动重跑控制。
6. **交付完整风险读数。** 输出原PDMS主差、全部分项、配对log区间、逐log/空分母，以及COMPARISON已登记双向安全/零分/高分/停车读数。成熟analyzer输出的完整 `PAIRED_SCENES.csv` 可支持既定转移汇总；不新增评价、阈值或选最佳epoch。最终报告须保留成本、失败和身份。

## 总预算与防重复

只有一个新endpoint：**1696条单候选轨迹、1696次主要评分，总admission cap=1728**，含全部worker、rank、调用和重试。32是全job剩余completion headroom，不是每rank/worker32，也不是每个scene两次都自动获批。`retry_attempts=1`、评分最多2次进程调用；未知或中断的ADMITTED仍占ledger预算，不删除ledger或换output重置计数。旧C1新增轨迹/评分=0。

沿用原单checkpoint超时边界：export内部7200s、外部7800s+60s清理，每rank最多2次；score内部3600s、外部4200s+60s，最多2次；analysis 2CPU/1800s，最多2次、无GPU/新评分。4GPU最大allocation若全部用尽两次外部上限为62,880 GPU秒（17.4667 GPU小时），是保守上限而非实测或预计成本。Systems可在首次执行前冻结更紧资源上限；不能为早期分数改科学预算。沿用1GiB新输出预留、0新teacher/metric/native复制，超出先报告不静默扩展。

export按rank文件锁防重复，同一token的NPZ和receipt绑定完整export identity；重入只在相同checkpoint/seed/source/environment/W下验证并跳过已完成项，同噪声完成未完scene。顶层IDENTITY/OBSERVATIONS使用成熟immutable原子发布，多rank只允许内容相同。`EVALUATION.lock`保证单score writer，固定chunk4和共享admission ledger跨重启保留。COMPLETE/CSV/analysis均不可覆写为另一人口或另一结果，任何checksum冲突立即停，不把失败当合法0分、也不删scene改变分母。

本次只做metadata/source/真实CLI帮助检查；未读DEV逐scene标签或分数选配置，未读模型/图像/cache payload，未调用GPU、export、score或bootstrap。当前所有执行文件均为准备状态；训练和评价启动仍归Lead。
