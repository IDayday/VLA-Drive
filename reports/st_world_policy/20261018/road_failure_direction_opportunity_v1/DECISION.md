# TRAIN 道路失败人口上的一次有限方向诊断

结论：**值得补一次有界诊断；当前 `BLOCKED_CANDIDATE_CONTEXT_BINDING / NOT_APPROVED`。**
这只是补充“真实道路失败能否被当前固定方向修复”的证据缺口，不是新的训练组，
不改变唯一 C1 scale 候选的审批、资源或日程。当前没有加载模型、轨迹 tensor 或
metric-cache payload，没有新评分、梯度计算、采样或派发。

## 为什么旧结果不足以停止，也不足以直接推进

旧 `road_direction_actual_v3` 保留原32个 TRAIN 场景：全部原 DAC=1，修复分母为0；
7个方向有定义，16个为 `ZERO_OR_NONFINITE_GRADIENT`，9个为 moving-corner boundary/tie。
原错误码不能进一步区分那16个的零与非有限梯度。0d0d19 修正真实时间算术后，32个
几何对照均通过原1e-9公差，不能继续把这25个未定义方向叫作尚未修复的几何实现错误。
负方向在7个有效样本上比正方向和固定随机方向的执行后自定义余量代价更低，但官方
PDMS平均下降0.00330635个百分点，来自EP；没有道路修复证据，也没有规划改进证据。

已完成、独立审核的完整 C1 TRAIN 审计包含524,256条候选/131,064次场景呈现，
其中3,639条候选有原生D事件。D分项具有组内差异的791次场景呈现对应764个唯一
scene、395个log。它们与全部D事件人口不同，也不能当作原C2训练的统计。
`analyze_full_feedback_support.py::process_profile` 已核对
`DAC = 1 - any(native_flag[0:41])`，`verify_and_analyze` 核对四个候选同scene、
同policy、同原始context/reference输入，`support_scene_slots.D` 保存DAC有差异的组。

这给出一个具体机会：**只考察同一原始context中同时存在DAC=0和DAC=1候选的组**。
安全兄弟表明该固定context中曾有一条真实采样命令达到道路合规；它不是新目标轨迹，
不用于复制、替换或新增评分。如果固定初始车身已在道路外，原生41点任意违规归约会使
所有同context候选DAC=0，后续命令不能消除t0事件。因此D-discriminative组在正确的
同初始状态合同下应排除这一结构性不可修复情况；实际绑定仍必须显式核对
`native_flag[0] == False`，不能只依赖上述推论。

安全兄弟不保证距失败命令0.005 raw L2内有解，更不保证command-space代理经过LQR/
自行车执行后仍建议正确方向。失败命令也可能因tracker而出界、但命令自身余量充足；
原边界/tie和零梯度门槛仍可能使大部分人口未定义。这些是本次诊断要保留的结果。
旧全DAC1人口不能估计失败人口的可微覆盖率；目前证据既不足以宣布代理有用，也不足以
断言全部失败人口都不可诊断。无需修改代理数学来检验这一窄问题。

## 已固定的人口与实际资产

`FRAME_PLAN.md` 在本次取样前写入固定seed42和SHA规则。`build_frame.py`只读已完成
compact JSON与小型rank receipts：从4096个已审计ledger按固定SHA顺序访问202个，
得到32个唯一TRAIN scene/31个log，涉及31个boundary/ledger，policy version范围290–4086。
这是已冻结的两阶段ledger-prefix样本，不是对全部D事件的均匀独立抽样，也不是某个
checkpoint的32场景测试。重复场景只取规则下首次出现，不用新方向结果改选版本。

- 主审计：`/mnt/project/st-world-policy-20261016/audits/consequence/full_train_feedback_support_v1/RESULT.json`，
  SHA `0a6a84d3652122aeb69760b5237a56c71fec269c16b4066e78a0ddb05876d8cc`，327,728B。
- 主PLAN：`/mnt/project/VLA-Drive-st-full-feedback-runtime-20261018/reports/st_world_policy/20261018/full_train_feedback_support_v1/PLAN_CANDIDATE.json`，
  SHA `2d50faf1c690bd8ebc487504a32ecae9637432e4742c561a9ea5f6442a995b06`；
  plan ID `4f52f393a2be7ebb69128e8bad401007500b7ae5a6ba0f23e0e58b176506b50d`。
- 原run：`/mnt/project/st-world-policy-20261016/formal/consequence/runs/C1_fulltrain_seed42_official_4096_v1`。
- 本次 `FRAME32.json` 文件SHA
  `5ea6524267bc4e02fde7bfd79389ba774bea3f77821898b2471b369f316b1841`；
  selected元数据内容hash
  `2d64fef716d817d3d2c1d873a5b3aafa255cf3e42c4e86901544ade344070c99`。
  逐项保存scene/log、boundary/rank/slot、原policy身份、rollout ID和完整实际资产path/SHA。

FRAME已核对来源hash并stat确认：31个原`RESULTS.pt`共106,612,704B、32个原rank rollout
共7,481,472B、31个`REQUESTS.json`共6,501,721B均存在；32个rank COMPLETE已读取并
验证hash。**原 `.pt` 均未打开；其payload hash沿用已审计回执，当前只确认stat存在。**
当前未读取选中组的逐candidate DAC/native flags，也未读取这些REQUESTS中的真实context
路径。因此不能宣布32条失败轨迹已绑定可用，更不能猜路径。`failed_candidate_id`、
`safe_sibling_candidate_id`、`native_flag_0`和最终candidate-selection hash仍为null。

`CANDIDATE_BINDING_CONTRACT.json`明确后续受限准备的合同：按FRAME固定顺序，仅在各组
原有DAC=0候选中按固定candidate SHA选一条，DAC=1安全兄弟按相同规则选一条；不能
从PDMS=0推断DAC=0。复核实际raw FP32[8,4]、原十步生成身份、物理轨迹checksum、
完整REQUESTS/RESULTS/ADMITTED/COMPLETE/rank receipts、TRAIN归属、原policy、
metric-cache path/hash、context和完整reference输入一致，显式检查t0。最终完整selected
JSON用`sort_keys=True,separators=(',',':'),allow_nan=False`计算SHA，必须在读取代理结果
或评分前冻结。任一缺失/冲突为BLOCKED或该固定行unresolved；不补样、不重采样。

## 只允许的实现适配

真实0d0d19 runner当前不能直接接这个人口：

1. `diagnose_road_direction.prepare` 固定原selection SHA、32scene/31log、boundary4095、
   candidate0及rank/slot全排列。
2. `run_road_direction.load_inputs` 固定一个feedback ledger、policy4095，始终取
   `raw_actions[slot,0]`；新人口需要按每行真实boundary/rank/slot/candidate加载并校验。
3. `run_road_direction.summarize` 把`all_original_DAC_equal_1=True`和修复分母0写死；
   新模式须从已验证原labels计算，并保留请求32、已绑定、方向有定义、全部评分有效的分母。

允许的变化仅限这些selection/receipt/report逻辑及新模式严格96次调用上限；旧默认
人口/旧回执保留，原128上限不能被新模式继承。不能绕过原完整性、独立审阅或预算门。
原 `road_margin_proxy.py` 保持逐字节不变；此处未实现任何适配。

## 冻结的数学、对照与预算

唯一假设：对已固定的、具有安全兄弟的真实DAC失败命令，原command-space余量的固定
负方向是否比同范数正方向/固定随机方向改善真实执行后的道路后果。必须同时区分
自定义余量、原生道路事件和官方规划分数，不能把它们互换。

完全继承 `0d0d19dd5ce2d2fd583d2cc265289e47649ca594`：

| 定义 | 固定值 |
|---|---|
| raw与扰动 | 完整[8,4]的x/y/sin/cos；负/正/随机均raw L2=0.005，**不是米**；不冻结yaw |
| 自定义代价 | 四角41点，共164项含t0；`mean(0.1*softplus((0.1-d)/0.1))`；原精确polygon union |
| 原始decode | FP32精确扩展至FP64，原物理命令anchoring；XY舍入包络不变；yaw decode gate为8×2^-23 rad |
| 原官方几何对照 | XY、周期yaw、corner误差均沿用1e-9 gate；原epoch乘1e-6时间算术 |
| 有定义门 | gradient norm>1e-12且有限；moving-corner boundary/tie原门不变；sin/cos radius>=0.25 |
| 有限差分 | 两个步长1e-4、1e-5；同离散regime，relative<=1e-4或absolute<=1e-6 |
| 几何扰动帽 | 最大corner位移0.15m、yaw变化0.025rad；初始状态严格不变 |
| 四variant | original/negative/positive/random；original仅严格同身份复用，安全兄弟也不新评分 |
| random | 沿用原函数：SHA256(scene_id+NUL+最终selection_hash+NUL+road_direction_v1)前8bytes big-endian作为CPU Generator seed；不试多个随机方向 |

新评分最多32×3=**96次**，是所有场景、variant、进程和中断attempt合计的硬上限；
合法原label无法复用时不额外重评分original。0模型加载、0GPU、0新轨迹生成、0训练更新。
若Lead后续准入，沿用成熟单CPU worker/单数值线程、1800s wall、8GiB聚合RSS、
256MiB逻辑资产读、64MiB单cache展开、1GiB输出边界。现有selected `.pt`最低读取量已
为114,094,176B，REQUESTS另6.50MB；真实context/index/所有核验读量尚未知，需先按实际
路径stat/绑定预算，不能声称必然装入256MiB或自动加预算。此次2CPU/2GiB/120CPU秒的
元数据准备预算与未来诊断预算分开。

只进行这一次固定人口诊断，无自动重试/替代scene；中断后native调用意图仍收费，按成熟
规则保留uncertain，不自动重放。0d0d19已有失败恢复合同须继续生效。任何重新准入只可
续处理未开始scene，总96不重置。若数学、公差、样本数或资源无法在合同内满足，本轮停止。

## 预先规定的解释与无法判定条件

请求分母始终32。另报：已核实的原DAC失败数E、原t0=false数、方向有定义数、三条新
variant全部有效的共同配对数K，以及逐原因未定义/失败/中断数。E或K为0时相应比例为null，
不能填0、声称无修复或仅报告幸存者。修复数分别报`r/K`（共同有效条件人口）与`r/32`
（固定请求人口的已观察成功下界，非把未知当失败）；不丢掉那32中任何一行。

主读数为negative相对positive/random的配对原生DAC恢复计数/差值，执行后自定义代价
差值单列。完整保留原生D trace的事件次数/首次时刻；延后首次出界但仍有D事件不叫DAC
修复。共享原context的安全兄弟只是可行性见证，不当作随机干预对照。

原DAC=0导致原PDMS=0，因此“PDMS没有降低”或“没有高分变零”在本人口不能证明安全。
报告每个variant完整NC/DAC/TTC/comfort/EP/方向分项与PDMS；按各原安全分项确实合规
的子集给新退化分母，同时报告EP/运动学/tracking变化、修复是否伴随新碰撞/TTC风险。
旧高分转零统计可以保留，但其原分母为0必须明示。不新设质量阈值或挑较有利指标。

- 若candidate/context/原生flag/hash无法绑定，保持`BLOCKED`，不开始评分。
- 若原固定32全为未定义、几何不通过或K=0，当前代理的这个诊断不可判定；停止这一轮，
  不放松tie、有限差分或公差来制造有效样本。
- 若负方向不改善执行后代理，或比正/随机更差，记录对该方向可迁移性的反证，停止
  自动推进；不能推广为所有可微道路代理必然失败。
- 若代理下降但DAC仍全0，只证明局部代理变化，固定步长的道路修复仍无证据；不增加
  步长、不换失败scene、不启DFQL/C5/G8或新训练。
- 若出现真实DAC修复，也仅属于固定历史TRAIN命令的局部诊断；结合安全分项与有效
  覆盖率交独立审阅，不能直接签训练PASS或声称DEV规划改进。

## 本次实际工作与资源

元数据读取脚本实际CPU0.189233s、wall0.560122s、读取4,224,130B，未读`.pt`/cache。
`METADATA_READ_RECEIPT.json`保存每次小文件读取的path/size/hash；此前探索读取以8MiB
保守额度计入，上界12,612,738B。封存源码pins另读270,127B，FRAME小文件读取另有记录，
总量仍低于16MiB；未扫描524k候选payload。该脚本的getrusage高水位2,095,808,512B
低于2GiB门，但它不是连续live RSS测量，不能冒称实际内存峰值曲线；封存阶段
`/proc/self/status`实际VmRSS/VmHWM为14,476KiB。全任务各shell源代码阅读没有连续CPU
追踪，不伪造总CPU数据；实际重计算仅上述stdlib元数据步骤。独立实现者已收到同一固定
FRAME和绑定合同；此证据建议不签实现/系统PASS，也未授权真实诊断。
