# 用户指定53k IL开启两组RL（2026-10-23）

用户明确要求使用53k权重开始RL，因此本轮不再等待100k IL及五个DEV检查点完成后选择父模型。保留原100k IL训练和DEV观察；只替换本轮RL父模型选择规则，原登记和已完成结果不改写。

父模型固定为完整原生53k检查点，model SHA256为`4c5188dec5674fc08f4b75b1a464c05348bf8a54623fd85f6598fb921ae5e786`。当前Qwen/provider与动作专家全部来自此IL，永久reference也固定在此。两组LoRA零初始化相同，不使用资格短训权重初始化正式训练。

53k已观察全量Navtest canonical v1.1 PDMS **87.016025**。本次是用户在观察该结果后的明确选择，记录`Navtest_used=true`，不再称原DEV-only父模型选择或完全盲测；RL检查点仍按既定DEV规则选择。

## 保持的正式配方

- CANONICAL与SOURCE各50k实际optimizer updates，seed42，各单机8卡。
- 相同公开hard list与本地TRAIN交集17037场景，不含DEV/Navtest。
- 每卡1 scene、G8/K10、shift5、仅7/8/9步随机，4次inner更新复用同一fresh链及old概率/优势。
- 原生3维x/y/heading、完整Qwen token读取；冻结Qwen/provider，只训练动作attention q/k/v/o的源LoRA，r16/alpha32/dropout0。
- 源创新和相对logprob裁剪、跨rank原始reward共同std、均值MSE anchor beta50，原3维权重[0.2,1,1]。
- AdamW 5e-5、100更新warmup后constant，保留源有效gradient clipping=0；无BC/replay/dual/learned-noise。
- 唯一方法差异仍是训练reward：CANONICAL 5:5:2，SOURCE 10:5:2；部署评价统一canonical v1.1单候选10步shift5 ODE、seed2101。

每组50k对应12500份fresh全局rollout、100000次scene曝光、800000条新candidate。DEV观察为0/5k/10k/20k/35k/50k，固定50k终点与DEV选择结果分开报告。正式RL终点与必要DEV锁定后再进行最终RL Navtest。

## 代码边界与验收

默认非milestone检查点仍不能直接成为正式RL父模型。新增独立的用户授权和selection schema，逐项绑定checkpoint、完整model hash、训练身份、已观察的完整Navtest证据及原预算；不能用qualification或旧interim flag替代。训练、导出和评价都传递同一sealed parent receipt。

旧五个DEV选择合同保留。旧等待controller通过`STOP_AFTER_TICK`退出，新的53k campaign使用独立控制目录与显式parent-selection参数，防止100k后重复自动启动另一组RL。运行中的IL不可变快照未修改。

已执行15项新父模型合同检查、16项旧interim合同回归、15项campaign调度回归，均通过。24个关键actor/model/collection/conditions/recipe/trainer配置文件与`eb64bf0`逐字节一致。该代码提交阶段只完成metadata入口；后续真实资格验收与正式启动证据见下节，不能把CPU检查当成RL效果。

保留最初[登记](RL_parent53k_registration.json)的`PENDING_SAME_PARENT_QUALIFICATION`原始快照；后续[正式启动收据](RL_parent53k_formal_start.json)明确更新为两组已启动，不重写旧登记或宣称50k已经完成。

父模型的完整DEV读出同时登记为0-update基线，固定终点完成条件也要求它完成；与后续RL的5k/10k/20k/35k/50k DEV分开记录。该补充仅属于评价队列，不改训练源码或loss。

## 已实际启动（10:02 UTC快照）

两组已通过真实八卡16-update资格短训，随后分别重新加载未更新的53k及相同LoRA初始化进入正式训练。资格权重未作为正式初始化，也没有重置一次进行中的正式optimizer。

| 正式组 | 主机 | GPU | 八rank一致更新数 | 已逐payload核验的恢复点 | 固定终点 |
|---|---|---:|---:|---:|---:|
| RL_CANONICAL | vla-zt | 8 | 208 | 100 | 50,000 |
| RL_SOURCE | tsinghua-zt4 | 8 | 188 | 100 | 50,000 |

快照时间为2026-10-23T10:02:03Z，16个GPU worker及两条torchrun均确认存活。原IL仍继续，八rank为54,585/100,000。此表的100是本次完整核验的恢复边界，不宣称它始终是最新checkpoint。

每组100更新的原生恢复点均通过：Adam全部480份state实际step=100、scheduler=100、480个非零一阶任务动量、25份fresh全局rollout、200次scene曝光、1600条新candidate、inner=0/READY、八rank事务一致。四次inner没有被算成四批新样本。两组前100更新的实际scene顺序逐rank相同；首批adapter身份相同，且与资格实验更新前的干净初始化一致。

资格实验中每组16次Adam/scheduler更新、480张量非零任务动量、全部A/B改变；冻结Qwen/provider及原动作reference hash前后一致。4份fresh rollout×8rank×3随机步，共96个实际采样mean/std对照均精确一致；正式前100更新中所有first-inner ratio误差也为0。这证明本次启动与更新合同通过，不等于已证明驾驶提升。

短训phase计时：CANONICAL约9.022秒/fresh collection、SOURCE约9.786秒，投射50k纯训练阶段分别约31.33/33.98小时。该投射仅来自4份fresh rollout，未计正式加载、500个恢复保存点、评价和未来共享竞争，不是完成时间承诺。实测峰值均为38,285,154,816 bytes。两组各800000新candidate的原预算已保持。

训练与评价固定使用不可变`434cba8`快照；controller固定使用`bfe2e52`快照，其中仅补齐父模型完整DEV的0-update读出。二者均独立于继续运行的IL `d4d0012`快照；没有热修改训练源码。当前没有新的RL单候选DEV/Navtest成绩，87.016025仍只是53k IL基线。

完整DEV按0/5k/10k/20k/35k/50k排队；固定50k终点与预登记DEV选择检查点分别报告。最终RL Navtest仍等待两组终点与选择锁定，不以训练随机策略或候选最好分代替部署成绩。

仅协同暂停了两台逐PID/UUID核实的原压力worker，各8个；原monitor、argv/environment捕获在服务器私有目录并保留，环境内容未提交。独立清理guard已启动，仅在本campaign及模型worker退出后解除本任务pause，恢复原1800秒观察策略，不强制启动新压力任务。未动IL、已暂停ReCogDrive或其他真实/未知任务。

证据：[原生100更新验收](RL_parent53k_formal_start.json)、[真实资格验收](RL_parent53k_real_qualification.json)、[启动存活快照](RL_parent53k_live_start.json)、[实际派发与源码](RL_parent53k_dispatch.json)、[17037场景数据预检](RL_parent53k_data_preflight.json)、[实测成本](RL_parent53k_qualified_cost.json)。

## 已执行入口

```bash
python -m scripts.analysis.qwen_sourceflow.select_user_parent \
  --checkpoint /mnt/project/qwen-sourceflow-native-20261018/interim_navtest_IL053000_20261023/pinned_checkpoint_update053000 \
  --authorization /mnt/project/qwen-sourceflow-native-20261018/parent53k_user20261023/authorization.json \
  --observed-navtest-summary /mnt/project/qwen-sourceflow-native-20261018/analysis/IL_update053000_NAVTEST_interim_20261023.json \
  --output /mnt/project/qwen-sourceflow-native-20261018/frozen_IL_parent53k_user20261023.json
```

以上命令已使用项目固定MODEL_PY执行；selection文件存在时拒绝覆盖。证据：[授权](RL_parent53k_authorization.json)、[选择](RL_parent53k_selection.json)、[metadata验收](RL_parent53k_metadata_validation.json)、[interim回归](RL_parent53k_interim_validation.json)、[controller回归](RL_parent53k_campaign_validation.json)、[算法文件未变](RL_parent53k_algorithm_unchanged.json)。

实际两组训练命令保存在[派发收据](RL_parent53k_dispatch.json)中，均使用`-m scripts.analysis.qwen_sourceflow.train_rl`，同一`--il-checkpoint`和`--parent-selection`，仅config和output不同，没有qualification/resume参数。

若后续任务已停止，恢复继续使用同一不可变训练快照、原run目录与完整原生checkpoint，在原命令末尾增加`--resume /absolute/path/to/committed_checkpoint`。原Adam、scheduler、RNG、chain、old概率和inner索引由现有恢复事务加载，不重新初始化LoRA或改用较新的IL父模型。当前两条训练仍在运行，不应再派发第二份同run训练。
