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

已执行15项新父模型合同检查、16项旧interim合同回归、15项campaign调度回归，均通过。24个关键actor/model/collection/conditions/recipe/trainer配置文件与`eb64bf0`逐字节一致。本阶段只完成metadata入口；下一步必须完成同父模型16-update真实8卡资格短训并从未更新53k重新启动两组正式训练，不能把CPU检查当成RL效果。

实际登记、资格与正式启动收据将在本目录补充。当前[登记](RL_parent53k_registration.json)的正式状态为`PENDING_SAME_PARENT_QUALIFICATION`，不宣称训练已经完成。

## 已执行入口

```bash
python -m scripts.analysis.qwen_sourceflow.select_user_parent \
  --checkpoint /mnt/project/qwen-sourceflow-native-20261018/interim_navtest_IL053000_20261023/pinned_checkpoint_update053000 \
  --authorization /mnt/project/qwen-sourceflow-native-20261018/parent53k_user20261023/authorization.json \
  --observed-navtest-summary /mnt/project/qwen-sourceflow-native-20261018/analysis/IL_update053000_NAVTEST_interim_20261023.json \
  --output /mnt/project/qwen-sourceflow-native-20261018/frozen_IL_parent53k_user20261023.json
```

以上命令已使用项目固定MODEL_PY执行；selection文件存在时拒绝覆盖。证据：[授权](RL_parent53k_authorization.json)、[选择](RL_parent53k_selection.json)、[metadata验收](RL_parent53k_metadata_validation.json)、[interim回归](RL_parent53k_interim_validation.json)、[controller回归](RL_parent53k_campaign_validation.json)、[算法文件未变](RL_parent53k_algorithm_unchanged.json)。
