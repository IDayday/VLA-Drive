# Qwen-World：最高已测IL父模型上的attention＋memory RL

**已按用户指示终止**：正式更新908，227份fresh全局rollout，14528条候选；八rank在同一保存边界退出，GPU进程与控制器已退出，后续自动Navtest取消，禁止自动恢复。本运行未完成原50k预算，没有部署提分结论。以下为历史登记方案。

用户单独授权的新运行：IL70k（完整Navtest88.792909907）→attention LoRA＋scene/predictive memory，Qwen冻结，50k更新。最高IL的选择时间在PARENT_SELECTION中锁定，不追逐之后新补测分数。Test已被观察，不能声称DEV选父或盲测。四架构主实验保持原100k IL/DEV选择/50k RL合同。

方法依据：原三条匹配50k终点memory90.628892，高于attention89.905612和time89.508000；这是单seed证据，不是已经证明方法普遍优越。原memory33.3k90.741310是中间已测峰值，未用其RL权重初始化。本轮从IL70k重新初始化LoRA与Adam。

配方保持G8/K10/shift5、随机7/8/9、canonical5:5:2、global reward std、relative logprob clamp7、clip0.005、inner4、anchor50和[0.2,1,1]、LoRA LR5e-5、memory LR5e-6、clip1。无IL辅助loss/BC/replay/dual。current memory每inner有梯度重算；永久reference为IL70k。不复用IL40k观测cache。

原IL40k加载断言与父模型schema完整保留。新增独立的用户选择schema，绑定完整Test分数、native权重、recipe和TRAIN manifest；新runtime仅替换父模型资格与元数据，复用原scope actor/采样/loss/optimizer/恢复。39项针对性测试已通过，真实8卡QUAL6→mid-inner恢复到16仍须通过后才启动正式50k；资格权重丢弃。

部署仍单候选clean-x十步shift5 ODE、seed2101、canonicalv1.1。50k完整提交后自动使用同一qualified exporter/evaluator做12146场景Navtest，配对IL70k。

```bash
/mnt/project/flow-safe-improvement-20261013/venv/bin/python -u -m tools.qwen_world_postcot.selected_parent_owner --registration /mnt/project/qwen-world-postcot-20261025/best_IL70_memory_RL_20261030/REGISTRATION.json
```

唯一owner和UUID lease防重复。vla-zt3八卡共享已知评测worker，实测每卡至少49GiB空闲；保留其他训练。故障不无记录重试；以原子合法边界恢复。正式状态和后续结果写资产根，完成后自动推送结果。训练启动不记为完成。
