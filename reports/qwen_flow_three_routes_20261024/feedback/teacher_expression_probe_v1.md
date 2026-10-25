# 四场景教师可表达性机制诊断

该项补齐本轮既定的小规模TRAIN拟合检查，不是新增正式方法臂，亦不更改S1/S2/S3。neutral外部轨迹四步资格及S3整体ADE改善不足以确认关键安全修复能被当前可训练范围表达。

固定四个不同TRAIN logs：唯一已准入的稳定零分修复、按(log,token)固定顺序的另外两个安全修复、一个安全EP改善。每scene只取单条已有真实评分的最高合格teacher，打破多目标冲突；全部为外部结构化候选，三个修复teacher PDMS1.0，进度teacher0.9108329。完整标签与token只存私有注册，未上传。

同一成熟BASE、同480个attention LoRA张量；纯正权重原生FM，无GT/anchor，仅用于拟合诊断。512次B4更新、2048次teacher曝光，固定5e-5、8步warmup；0/128/256/512各四scene×五seed单ODE，共80条真实评分。独立FM噪声/时间，未使用候选生成root耦合路径。目标是请求轨迹的原生3维编码，不是模拟执行轨迹。

主推理仍为当前观测、固定场景绑定root、source shift5十步ODE。teacher只用于训练损失和事后距离统计，不参与输出选择。即使四个场景拟合成功，也不能作为泛化、Navtest成绩或后续GRPO有效的证据；若未充分拟合，也不能直接否定完整DiT架构。

source10de893已冻结。v1/v2注册只完成CPU准备/事务加固，无GPU训练或新评分；root在公开快照重新生成v3，避免私有旧Native文件hash与公开默认执行guard的身份混用。CPU tiny原生LoRA/FM、READY恢复、分段评价恢复及预算拒绝通过，实际大模型结果尚未完成。

[CPU证据](teacher_expression_probe_v1_CPU.json)、[单独预算](teacher_expression_probe_v1_budget.json)。实际运行：R/feedback/teacher_expression_probe_v3/run；永久允许zt5 GPU7，独立worker与退出后资源释放guard。R为/mnt/project/qwen-flow-three-routes-20261024。

```bash
python -m scripts.analysis.qwen_flow_exploration.feedback_expression_probe train \
  --protocol /mnt/project/qwen-flow-three-routes-20261024/common_protocol.json \
  --registration /mnt/project/qwen-flow-three-routes-20261024/feedback/teacher_expression_probe_v3/registration.json \
  --budget-lock /mnt/project/qwen-flow-three-routes-20261024/feedback/teacher_expression_probe_v3/BUDGET_LOCK.json \
  --output /mnt/project/qwen-flow-three-routes-20261024/feedback/teacher_expression_probe_v3/run
```

命令已实际派发，运行源码必须使用冻结teacher_expression_v1快照；恢复增加`--resume <该run合法latest checkpoint>`，保留完整LoRA/Adam/调度器/噪声曝光游标和评价提交记录。
