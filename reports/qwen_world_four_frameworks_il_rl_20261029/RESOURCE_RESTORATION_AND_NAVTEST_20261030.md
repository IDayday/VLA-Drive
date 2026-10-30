# 2026-10-30：资源恢复与用户追加任务

vla-zt3与tsinghua-zt2重新获临时授权，直到撤回；vla-zt修复后重新确认可用。固定权限未改变，tsinghua-zt6仍禁用。两次实测vla-zt有约920GiB可用主机内存，vla-zt3八卡最初为空闲。清华zt2虽然获授权，但GPU存在高利用率与超过45GiB的未归属分配，评测队列暂不使用；不停止或重置未知任务。

四架构正式源码fdaff80、IL100k/RL50k预算和单owner保持。新增授权不会重启任何旧实验。

用户另行要求最新及有价值中间权重Navtest：独立分支eval/qwen-world-latest-midpoints-20261030，已锁定10个未测checkpoint并部署32个推理worker；前四模型均已通过真实native/export预测一致性。全部使用12146场景、seed2101、单候选clean-x K10shift5、canonicalv1.1；优化IL40k现有owner复用，三RL50k终点旧成绩复用。队列注册与源身份见配套JSON和该评测分支报告，不改变四架构DEV父选择。

用户又追加一条RL：当前最高已测Qwen-World IL70k88.792910为固定父，采用匹配50k终点最好的attention＋memory方案，从零LoRA/Adam开始完整50k；永久reference为IL70k，Qwen冻结，current memory每inner重算。独立分支feature/qwen-world-best-il-memory-rl-20261030，仅新增父资格schema和调度适配。九个actor/采样/loss/optimizer合同文件hash与原memory run完全相同；39项针对测试通过，真实8卡资格和恢复验收进行中，尚不能写为正式训练完成。

这条追加运行是用户按已观察Test选择的父模型，不是四架构中的第五个主实验，也不是盲测或DEV选择；其训练与未来完整Navtest单列归因。
