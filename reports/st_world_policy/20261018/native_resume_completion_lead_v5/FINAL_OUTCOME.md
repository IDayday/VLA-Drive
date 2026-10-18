# 真实恢复验证闭环

本报告补充历史 `EXECUTION.json`，不改写当时“B2完成、B续训已启动”的记录。
正式训练器仍为 `751fcfa2b41549ae16f7b3204aef8dd18b912016`；外部观察器和完成协调器没有修改训练数学定义。

实际连续A为4步；新B先训练2步，在完整native边界退出，再由新进程恢复2步。A未重跑。四rank的模型与buffer、FP32 master、Adam、原生计数、client/data cursor及RNG哈希见证一致；恢复后样本ID、原4096步学习率、loss、实际rollout与后果归约审计也一致。原比较器给出 **EXACT_REPLAY**，独立Correctness Agent只读复核通过。完整证据路径与SHA在 `FINAL_RESULT.json`。

累计实际8次optimizer更新、256次scene呈现、32个唯一合法TRAIN场景、1024次官方评分、零评分重试。6次训练器启动包括历史失败；旧异常、旧B冷启动的三份metadata和旧租约全部保留。总输出55,773,576,699字节，低于104GiB上限。真实最终检查确认6份EXIT、518个PID/start身份均无活动残留，zt6无GPU计算进程。

比较本身耗时4.732秒、waited-child CPU 4.817秒、RSS high-water 665,030,656字节，没有重新加载GB级模型/optimizer。完整训练GPU busy时间和全campaign CPU成本未测得，不以局部NCCL-ready allocation区间替代。采样IO是下界，历史累计只继承一次。BC noise/time未直接存储，完整FP64参数更新delta/cosine未计算；不增加这些项目的等价性声明。

这是一项有限的真实工程资格验证，不是规划收益实验。所有SMOKE权重禁止用于正式初始化。没有新增navtest、没有恢复特征研究、没有更新ACTIVE，也未推送。

正式后果训练继续原run：`C1_rollout_std_fulltrain_seed42_official_4096_v1`。归档时为1113/4096步、142464次官方评分，loss/gradient有限，FP32 master实际更新；最近完整native为1024步。最近100步日志中最慢rank的常规update为mean15.426秒、p50 15.339秒、p95 16.820秒；该口径排除checkpoint、启动和外层调度时间。

下一判断点保持已登记的4096终点和固定DEV1696。训练期评分不是留出收益，不能据此宣称PDMS提高。真实8小时调用边界若暂停，先核验本任务进程退出和完整native，再从同一run恢复，保留原scheduler时间轴、RNG和数据游标；不会新建科学实验或使用SMOKE权重。终点评估的实际checkpoint、资源租约及独立审批须在产物存在后绑定，不预填成功状态。
