# A/V100k训练完成与评估启动恢复（2026-10-08）

核对时间：2026-10-08T00:14:26.175125+00:00。三组训练均COMPLETE100000updates，3199752实际场景曝光。完整100k Navtest结果尚未完成，不用任何局部预测或早期checkpoint分数替代。

| 组别 | 训练updates | 已导出场景receipt | 完整Navtest |
| --- | ---: | ---: | --- |
| A_ACTION | 100000 | 781 / 12146 | PENDING |
| A_NO_MAE | 100000 | 302 / 12146 | PENDING |
| V_QUERY | 100000 | 2635 / 12146 | PENDING |

原队列与观察器PID443350继续负责原注册的三个100k bank，不复制或重启观察器。三组都已启动真实纯当前图像推理，任务范围为完整12146场景/136logs。所有模型仍使用原FP32 optimizer-master恢复、FP32推理、TF32关闭、原10步Euler、单候选、scene-bound seed42；正式评分环境、checkpoint身份和原world_size4分区均保持原样。

追加资源调度器23b66776在Python3.10环境把signal枚举字符串写成未定义的Signals.SIGSTOP/SIGCONT。原始导出持续输出；追加helper失败，没有改变轨迹协议或得到可接受的完整分数。保留原失败run/rescue日志、meter及注册。

修复仅把远端信号值序列化为整数，新增三项SIGSTOP/SIGCONT/旧枚举字符串回归，连同原tail分区四项测试共7项通过。修复源码ed9762bf6b54bb3fff5cdc97a391a05b8fe20945已push/远端SHA核对，固定在/mnt/project/VLA-Drive-navtest-expanded-source-v4；六台相关主机均核对该文件hash一致。原A/V训练与模型推理源码不变。

只在核对PID、命令路径和精确plan参数后终止旧的本任务allocation-only watcher/run/rescue。旧helper按精确request身份停止并保留记录；原exporter核对精确module和bank后收到幂等SIGCONT。没有停止训练、官方评分进程、压力保护器或无关任务。

新allocation-only companion实际运行PID540238，使用独立navtest_AV_expanded_future_20261008_v2注册，最多36个额外helper容量；实际启动仍逐GPU做UUID、可用显存、归属和独占/共享lease检查。36是登记容量，不是已运行36卡。仅补充既有六个90k/100k bank，已完成90k不重复推理；不创建新的评价任务或修改方法。

现阶段仅保证调度恢复与真实导出在运行，不宣称已获得100k规划收益或加速倍数。完整12146场景官方评分、零缺失/失败校验和新的官方参考抽查完成后才能接受分数。

聚合证据：evidence/NAVTEST_AV_100K_STARTED_20261008.json。现有pipeline自行继续，不需要重复启动。
