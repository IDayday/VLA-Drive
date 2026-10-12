# G2迁移到vlawm-zt4，2026-10-12

G2_EVENT_UNIFORM已从退役的`training-vlawm-zt`在23,269更新的完整optimizer边界保存，并在`training-vlawm-zt4`的八张A800上原生恢复。恢复报告确认FP32参数和buffers、master权重、Adam moments/steps/LR逐位一致，所有任务RNG一致；epoch=7、offset=1044、exposure=744552。恢复后更新连续，未退回20k或重新初始化。

训练仍使用冻结源码`0c54f1c49e7ed2a76b22826b654d6c72625f5b6d`、`io_preserving_v1`、global batch32、microbatch4、FM repeat8及原100k计划。运行identity仍为`a4226d2ce8974c3536118b01b4e775d0f0e16cf30ce4b670dee8ed5446912589`。共享标签及当前DINO通过原路径的只读链接使用，Python环境保持登记版本，Qwen/DeepSpeed关键源码hash匹配。没有重做几何准备或标签缓存。

目标机原有八个IQE FP32缓存worker继续工作；此次是获授权的共享GPU迁移。02:39 UTC附近G2最近50更新中位约8.6秒，原服务器迁移前100更新中位3.13秒。这个共享资源窗口不是孤立性能基准，也不是新算法或执行加速。原服务器的训练进程已退出，现行训练和评估路由均排除它。

原G2监督器在主动暂停发生于25k前退出，因此新增`tools/structured_world/continue_migrated_navsim_group.py`仅承接阶段调度。它等待当前25k进程自然退出，持有原run queue锁、目标物理GPU锁和迁移lineage锁，再按原命令恢复到50k/75k/100k。它不重启失败阶段、不发送进程信号、不改变训练器。25k/50k/100k沿用原Navtest入队接口，60k/70k/80k/90k由既有cadence负责；不增加75k评价。实际部署状态另存独立回执。

CPU测试直接运行：

```bash
python3 -B tools/structured_world/test_continue_migrated_navsim_group.py -v
```

15项通过，覆盖原阶段命令、锁冲突、重复启动、PID复用、停止标记、恢复边界、checkpoint缺失、失败不重试、共享GPU准入和环境白名单。实际目标机`--prepare`还需核对当前存活launcher与完整原训练注册。完整进程环境不写入磁盘，只记录允许的训练环境变量；恢复controller不读取或保存凭证字段。

迁移原始证据：`/mnt/project/iqe-runtime-audit-20261009/migrations/20261012_vlawm_zt_offline/`。调度修正审计：`/mnt/project/structured-world-fgtr-round1-artifacts/20261008/g2_zt4_migration_20261012/`。base评价策略移除退役host后，仅同步cadence引用hash；评价人口、科学协议及50k后每10k规则不变。

训练与已完成Navtest快照见同目录`G_SERIES_STATUS_AND_G2_MIGRATION_20261012.json`。当时四组均未到100k。25k的G0高于G1/G3，50k的G1高于G3；未来监督的因果收益尚未建立。G3 70k评价的SSH分片失败及80k排队状态如实保留，没有将部分分片写成完整成绩。
