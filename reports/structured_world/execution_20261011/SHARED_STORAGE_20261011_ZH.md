# 共享存储迁移的实际状态

当前 NAVSIM 训练图像和 metadata 已部署到：

`/mnt/project/structured-world-fgtr-execution-assets/20261011/navsim_current_train_v1/`

304,776 张图像逐文件 SHA256 与原 manifest 一致，实际图像字节数 63,670,100,680，核验 100.24 秒。当前/ego 等 metadata 共 203,190 文件、603,378,605 bytes，逐文件内容、目录人口与源缓存完全一致，核验 137.55 秒。未重生成标签，未改 ROI、相机、输入内容和 teacher 身份。复制进程的退出记录未保留，因此其 exit 状态记为 unknown；内容完整性由独立的全量核验确认。

图像核验：`SHARED_IMAGES_VERIFIED.json`，SHA256 `c8c5d6d794d1146bdad4e2a223b609783e9f5e5d55bce6fd7a628407835cf9e5`。

元数据核验：`/mnt/project/structured-world-fgtr-execution-artifacts/20261011/zt3_recovery_1459_v1/SHARED_STUDENT_TRAIN_METADATA_ACTUAL_VERIFY_V1.json`，SHA256 `af0fb2c5933457f7cb38bc5d3385371a2760d0fadaaff3df3006fe9fc2692610`。

vlawm-zt3 的旧路径 `/var/tmp/ddp-full-foresight-20260929` 已在确认无消费者后改为共享根目录的兼容软链接。原本未完成的容器内复制移到 `.partial_copy_20261011T1500`，不作为训练输入。冻结代码仍看到相同路径和 `local_stage.json`，实际文件由共享存储提供，没有热修改运行代码。

Python/依赖环境按用户授权重建，zt3 与 zt4 完整 84,661 个 regular files / 9,060,528,180 bytes 的环境内容核验一致。FusedAdam 已编译的小缓存另行恢复并 CPU dlopen / ninja dry-run 核验；没有整体升级或修改其他任务的环境。CPU 核验不能代替真实 CUDA optimizer / 多卡恢复验证。

15:54 UTC 尝试开启 full-gradient 对照时，资源门检查发现 zt4 正在启动另一项真实 IQE bounded 200-update 训练，因此没有创建本任务 GPU worker。zt3 同时执行真实 IQE 缓存任务，均予以保留。失败目录 `p0_gradient_v1_zt4_actual` 记录的是资源检查阻止执行，不是模型数值失败。下一次执行使用新目录和预登记，原失败记录不覆盖。

正式数据、标签、Qwen、共享几何准备和已有正式恢复起点本来在 `/mnt/project`，继续保留。新数值权重采用共享目录逐组存储，比较结束后清理非必要中间点；报告与校验身份始终保留。

原先只有 zt4 容器内副本的 G1/G3 early checkpoint（共同几何准备后、驾驶 update2）也已完成共享备份：`/mnt/project/structured-world-fgtr-execution-assets/20261011/navsim_initial_origins_v1/`。两组34个模型/optimizer/RNG payload逐文件hash与各自原COMPLETE一致，共90,791,866,272 bytes，核验25.46秒。它们是本轮必要的对照起点，复制既有权重，不是新增训练保存。

空闲窗口出现后，使用新的预登记与 `p0_gradient_v2_zt4_actual` 目录开始完整梯度对照。旧资源门拒绝记录保留；真实worker启动前再次通过物理GPU/无争用检查，未重试旧失败目录或终止其他任务。该阶段仍是数值观察，不代表正式训练验收已经通过。
