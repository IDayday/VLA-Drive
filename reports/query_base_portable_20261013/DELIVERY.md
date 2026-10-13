# IL-only Query Base 独立分支

- 起点：公开提交 `3931fedbd30326c3a98f32fd0259e5d06a8588c5`。
- 分支：`feature/query-base-portable-training-20261013`。
- 最新用户决定：新实验仅轨迹 IL；本地四任务 Base 原样继续；不做 checkpoint 转换；尽快推送。
- 新入口 `iqe.base_training`；源码构造沿用固定原框架已有 `W_ACTION_ONLY` 配置。
- 三项辅助任务不构造、不前向、不读标签。保留原规划输入、Query IL、归一化和 repeat8。
- 同实验完整恢复；显式新初始化；历史四任务 checkpoint 不能作为新实验 resume。
- 新分支基于动态缓存实现之前的已公开历史。未 cherry-pick 动态缓存的任何提交、生产/调度脚本或运行产物。
- 未改动运行中 Base/缓存工作区，未停止、重启或迁移现有任务，未启动新训练。

实际检查：6 个小型测试通过；CLI、固定源码 import、IL-only ForesightConfig 校验通过。
真实原图 loader 检查 1 个场景，三张 1024×576 当前图像，唯一目标 ego [8,4]。
读取了已有完整 73k checkpoint 的清单/配方元数据（90892 场景），没有转换模型/Adam，
没有用其驾驶权重初始化新实验。没有运行新的 GPU forward/backward、全量回归或 navtest。

源码、依赖安装、数据目录挂载、准备/训练/恢复命令见 `docs/query_base/RUNBOOK.md`。
模型/数据本身不随仓库分发。当前 Base 和锁定 22k 的结果仍属于原实验。
