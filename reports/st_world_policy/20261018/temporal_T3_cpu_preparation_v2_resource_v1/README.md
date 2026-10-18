支持 exact T3 CPU preparation v2 的资源继承；本 receipt 不派发，也不替代实际 admission 或科学 gate。绑定 owner PLAN `0fd1b5d9…` 与实际 unsigned proposal `d02545a2…`，完整文件 pin 和 fresh 快照见 [REVIEW.json](REVIEW.json)。

v1 失败根目录及其完整 8 MiB 承诺保留，v2 另加 8 MiB（含 1 MiB terminal controls），保守总量为 7840 + history32 + failed-v1 8 + v2 8 = **7888/8192 MiB**，余 **304 MiB**。这是行政输出核账；没有改 typed claims、cache policy 或 ledger。v1 实耗 155,131,761 B rchar、652,754,944 B peak tree RSS、5.964478 s，失败记录不被清零或当作成功的两 CLI 成本测量。

CPU [2,3]、4 GiB parent/child rchar、2 GiB tree RSS、300+40 s、整个新 root 8 MiB 均保持。prepare_once 仅刷新 runtime/root/PLAN_SHA/COMMIT 四常量，已有 supervisor、读计费、输出 census 与 controls 限额继承；源码数学兼容由 correctness reviewer 负责。本次未重跑 profile、CLI、测试、DB hash 或模型/教师 payload。

CP2703 仍 CURRENT 至 2026-10-18 01:44:11 UTC，fresh statvfs 满足保守 **4503 GiB**：CP2703 + floor1024 + 全 cache768 + 全 metadata8。实际签发仍需复核 freshness；这里没有提前分配 GPU 或 checkpoint 预算。

后续 4-step PROFILE 的 80 GiB 来源是旧 topology240 未签发的请求，当前 CP2703 将它保守计入 `additional_conservative_unissued_GiB`。它尚未绑定 T3。只有实际 PROFILE 的保留 native 目录、同时 staging 和其他输出峰值明确落在 80 GiB 内，并由 Lead 唯一绑定 run 后，才可在既有总额内承接；本次 CPU preparation 不能把它当作已授权 checkpoint reservation。

实际完成补记：Lead 已完成 v2。新 RESULT SHA `443e6ad1…` 为 `PREPARATION_COMPLETE`：7.950804 s、1,610,906,263 B charged rchar、684,961,792 B peak tree RSS、return0、owned0，均在既有上限内；无新 teacher tensor、training、ledger 或 checkpoint reservation。两次实际读取合计 1,766,038,024 B，v1 失败记录与承诺继续保留。这不构成后续 GPU PROFILE 或 resume 验证。
