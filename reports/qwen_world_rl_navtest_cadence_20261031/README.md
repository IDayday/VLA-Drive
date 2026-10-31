# 独立 NAVTEST 每2k评测旁路

已由 root 审核启动：PID2342200 / start_ticks1485735603，2026-10-31T03:52:22UTC。运行脚本 SHA256 为 `b96dd4450aa5a85ca9be4e46b8028166ac3da42ecdfba0ff31406b08a4acbe30`；本目录保存精确源码与当时真实行政证据。

只处理 RUNS 中已有首个任务且明确获准预算>10k的健康路线。checkpoint 必须原生 COMPLETE、updates 精确匹配、next_inner=0。新增12k、14k、16k、18k、22k……等每2k点；20k/35k/50k等旧门槛仍交原控制器。训练预算、TRAIN频率和科学 runtime 均不改，停止或未启动路线不会复活。

原 v3 官方 scorer，全12146 NAVTEST，单候选 ODE K10/shift5/seed2101。新评测请求最多32个合法可用槽位、至少8个，实际world在首次spec中冻结。旧world16等布局不改。最多同时评2个模型，CPU export串行；复用全局label锁、EVALUATION_DISPATCH锁、原UUID/visible-owner/资源准入证明与共享claims，并按实际CPU/RAM余量控制新增负载。每个成功worker立即持久化claim，部分启动失败不会暴露已派兄弟的GPU预留；真正child错误保留证据不自动科学重试，pre-child75只进入容量等待。

旁路持续监听原生checkpoint COMPLETE和RUNS后续合法晋级，1800s元数据heartbeat，不读训练journal、不等待训练阶段、不以补完当前两点为终点。STATE和RESULTS分别列全部评测状态及真实完整SUMMARY索引。actual分片、场景数与成绩以标准spec/SUMMARY为准，requested32不是已经运行32的证据。初始可补点为Safety12k/14k；未来其他路线仅在正式晋级并产生checkpoint后纳入。

启动时每份元数据单独原子读取，INITIAL_STATE/RESULTS不是跨文件事务快照。未归档NPZ、训练tensor、日志或完整场景JSON。SOURCE_ARCHIVE逐文件列原路径、SHA和大小。v1在inventory前检查动态STOP/预算；root已知的可选改进为inventory后再检查，当前运行源码保持冻结。
