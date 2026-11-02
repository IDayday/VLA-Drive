# M2 日志恢复修复

S3 FPO 完成 u404 后，首次续训在日志校验阶段失败，尚未执行新的 Adam 更新。
旧校验将每 collection 的更新数写死为 4，S2/S3 实际使用 M2；这不能作为算法崩溃或无效的证据。
原 `formal_to009684_attempt001/RESULT.json` 的 returncode1 与 traceback 保留。

修复从封存 identity 的 `recipe.research.performance.inner_updates` 读取 M2/M4；
没有该字段的旧 M4 路线保持原行为。科学冻结源为 `7318a3a003ba7db480d7068dfbc98fb63c7d0431`，
目录 `/mnt/project/qwen-world-performance-rl-20261103/runtime/m2_journal_guard_v1`。
实际运行 source inventory 仅 `starVLA/rl/simwam_recipe/invocation.py` 一个文件不同；
actor、loss、采样、scheduler 和 scorer 源码不变。

实际 CPU 审查重放两实验各八 rank 的完整 404 行前缀，共 6464 行；
旧 M4 规则均重现错误，修复后全部通过。冻结源又分别读取并校验八份原生 rank 状态、
optimizer、scheduler、pending/scene-stream 状态，均通过。
[完整前缀审查](results/M2_JOURNAL_CPU_PREFIX_CHECK.json)；
[八 rank 原生 CPU 校验](results/M2_JOURNAL_NATIVE_CPU_VALIDATION.json)。

执行迁移只允许这两个原 run 的封存 `update000404_inner0`，
保留原 scientific identity，使用完整 Adam/RNG/buffer，禁止新的初始化与 cache flag。
typed 收据只增加 `--migration-receipt`。原失败由明确工程复核归档，未伪改成成功。
S3 已恢复到资源等待；S2 原有资源让位状态保留。
**本次尚未取得修复后的 GPU 续训结果**，不能称完整 GPU 恢复验证通过。

CPU owners 已切换到冻结 `ops_v9`（`d0e33bb`），原有 S1 GPU 子进程继续运行，
没有重置其优化器。六台临时服务器于硬截止前完成自有进程退出；
后续只有取得合法容量时才从本实验自身 checkpoint 继续。
[实际执行采用记录](results/M2_EXECUTION_ONLY_ADOPTION.json)；
[截止前实际归还记录](results/TEMPORARY_HOSTS_ACTUAL_PRE_EXPIRY_RELEASE.json)。

主工作树整合后的相关 CPU 检查共 7 项通过，包含 M2 前缀、迁移收据、旧缓存回归与成本统计。
这些检查没有产生新的驾驶成绩。当前最高完整部署 PDMS 仍为 91.45359756。
