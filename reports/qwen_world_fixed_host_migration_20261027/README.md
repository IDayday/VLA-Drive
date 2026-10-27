# World训练迁移至固定许可主机

用户于2026-10-27要求撤回全部临时服务器。三个受影响的训练均已迁至固定许可主机，并通过真实八rank更新和源码hash核验；原动作attention-only RL保持运行。所有临时主机已退出本任务，未停止其它项目。

| 任务 | 旧主机 | 当前固定主机 | 合法恢复点 | 23:51 UTC实际共同更新 | 本次未封存更新回滚 |
|---|---|---|---:|---:|---:|
| World IL | rl-zt2 | tsinghua-zt5 | 52,156 | 52,366 /100,000 | 0 |
| Attention＋Memory RL | tsinghua-zt2 | tsinghua-zt4 | 18,600 | 19,116 /50,000 | 50 |
| Time/AdaLN LoRA RL | tsinghua-zt6 | tsinghua-zt3 | 36,000 | 37,764 /50,000 | 13 |
| Attention-only RL | tsinghua-zt | tsinghua-zt（未迁移） | — | 本轮迁移未重新读取 | 0 |

表中更新是时间快照。完整规范主机名、实际PID/start、八rank闭包、原始source与库存hash见MIGRATION_CONFIRMED.json。撤回的三台为training-rl-zt2、training-vlawm-tsinghua-zt2、training-vlawm-tsinghua-zt6；23:08:58 UTC的逐机退出核验均无World进程，NVIDIA计算进程列表为空。没有在撤回主机部署后续训练、重试、评测或压力任务。

本轮所有新调度只允许training-vla-zt及training-vlawm-tsinghua-zt/zt3/zt4/zt5。资源授权通过新的不可变收据和全局精确主机禁派发共同约束。原临时授权不再生效。固定主机采用用户已允许的GPU共享，原有ReCogDrive等真实任务均保留。

## 恢复事实与失败证据

此次不能称为完全无损：先撤回授权、再请求训练保存的操作顺序触发了旧RL入口的次序缺陷。其training_dispatch先require_dispatch，再处理边界退出；Memory与Time在发布最终保存点之前PermissionError退出。Memory最后观察18,650、Time36,013，分别从合法18,600和36,000重新执行。IL完成原生52,156保存并正常退出，随后完整续训。

两个RL完整恢复点均处于READY边界，不复用故障时未提交的局部chain；恢复保留Adam、scheduler、RNG、数据游标、reference与同一科学身份。旧failure evidence、行为和aborted invocation日志原样保留。回滚50/13次更新及已消耗物理成本不能从历史账本消失；不能把重算样本再解释为额外有效预算。完整证据根为`/mnt/project/qwen-world-postcot-20261025/temporary_withdrawal_20261027T230222Z`，包括FAILURE_AND_RECOVERY_AUDIT.json、LEGAL_RECOVERY_PLANS.json及原run的invocations/failures。后续撤回操作应先协同保存退出，再收缩训练授权；本轮没有热改正在运行的actor源码。

## 仅调度与恢复入口的补丁

- `registered_occupants.py`及job_worker可选入口：仅在当前用户已允许共享时，接受显式登记的小型现存任务。收据绑定task、当前授权SHA、host、UUID、PID/start、argv SHA、cwd、内存上限与短期准入窗口；不改变、停止或接管登记任务，不放行未知PID。未提供收据时旧准入逻辑保持。
- `resume_il_owner.py`：允许有完整原生保存证据的计划内正常退出继续原IL；严格验证原命令和新的单一--resume、checkpoint库存及零IL回滚。可以接管已启动的唯一worker，不重复开训、不自动新增RL。
- 当前原生训练源码保持：IL为snapshots/formal_main_v2，两个scope RL为snapshots/367bffd。22/102/102个source_files全部hash匹配。原actor、sampler、reward、loss、optimizer配方和预算均未改变。两个RL恢复后第一inner ratio最大误差均0，实际梯度有限；IL optimizer_global_steps52,157、skipped0。
- 固定主机准入脚本快照为证据根admission_source（ec8ca142）；它仅是worker/CPU调度入口，child训练cwd和source仍是上面的原不可变快照。

39项针对性测试通过，实际命令及测试源码身份见targeted_tests.json。包括精确共用者接纳与替换/过期/授权/内存拒绝、IL恢复命令和日志边界、原临时截止合同。实际八卡恢复证据与单测分开保存在MIGRATION_CONFIRMED.json；测试通过不代表规划分数提高。

## 已授权的新checkpoint Navtest继续执行

本轮先前要求的新权重评测没有取消。IL52,000、Attention RL26,600、Memory RL18,400、Time RL35,600已在迁移前锁定完整原生库存。每组12146场景、seed2101、canonical NAVSIM v1.1、单候选clean-x十步shift5 ODE，无teacher/search/scorer选轨。四组都已通过独立原生恢复与导出完整ODE的一致性检查（动作与请求轨迹最大差0），并写入真实评分场景；完整新分数尚未产生。

三组RL以共同IL40k（已完成Navtest87.972382）为自身控制；IL52k是独立IL续训观察。中间预算不同，不据此宣称方法终点排名。评测唯一owner及每组原生一致性证据见navtest_launch.json；执行代码为单独已push分支`eval/qwen-world-latest-il-rl-20261027`、1efcfe9cfa8c32b7355a743c59d308c8fa68ae9c。评测也只使用固定五台。

## 实际入口与避免重复

本次已经在运行，以下是审计入口，不能重复启动训练：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/mnt/project/VLA-Drive-qwen-world-fixed-migration-20261027 \
/mnt/project/flow-safe-improvement-20261013/venv/bin/python \
/mnt/project/qwen-world-postcot-20261025/temporary_withdrawal_20261027T230222Z/verify_migration.py
```

该命令实际已执行，审计结果一次性发布；若原结果已存在会拒绝覆盖。需要新的状态快照时另立审计文件，不重写原证据。

后续真实故障恢复应读取总根ACTIVE_IL_OWNER.json及各scope ACTIVE_OWNER.json，再验证最新完整checkpoint与单一owner退出。当前IL采用jobs/formal_IL_attempt003，Memory/Time采用证据根各自jobs/FORMAL_RESUME；正常运行不重复派发。不重置Adam、不从资格权重启动、不新增seed或算法臂。停止/完成实验清理授权不适用于这些活跃恢复点、IL40k父模型或已锁定评测权重。
