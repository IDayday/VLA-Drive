# 新增临时资源与同run恢复

2026-10-27用户先授权zt6临时使用、暂不限时，随后授权tsinghua-zt2与rl-zt2临时可用。
后两台没有指定截止时间，登记为可撤销的任务级临时授权，不沿用已过期的九小时租约。
没有修改其它项目权限，没有恢复已暂停的旧研究任务，也没有停止或抢占其它任务。

| 主机 | 本次用途 | GPU |
|---|---|---|
| training-vlawm-tsinghua-zt | 已有attention-only RL对照继续 | 原八卡配置保持 |
| training-vlawm-tsinghua-zt6 | TIME_MOD_LORA继续，移走共享Memory进程 | 0–7 |
| training-vlawm-tsinghua-zt2 | ATTN_LORA_MEMORY原run恢复 | 0–7 |
| training-rl-zt2 | 后续已登记评测的可用容量 | 当前未部署新任务 |

新增两台各有8×A800 80GB。两次库存核查均无GPU计算进程，最低空闲显存分别81153、81221 MiB；
CPU可见亲和性均128。原生派发器在UUID锁内再次检查实际占用、利用率与显存，不凭旧库存直接启动。
这次将两组拆到独立八卡是消除资源争用；没有改变G、K、batch、inner次数或训练预算，
尚无迁移后的足够完整速度窗口，不能声称具体提速倍数。

## Memory恢复边界

仅向Memory自己的原生worker设置`STOP_REQUESTED`；没有向未知进程发信号。
原八rank在update391、collection98、inner3完成后共同原子保存并退出0：

```text
ATTN_LORA_MEMORY/runs/RL_ATTN_LORA_MEMORY_IL040000_seed42/
  checkpoints/update000391_inner3
files.json SHA256:
  a1ea4962172aee26a0c7b9593cd6e35f4e2835e55f01f339941a4f09d562c421
```

全部保存文件hash与八rank的phase、cursor、rollout、old概率及advantage合同通过只读验证。
此前已经合法提交98份fresh全局rollout、6272个候选；不是391份fresh数据。
恢复时先完成同一collection98的inner4，再生成下一个behavior版本的真实候选。
不重新生成一条链来冒用原概率、不清空Adam、不重置scheduler或随机数。

旧固定终点控制器发现子进程未达50k，会写`FAILED_REVIEW_REQUIRED`。
此处是明确记录的资源迁移：native训练结果为`STOPPED_AT_SAVED_BOUNDARY`、退出码0，
不是一次未解释的训练故障。原日志、成本、owner与停止结果全部保留。

新owner直接使用已有的`tools.qwen_world_postcot.job_worker`执行原命令加`--resume`，
运行目录及训练源码仍是不可变`scope_rl_IL40k_20261027/snapshots/367bffd`。
其GPU门禁、进程监管、原生恢复、日志invocation归属及50k终点均复用现有实现；
没有另建RL训练器，也没有为这次迁移增加资格训练或评分数据。

## 实际记录与恢复命令

以下路径相对于`/mnt/project/qwen-world-postcot-20261025/scope_rl_IL40k_20261027`：

- `AUTHORIZATION_zt2_hosts_20261027T145013Z.json`：不可变新授权。
- `NEW_ZT2_RESOURCE_INVENTORY.json`：两次实际库存。
- `ATTN_LORA_MEMORY/formal_migration_tsinghua_zt2_attempt004/MIGRATION_INTENT.json`：停止前意图。
- 同目录`LEGAL_RESUME_PROOF.json`：全量文件及八rank合法边界验证。
- 同目录`REGISTRATION.json`：实际原生worker和训练argv，带原checkpoint hash。
- 同目录`OWNER_LAUNCH.json`：新远端worker PID/start身份。
- 同目录`jobs/FORMAL50000_RESUME391/RUNNING.json`或`RESULT.json`：原生worker真实状态。
- `ATTN_LORA_MEMORY/ACTIVE_OWNER.json`：唯一活动owner指针，`owner_kind=native_job_worker`。

训练实际argv是既有`torch.distributed.run --nproc-per-node=8 -m
tools.qwen_world_postcot.train_scope_rl`与原全部参数，再加：

```text
--resume /mnt/project/qwen-world-postcot-20261025/scope_rl_IL40k_20261027/ATTN_LORA_MEMORY/runs/RL_ATTN_LORA_MEMORY_IL040000_seed42/checkpoints/update000391_inner3
```

以上命令已经执行。继续操作应先读`ACTIVE_OWNER.json`，不重复派发仍运行的任务。
新的阶段状态和精确实际计数见 [resource_assignment.json](resource_assignment.json)。
当前没有本轮两种新scope的Navtest分数；验收与恢复成功不能证明规划性能已提升。
