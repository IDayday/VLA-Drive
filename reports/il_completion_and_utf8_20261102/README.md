# 新完整 IL Navtest 结果与发布修复

2026-11-02，四个检查点均已完成12,146场景、invalid=0的单候选canonical NAVSIM v1.1评测，seed2101、K10、shift5不变。

| 检查点 | PDMS | NC | DAC | TTC | EP |
|---|---:|---:|---:|---:|---:|
| B36k | 85.9877 | 97.3201 | 94.8460 | 91.7257 | 82.3339 |
| B44k | 86.2519 | 98.1146 | 94.5990 | 93.9322 | 80.9906 |
| C40k | 83.9705 | 97.9952 | 92.5902 | 93.2241 | 78.8685 |
| D34k | **88.8283** | 98.5180 | 96.7232 | 95.1013 | 83.0824 |

所有表项均为百分制。全部分项、模型/评分/人口身份和实际SUMMARY路径见 [completed_scores.json](completed_scores.json)。当前已测IL最高仍为D26k的89.0477481734；这四个观察点没有刷新最高分，不代表稳定单调增长。

## 为什么结果没有及时出现

B36k、C40k、B44k分别已在02:03、01:59、02:02 UTC生成完整SUMMARY。报告的定时刷新仍在运行，但cadence发出的刷新请求缺少 `scientific_result=false`，现有事件读取器因此忽略该请求。已修正字段并通过跨组件测试；实际02:31:32的报告包含176个完整观察点，已补入上述三点。

D34k另有末尾30/31两个分片，在wm-zt5完成一条合法评分后，因权限文件的中文内容按ASCII解码失败而退出。完整数据保留，不将失败变成reward0。操作脚本先验证两个异常都为同一权限文件I/O错误，保留前30个完成分片和各自已缓存的1条合法场景，只重排这两个原分片。

新worker启动时显式设置 `PYTHONUTF8=1`；模拟native scorer将locale切换为C后，真实子进程仍能读取Unicode JSON。只修复文本I/O，不改变模型参数、浮点精度、attention backend、轨迹、seed、world或评分公式。另将可变调度权限文件原子转为ASCII JSON转义形式，所有解析后字段值严格相等；不可变授权记录没有重写。

恢复后的两分片均正常结束，每个复用1条原缓存并产生378条新记录；D34k完整SUMMARY已产生，PDMS=88.82826337264599。没有重跑其他30个分片，也没有更换模型或scorer。

## 执行与验收

```bash
/usr/bin/python3 -m unittest discover -s tests/il_report -p test_burst_recovery.py -v
/usr/bin/python3 -m unittest discover -s tests/il_report -p test_navtest_cadence.py -v
/usr/bin/python3 -u -m tools.il_report.recover_d34_completion_20261102
```

9项burst测试、8项cadence测试通过，包括真实UTF8子进程和完成请求被现有事件读取器识别。恢复脚本已实际执行，为一次性明确审查；已有完成凭证时拒绝再次启动。

第一次操作资格检查假设仅31号失败，发现30号也未完成后在任何CPU信号/状态修改前拒绝。随后只对两个相同I/O异常进行明确资格检查并恢复。旧失败RESULT、rank记录和物理费用保留，没有伪造正常退出。

原报告owner继续使用，唯一cadence接续至CPU3525699；旧3492522不得重启。新执行源码位于 `/mnt/project/VLA-Drive-il-completion-utf8-20261102`，科学预测/评分源码仍为原matched-il-navtest，资源worker仍为已验证17c23cd。B/D和其他训练进程没有收到信号；TIME85,711终止等用户决定保留。

实际收据：

```text
/mnt/project/qwen-world-il-research-20261030/maintenance/completion_event_and_D34_20261102/
    POLICY_ENCODING_RECEIPT.json
    RECOVERY_LAUNCH.json
    FAILED_RANK30_BEFORE.json
    FAILED_RANK31_BEFORE.json
    D34_COMPLETE_VERIFIED.json
```

C42k、D36k等后续点继续由现有每2k规则排队。RL新检查点仍有未完成分片，其部分数据不作为完整新分数；本次没有改变RL训练配方或其冻结评分源码。
