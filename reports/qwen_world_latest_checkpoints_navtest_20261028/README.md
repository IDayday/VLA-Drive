# 新检查点Navtest登记

状态：已锁定检查点，待实际推理/评分；没有新分数。

| 模型 | optimizer updates |
|---|---:|
| World IL | 60000 |
| Attention LoRA RL | 46100 |
| Attention＋Memory RL | 33300 |
| Time modulation LoRA RL | 50000，固定终点 |

每组全Navtest12146场景，seed2101，原单候选clean-x十步shift5 ODE，canonical NAVSIMv1.1；每条候选独立固定官方参考评分。三RL父模型保持原IL40k，不重新选父模型，不改变训练。每组先做native恢复/导出真实轨迹一致性，再扩展分片。

仅使用当前固定五台授权主机；每次派发检查实际GPU UUID、可见进程归属与显存，共享但不停止其它任务。上一轮八组已全部COMPLETE/发布，本轮不重复推理。临时撤回主机不用于评测或重试。

每组额外与上一已评测同路线权重配对；完成后由唯一owner汇总和推送实际结果。本登记不声称任何性能提升。动作后分项预测准确性另在隔离分析分支检查，不改变本轮部署输出或评分。
