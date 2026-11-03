# D0 双节点正式训练：当前完整周期实测2.07×

固定快照为 **2026-11-03 13:20:08 UTC**。从真实正式训练journal得到：在 **global B64/G16/M2不变**的情况下，完整collection周期（rollout + 官方评分 + 两次Adam）由 **43.77093秒降至21.11764秒**，当前墙钟吞吐加速为 **2.072719×**。

这不是只比较纯update的数字，也没有把原先与旧I共享GPU的速度当作独占8卡基准。旧窗口取旧I实际EXIT0后额外15秒的独占TSH5时段；两边rank0都在TSH5，时间戳来自同一主机时钟。

## 窗口与实测口径

| 项目 | 原单节点world8 | 新双节点world16 |
|---|---:|---:|
| 真实更新窗口 | u1677–1714 | u1719–1796 |
| 窗口更新行数 | 38 | 78 |
| 用于steady的完整collection | 17 | 38 |
| steady完整周期总时长 | 744.10584s | 802.47025s |
| steady真实global场景数 | 1,088 | 2,432 |
| **完整collection周期均值** | **43.77093192s** | **21.11763806s** |
| **global场景/秒** | **1.46215758** | **3.03064196** |
| 单次optimizer耗时中位数 | 10.68206901s | 5.18799161s |
| 记录的generation耗时中位数 | 18.77941374s | 9.25936351s |
| 记录的scoring耗时中位数 | 1.55973901s | 0.78631927s |

完整周期倍率是 **2.0727191076×**；纯optimizer中位耗时倍率为 **2.0589988979×**。分项中位数不能直接相加来重建完整周期均值；完整周期以实际连续collection结束时间间隔计。

- 新窗口跳过正式world16最初4次warm updates，u1719起观察；不是把模型冷加载时间记为训练吞吐。
- 旧窗口有18个完整结束点间隔，其中经过checkpoint1700发布的1700→1702一条间隔从steady剔除，留下17个。**包含该间隔时整窗仍为1.46181406场景/秒**，与steady几乎相同，2.07×并非靠删除一个很慢的存盘间隔取得。
- 新窗口38条完整collection间隔尚未跨checkpoint发布，因此这里不能估计其长期存盘开销。
- 单个collection均为global64个场景，M2复用同一组材料更新两次；16rank日志中的global曝光不能再乘16。

## 可以说什么，尚不能说什么

已可确认：这次正式world16不仅通过有限资格，也已有真实新更新，在上述短窗口中，包含rollout、评分和优化的实际墙钟吞吐约为原独占world8的 **2.07倍**。

若当前steady场景吞吐保持，103,288场景一轮的纯训练量级约由 **19.62小时变为9.47小时**。这是当下速度外推，**不是长期交付时间承诺**：不含模型/评分服务启动、完整NAVTEST评测、未来checkpoint发布、资源等待及场景/缓存变化。

两个窗口的场景序列、模型训练时点、节点数量及长度不同，**不是同场景严格匹配的吞吐因果实验**；旧I退出后的独占基准减少了原共享的混杂，但没有消除其它差异。2.072719×略高于GPU数翻倍的2×，不能据此宣称稳定的超线性加速或等比例GPU时节省。

本页不新增部署分数，不把吞吐改进当作PDMS收益。[正式16rank实际更新和恢复证据](FORMAL_WORLD16_DEPLOYMENT.md)另行绑定；原D0完整90.17112884仍对应world8的u1400，不能转记到新world16 checkpoint。

## 来源与归档边界

只读归档原NAS `/mnt/project/d46-priority-rl-20261103/analysis/D0_WORLD16_THROUGHPUT_WORKING.json` 的完整9,992字节：[THROUGHPUT_WORKING.json](results/world16_formal_v2/THROUGHPUT_WORKING.json)。实际文件SHA为：

`5bfd5be0e34eacc4266e90f454f0d91f4436b3e61e55e0639e4e44fb98f8f6d6`

JSON保留各条真实完整周期间隔、checkpoint标记及计算结果。[固定窗口复算JSON](results/world16_formal_v2/THROUGHPUT_REPRODUCIBLE.json)与[复算脚本](../../tools/d46_priority_rl/analyze_throughput.py)也已归档。使用同一组固定窗口，可执行：

```bash
python3 -m tools.d46_priority_rl.analyze_throughput --output /tmp/d46_throughput_new.json
```

该命令只读既有journal，输出不得覆盖已有证据；不会启动模型、训练或评分。
