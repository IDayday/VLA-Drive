# 修复IL报告最高分汇总

报告并未停止刷新：核验时最近一次渲染为2026-11-01 15:02:18 UTC，已包含161个完整IL观察点。然而首页“当前已测IL最高”使用了错误筛选，主动排除了`ARCH_*`，导致完整结果表中的ABCD高分未进入首页最高分。

已移除此筛选，首页与完整结果表使用同一份经核验的IL Navtest结果。没有新增推理、评分或修改已有分数；仍为每点12,146场景、invalid0、canonical v1.1、单候选K10/shift5、seed2101。

当前已核验最高为 **D组26k：89.0477481734**；旧首页取原始Qwen-World IL70k的88.7929099070。D28k为87.9830815164，不代表最高值被新终点覆盖，也不根据此卡片选择RL父模型。

实际使用原161点snapshot渲染并通过HTML parser检查首页第一张card，确认包含89.0477、D组和26,000更新。证据：`/mnt/project/qwen-world-il-research-20261030/maintenance/headline_best_il_20261101/VERIFIED.json`。保留既有30分钟常规刷新、完整阶段事件刷新、watch配额故障回退和T系列暂停状态；只接管原单一HTML owner，不创建第二个watcher。
