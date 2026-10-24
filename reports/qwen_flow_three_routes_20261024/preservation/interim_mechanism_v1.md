# P0–P5 中期：更新信号、约束与成本

机制快照：**2026-10-24T23:24:13.464798+00:00 UTC**；成本另在**2026-10-24T23:29:33.903345+00:00**冻结读取。训练尚未结束，下面不是部署收益结论。

当前新source-SDE采样人口里，六臂已观察组的有效PG比例均为100%，progress-only为0，约74%–76%的组有NC/DAC/TTC差异。有非零优势只说明可以更新，不说明探索有用或部署提高；它**不否定ODE候选存在满分饱和**。

|臂|已提交更新|fresh collection|有效PG|后3inner clipping|共享worker小时*|
|---|---:|---:|---:|---:|---:|
|P0|119|30|100.0%|0.420%|4.068|
|P1|88|22|100.0%|0.132%|3.729|
|P2|80|20|100.0%|0.302%|3.764|
|P3|79|20|100.0%|0.340%|3.765|
|P4|84|21|100.0%|0.289%|3.790|
|P5|80|20|100.0%|0.319%|3.783|

*时间不是独占GPU计算小时；包含原P0 OOM和跨主机迁移进程，运行中elapsed仍是估计，且与机制采用不同冻结时刻。

第一inner的ratio均为1附近、clipping均为0，这是固定behavior的正常起点；第二至第四inner的实际clipping仅约0.13%–0.42%。本窗口不支持继续用“PPO clip挡住更新”解释结果。ratio仍是源逐维平均随机转移代理，不能称整条轨迹的精确概率比。

P4/P5的λ分别约`[1.000056,0.993408,1.003159]`和`[1.000322,0.993945,1.003809]`，与初始1的变化均不足0.7%。每fresh collection一次dual更新的符号、成本均值和保存A已复核；当前短窗中初始成本惩罚占主要作用，尚不能将结果归因于乘子自适应学出了新权衡。

违规候选仍可获得正A，组内有完全安全替代时也会出现。这是相对优势与统计成本surrogate的事实，不能直接写成“约束已经满足”，也不能单凭这一点认定代码错误。当前各臂进度不同，不用累计计数比较安全胜负。

**限制保留：**P1漂移匹配仍为`NOT_MATCHED`，49.1%误差超过20%容差；replay系数封顶1，固定4scene校准的replay/PG梯度比仅5.29%，不能称达到25%目标。均不追加调参。

只读详情见[aggregate JSON](interim_mechanism_v1.json)，不公开collection CSV或逐scene数据。终点必须重新采样评价：source-SDE均值与原单候选ODE分表；是否减少退化、提高PDMS仍待完整匹配结果。

实际读取命令（CPU，真实入口）：

```bash
CUDA_VISIBLE_DEVICES='' /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  /mnt/project/qwen-flow-three-routes-20261024/preservation/tools/summarize_committed_v1.py \
  --require-complete --verify-all-payload-hashes \
  --output /mnt/project/qwen-flow-three-routes-20261024/preservation/analysis/terminal_complete_v1
```

该终点命令要求六条原生run均COMPLETE；paused worker退出0不等于训练完成。逻辑终点每臂256更新/64份fresh/1024次scene曝光/16384条新候选；四个inner与恢复不能重复乘算新数据。物理账本跨全部invocation汇总，原OOM、暂停/迁移、校准与共同建库成本分别保留。
