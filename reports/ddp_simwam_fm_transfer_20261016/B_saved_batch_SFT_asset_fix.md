# 保存批次诊断的永久SFT资产路径修正

2026-10-17 14:26 UTC，B的15k原生检查点已封存，正式训练继续。辅助梯度诊断首次尝试失败：其`--hard-manifest`和`--sft-screen`把永久参考SFT的100k路径误写为150k。检查同模板的20k辅助脚本，发现对应路径同样被误写为200k。两者都应始终绑定B自己的100k SFT，与RL观察点15k/20k无关。

失败发生在既有`TransferDiagnostic.prepare`读取不存在的hard manifest时，早于`make_actor`。未生成新候选、未评分、未进行梯度诊断或optimizer更新。原15k失败owner与child均已退出；失败日志、调度参数和状态原样保留。该尝试dispatch到exit耗时1.458313秒，计入额外物理开销；不能因没有有效输出而从成本中删除。正式训练、检查点、已完成SFT/RL结果及正在运行的数值评价不受影响。

此前辅助任务检查覆盖了脚本编译、阶段条件和等待衔接，没有覆盖最终CLI引用的SFT资产。这一缺口已补上：新attempt在任何GPU派发前实际调用同一个`TransferDiagnostic.prepare`，核对正式run身份、永久SFT身份、hard manifest文件hash、筛选合同及物理screen manifest。命令使用固定HARD/SCREEN变量，不再让RL步数改变SFT路径。没有修改fd快照中的诊断数学、225b正式训练或45d数值评价。

六项已执行检查为：15k与20k真实`--preflight-only`均绑定100k SFT；两个旧错误路径在模型构建前均被拒绝；错误hard-manifest hash被拒绝；另一条路线的物理screen被拒绝。共同永久SFT模型hash为`4e71b1fcf963d38456174de1f8d17762a9a4ccf1597c22b7cec262026d9201b7`，hard manifest文件hash为`cc6017f0740c080dfae73f36ddc24bb657ae2996a8440b7c035ad4821612ec4a`。

15k以独立v2目录显式重试，经过两次实际空闲GPU检查后，在授权vla-zt的GPU1运行并完成。owner 544792、child 544802均已退出；诊断记录60.972904秒，新增候选/评分/optimizer更新均0，分析前后adapter精确不变。输出中`optimizer_updates=15000`表示被分析检查点的已有更新数，不表示诊断又训练了15000步。完整梯度结果由15k报告另行解释，不能用本修复记录代替规划效果。

20k尚未到达。准确核对PID、start-ticks、命令、等待阶段及无child/无dispatch后，仅停止了旧v2诊断等待owner 521880；保留其目录和旁路superseded记录。独立的v2成本捕获owner 521881保持运行，未重启。新v3诊断owner为544793、start-ticks 1368606709，已通过真实资产预检查，正在等待COMPLETE终点。两个修正attempt均保留原绝对截止时间，没有延长训练、诊断样本或评分预算。

只读汇总owner、里程碑评价owner和最终Navtest owner保持不变。修复时DEV与TRAIN的16个评价worker已在原45d源码运行；B训练原PID仍在推进，15,200步恢复点已封存。只修复辅助任务参数和准入检查，不改变算法或现有结果。

已实际执行的预检查命令如下。当前15k已完成、20k已有唯一owner，不应重复启动普通入口：

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=/mnt/project/fm-transfer-20261016/source/fd440275edc3fdf40b2aaa46c38da1d038068902 \
  /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  /mnt/project/fm-transfer-20261016/resources/B_RL_15k_saved_batch_dispatch_20261017_v2/run.py --preflight-only

env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=/mnt/project/fm-transfer-20261016/source/fd440275edc3fdf40b2aaa46c38da1d038068902 \
  /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  /mnt/project/fm-transfer-20261016/resources/B_RL_20k_saved_batch_dispatch_20261017_v3/run.py --preflight-only
```

[15k修正入口](reproduce_B_RL15k_saved_batch_v2.py)、[20k修正入口](reproduce_B_RL20k_saved_batch_v3.py)、[六项检查、失败成本、明确接管与真实15k完成证据](B_saved_batch_SFT_asset_fix_validation.json)。这里的完成仅指15k辅助诊断修复，不代表B20k训练或最终Navtest完成。
