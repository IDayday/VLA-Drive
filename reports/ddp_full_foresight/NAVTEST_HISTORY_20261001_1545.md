# Navtest 完整历史汇总（2026-10-01 15:45 UTC）

已完成17次完整固定checkpoint Navtest评测：每次12146场景/136 logs，共206482个场景/检查点行，全部0失败。FP32 master恢复/FP32计算/TF32关闭，训练seed42、推理seed42、10步原Euler FM、单ego候选、无scorer；当前相机输入，保留W并删除辅助头。原完整精度官方v1 metric cache与全对象环境不变；逐行核对原cache hash、有限性与PDMS公式。

| optimizer updates | C0 PDMS | C1 PDMS | C4 PDMS |
|---:|---:|---:|---:|
| 18400 | — | — | 82.8002 |
| 25000 | 84.8514 | 85.3063 | — |
| 26000 | — | — | 86.7090 |
| 31600 | 87.6001 | 87.1887 | — |
| 40200 | — | — | 87.1357 |
| 51400 | 88.3269 | 88.1848 | — |
| 66200 | 88.6393 | 88.3947 | — |
| 70000 | 88.9967 | 89.1057 | — |
| 80000 | 89.0116 | 89.1966 | — |
| 90000 | 88.9124 | 89.1594 | — |

90k相对80k：C0下降0.0992points，C1下降0.0373points。C1−C0在90k为+0.2470points，136log配对bootstrap95%区间[-0.0407,+0.5463]，无法确认稳定优势；单训练seed/单推理seed，不代替跨训练重复。C4只完成至40200的Navtest，不能与90k作为匹配终点的配置排名。100k和C4后续指定步数仍由原自动观察器执行；不据Navtest改训练配置或选择checkpoint。

表内单位为0–100points，NC/DAC/TTC/EP/Comfort/DDC和零分数量均在同目录NAVTEST_HISTORY.csv。完整版逐场景文件保留外部，共75.82MiB，不上传Git：

/mnt/project/ddp-full-foresight-study-artifacts/20260929/navtest_milestones_20261001/navtest_summary_20261001_154742/ALL_SCENES.csv

现有训练源码d1d40854299b9599b2accc382bcfc4b676dd7623，运行中评测源码693a1a967fb1304e34551d07a347b13e526eb4d6均未修改。此次为只读结果整理，0新增推理/optimizer更新。汇总与所有CSV的SHA256见同目录identity.json。
