# Data and leakage audit

真实训练源 `student_train_v1` 有101592条 index 记录；独立开发源 `student_dev_v1` 为16 logs/1696 scenes。均为原轨迹框架的数据。基础权重不复用，因此新 Query 基础 fit/val/cal 可在其训练前分开。所有标签、当前观察、图像内容及 metric context 以实际 ID/hash 绑定。

本次锁定128条诊断清单：incremental_fit 32条/30logs，stage_val 32条/16logs，selector_cal 32条/18logs，dev_report 32条/1log。共65组、128唯一观测，log/source-parent/observation 隔离通过。开发诊断只是固定ID前缀，只有1log，不能做科学 CI 门槛。正式 build-splits 使用全 index 的稳定log hash，不按单时间窗随机切分。

原始训练缓存缺正式参考；96个 train-domain 诊断 context 已从实际日志/地图按官方 v1.1重建，dev 使用原官方 cache。旧数据/cache 未改。清单及日志在 `/mnt/project/iqe-runtime-audit-20261009/outputs`，实际比例详见 `split_audit.json` 与 source manifest。不是全量数据接入执行证据。

没有新增真实/合成观测，本轮数据实验只能称 resampling_only。实现具备合成观测与 context/渲染证据绑定、unverified隔离、替代目标显式启用、同观测固定target规则；这些接入行为经合成单测验证，未声称收集到了新增数据或完成 E3。

训练、检索、Scorer仅用 incremental_fit；stage_val只选checkpoint；selector_cal只校准；dev不回调阈值；final_test独立入口且本次未访问。来源组/父组/图像/标签hash错误会拒绝。异常监督进入队列，评分错误保持无效，不自动归零有效标签。

测试覆盖来源交叉泄漏、dev/test注入、冲突target、缺桶转移/显式anchor=0、累计曝光、确定性检索排除自身组、cache/augmentation hash和第二轮旧新重放。真实大规模新数据接入仍未运行。

真实双轮使用上述清单的固定44条子集：fit32、stage_val4、selector_cal4、dev4；再次按log/group/observation核对隔离，见evidence/REAL_SMOKE_SPLITS.json。Query base实际只消耗fit中的1条样本，真实曝光记录保存在base_cpu_probe/base_exposure.json；不能把允许训练的32条全部标记为已见。两轮hard_original为31/27条，均通过真实官方目标审计，probe=false，阈值没有降低。每轮专家实际8次曝光，H/N/G为6/1/1，即75%/12.5%/12.5%；这是8次短计划的整数配额，不能称为精确80%/10%/10%。极小bucket配额下单组占比也无法达到0.10，实际曝光保留在原manifest；当前sampler进一步显式记录整数可行上限与实际最大比例，没有追改已冻结的smoke manifest。

全量原数据接入实际执行完成：fit90892场景/1056logs，stage_val5563/56，selector_cal5137/64，dev1696/16；1192组、103288唯一观测，log/group/observation隔离PASS。train-domain实际场景比例89.468%/5.476%/5.057%，按log约89.796%/4.762%/5.442%。完整读入/hash采用16线程有序map，392.85秒；与原串行384条真实记录的顺序及全部字段/hash一致。没有新增观测，全量接入仍是原始数据。新Query基础正式训练只消费fit，val/cal在其训练前已固定排除。
