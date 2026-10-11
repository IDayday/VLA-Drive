# G 系列完整执行性能：已完成证据，尚未授予正式承接

24 个最终性能窗口全部完成。相同 G1/G3、数据顺序、global batch32、microbatch4、FM repeat8、真实10步proposal、8个未来位置、K1024、分辨率与精度下，`engineering_fast` 吞吐提高12.0%–15.1%，相同曝光的稳态GPU成本减少10.7%–13.1%。这不是减少任务后的速度，也不代表逐位相同的优化器更新。

| 训练状态 | IO median / P90 秒 | P0 median / P90 秒 | 吞吐增加 | 同曝光GPU成本减少 |
|---|---:|---:|---:|---:|
| G1 early | 3.198 / 3.206 | 2.857 / 2.858 | 11.98% | 10.70% |
| G1 25k | 3.229 / 3.233 | 2.853 / 2.855 | 13.16% | 11.63% |
| G3 early | 3.272 / 3.289 | 2.847 / 2.848 | 15.13% | 13.14% |
| G3 50k | 3.202 / 3.221 | 2.852 / 2.867 | 12.24% | 10.91% |

每种状态/模式各3个交替窗口，每窗口20步预热、200个稳定optimizer updates。median/P90的样本是三个窗口的最慢rank同步边界墙钟/200，不是200个异步step记录。主时间包含正常日志flush，排除末步小日志和边界外MAX collective。每窗口稳定曝光6400，完整24窗口共5280 updates/168960次场景曝光；重复曝光不称独立样本。完整进程成本41.2302GPUh，其中复用已完成旧窗口1.7913GPUh只计一次，新23窗口39.4389GPUh。初始恢复/散列包含在进程成本内；这些临时性能窗口没有末尾大检查点，因此稳态数字没有冒称包含保存摊销。

allocated峰值约31.90→30.58GiB，reserved峰值约40.81→40.00GiB；这是全部rank进程累计峰值，包含启动与预热。early起点是共享5epoch几何准备后已有2个驾驶更新的检查点，不称随机step0。

详细原始窗口结果见 [RESULTS.json](performance/final_windows_v4_p0_actual/RESULTS.json)，192份rank证据的独立检查与定义见 [FINAL_24_WINDOWS_INDEPENDENT_REVIEW_V1.json](performance/FINAL_24_WINDOWS_INDEPENDENT_REVIEW_V1.json)。

## 收益来自哪里

同G1 25k的两组完整32更新profile已实际完成，模型be5e722、adapter884e3bc，额外profile成本1.4098GPUh。每rank只记录3个active CUDA steps，不能把trace吞吐当成上表基准。

Qwen processor仍约268ms/次。参考主线程在apply_chat_template范围内出现约234–257ms无GPU活动空档；P0将真实处理移到CPU worker，最大空档约6.45ms。计算没有消失。trace后22个更新的rank-max current-stream event中位数，Qwen current813→509ms，vision288→291ms；视觉GPU计算没有减半。language CPU范围约121→123ms，proposal376→382ms、反向1645→1620ms、optimizer97→97ms，后两者与10步proposal仍是主要成本。

同口径CPU范围中，sample_queries约29.94→1.61ms，structured auxiliary losses106.31→57.41ms。范围可能嵌套、含等待、跨线程/stream，不能把这些数字逐项相加。GPU活动必须取stream区间并集；NCCL与其他kernel重叠不能全部称网络时间。最后一个active optimizer范围含85–91秒trace导出，已明确从局部optimizer/稳态解释中排除。

详见 [PROFILE_PAIRED_V2_ACTUAL_ANALYSIS.json](performance/profile_paired_mid_v2_actual_analysis/PROFILE_PAIRED_V2_ACTUAL_ANALYSIS.json) 与[中文时间线分析](performance/profile_paired_mid_v2_actual_analysis/PROFILE_PAIRED_V2_ACTUAL_ANALYSIS.ZH.txt)。原始trace留在受控zt4存储，不上传。

## 算法与执行边界

P0包含实例级Qwen视觉length tuple复用、CPU当前输入预处理与批量pinned传输、窗口日志、每forward一次MLN与TokenFuser空间mix、同标签身份的候选列表、footprint和匹配时间二维插值复用。未离线缓存可学习表示，也未打开旧teacher/cache白名单。未改q0步数、FM或监督任务、K、时刻、网格、分辨率、microbatch、LR或梯度边界。

G3改变road/current occupancy/future/body relations共同使用的结构化监督采样分布，准确名称是planning-local sampling for structured scene supervision。FGTR读取受结构化监督塑造的Bt latent；道路/占用数值没有作为硬约束直接进入执行规划。q0的detach是既有定义；refine梯度通过H_A/Bt/FGTR回传，不通过10步q0链。

时间向量化未通过原登记Bt容差，保持关闭；旧严格失败没有重命名为通过。局部几何BF16只是独立E3候选，未计入上述P0性能；全模型BF16、TF32、梯度通信精度改变、关闭重计算、proposal compile均未作为默认执行。proposal-local cross K/V实测约321→304ms且bitwise，但不足完整step的1%，未采用。

## 源码身份和未完成验收

正式NAVSIM科学/评估冻结源0c54f1c；上表实际IO与P0训练源1e5b109，benchmark driver79fde5c；数值native capture源be5e722。IO控制也继承公共执行重构，例如不请求返回时不保留DiT hidden列表，不能把上表说成实际运行未经修改的0c。静态审阅没有发现NAVSIM目标变化，不能代替完整跨源码GPU数值证明。

nuScenes实际正式源41da7c198，其校准后的配置与1e5/be5一致；旧0c debug全1正类权重不是正式Nu配方。兼容性需要绑定这个实际配置，不能混用0c debug目标。详见 [源码审阅](acceptance/REFERENCE_SOURCE_SEMANTICS_AUDIT_V1_ZH.md) 和 [Nu配方身份](acceptance/NUSCENES_FORMAL_COMPAT_RECIPE_BOUNDARY_REVIEW_V1.json)。

本快照中，500更新全量master/Adam/BN/RNG与恢复、完整dev1696验收尚未全部完成，`engineering_fast`/`precision_fast`正式scope guard保留。已完成数值结果中的原strict梯度失败保留；新增全参数AA–B统计是独立观察补缺，不能根据A–AA和A–B推算。

zt3在完成G1三组500更新后SSH不可达，G3只有已确认首步/不完整保存；旧进程与完整成本未知，不记作0、不自动杀进程。后续在空闲zt4独立登记采集，旧尝试保留。本轮新pilot checkpoint可逐payload校验后迁移至本机受控存储，释放的仅是验证过的这轮新副本；正式权重、旧checkpoint、cache和负结果均保留。

若按相同硬件与曝光线性估计，不含启动/保存/评价，G1每100k约718→634GPUh，G3约713→635GPUh。剩余正式训练须结合实际进度、保存频率、硬件负载再计算，不能把该外推当已用预算或长期数值保证。
