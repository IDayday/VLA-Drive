# 实际正式注册的 Evidence 审阅

**Evidence：PASS。** 实际 `C1_rollout_std_fulltrain_seed42_official_4096_v1` 符合已封存的唯一科学对照；不需要先看到质量提分再决定该证据角色是否通过。本结论只审研究依据与注册绑定，不签正确性、效率、资源准入、评价或最终启动批准。

注册：`/mnt/project/st-world-policy-20261016/formal/consequence/registrations/C1_rollout_std_fulltrain_seed42_official_4096_v1/REGISTRATION.json`，SHA `928b9788404c4dcb9316b8457dc1a627010f892b85308f52aacf07b2cbb0783f`。实际运行源码为 `751fcfa2b41549ae16f7b3204aef8dd18b912016`，执行 source fingerprint 为 `569a6a69374bd0cd25bbafd91489c5f07c78de9f9201b7268a391bbd3d68a9db`。`FINGERPRINT.json` 是该 runtime-v2 源码的成熟 `check_formal_approval.verify_registration` 实际返回值，不是手写或从另一个注册复制的 fingerprint。

与历史 C1 的递归差分结果保存在 `BINDING_DIFF.json`：CONFIG 只有 `/fpo/advantage` 与说明文字 `/runtime/official_execution/adaptation_reason` 变化，TARGET 只有 `/policy/fpo/advantage` 变化。execution 全部相同；base binding、TRAIN manifest、environment、NO_TEACHER 四项 artifact hash 相同。run 名与预登记建议标签的 `rollout_scale`/`rollout_std` 拼写不同仅属命名，没有新增实验臂。

共同原 S0 learned FP32 masters、完整101592 TRAIN、seed42、global32/world4/micro4、G4/MC50、BC1/policy1、原精度、4096 LR轴/warmup100、canonical score协议均保留。唯一科学干预为 `scene_center_rollout_std_v1`，由封存提案 `b1a8cf49` 定义；不新增per-scene std、全局分子center、cap、epsilon/权重/seed搜索、reward或结构变化。共同原S0新建optimizer；SMOKE/PROFILE权重不进入正式起点。新组自产on-policy反馈，历史C1只复用为已完成legacy控制，不将其账本作为新组永久训练数据。

历史控制复用继承独立 `rollout_scale_runtime_correctness_v1/LEGACY_CONTROL_BINDING.json` 的有条件等价结论；本次实际科学配置差分满足其条件。正式效果主比较仍为新组@4096减C1@4096，原DEV1696/16logs、单轨迹Euler10、scene seed42、FP32 master eval/TF32 off，以及既有配对log-cluster统计、全部安全/进度分项、双向零分/安全转移和停车代理。实际训练注册不替代未来共同端点评测注册；Lead仍须绑定真实checkpoint/DEV/scorer身份，不能按中间分数换终点。

完整TRAIN的数值/四窗口/安全支持已由封存 evidence decision 判定满足资料条件。仍保留明确风险：safe-EP-only与D/C/T可区分update的scale中位数比78.33，方法改变跨update相对权重，并非无损修复或安全保证；优势值协方差不是参数梯度份额，固定3584的policy/BC比例不代表全训练，也不选择固定1600倍。一个seed与有限已见开发人口不能证明跨seed泛化或安全不劣。

本lease声明每次最多8小时、最多6次invocation；**整个run累计终点为4096，不是每次再训练4096**。恢复必须沿同一native optimizer/scheduler/RNG/data cursor与统计identity继续，不能重置时轴。六次上限目前写在lease purpose文本内；实际控制器执行约束、当前host/资源可用性、真实240-update PROFILE成本和生命周期由独立效率/系统审阅负责，本Evidence不替其验证或批准。Evidence PASS本身不满足正式入口的三角色与Lead `RUN_APPROVAL` 要求。

验证读取边界：按本次明确要求运行成熟验证器，流式hash注册artifact，其中原TRAIN manifest为176,439,631 B；不解析其完整JSON、不读取模型、图像、metric cache、反馈payload或DEV逐scene数据。未运行训练、评分、bootstrap、GPU或新质量实验。前两次核验被本作者自加256MiB虚拟地址限制阻断Git线程/pack映射，未产生fingerprint/PASS；失败记录保留。移除不适合Git mmap的地址限制后，保持30秒CPU/60秒wall与进程内禁用Git索引预加载，原验证器成功：1.7206秒wall、0.2185秒self CPU、峰值self RSS32,236KiB。这些不含子进程总CPU/总RSS，也不是训练profile。

三次artifact流式hash合计529,393,878 B；没有把失败尝试从成本中删除。原验证器、源码、注册、方法与数据均未修改。只生成本owned报告和该注册的 `reviews/EVIDENCE.json`；审阅者 `/root/consequence_next_hypothesis` 不在注册implementer名单中，不签其他角色。精确依赖均由最终 `EVIDENCE.json` 的实际路径与SHA绑定。
