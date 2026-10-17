# PCA256 first-5k：独立正确性增量审查

Reviewer `/root/st_road_proxy_hypothesis`，非 channel-control 配置作者。审查对象为 Lead `633fc59a6518f92e9fc0508fdb80f3be3c23657c` 中的 PCA256 配置，执行源继续使用 clean immutable runtime `fdf25f99b702754787904215bf1c2a2b7515ea75`。

**这是一项配置维度变化的 affected-part 审查，不需要重验整个 VLM、原 Flow 目标、数据生产或 native checkpoint 引擎。旧 SP2 审批可提供不变模块的证据，不能直接充当新 run 的审批。** 新实际 registration/environment/resource lease 已于本审查最后阶段绑定，见第5节；本文件只支持正确性角色，不签发三项 PASS 或 formal 启动许可。

## 1. 新 registration 的精确输入

机器可读的实际路径与完整 SHA256 在 `REGISTRATION_INPUTS.json`，无需猜测路径。必须使用：

- runtime/source：`/mnt/project/VLA-Drive-st-official-runtime-v2-20261016` / `fdf25f99b702754787904215bf1c2a2b7515ea75`。
- execution fingerprint：`bfde9af74eb615f8236d44196776f043f6293df1a85a0c01c8735ff664adcaec`。
- 256 config hash：`f71f1cc013701e1f271aad5dffc399f4162c46fabdd1ee896691e4a0f2bcd35e`。
- 256 target contract hash：`d598d570bde6b2e1725fabe5bb7df176618af48cc8c5e60fb719226ef4636de2`。
- 原 Base binding、完整 train manifest、native384 教师 manifest：继承原 SP2 的**同路径与同 hash**，不改数据/权重；它们的完整引用在 JSON 中。
- PCA identity：`268723c8da3151492fd20a81b0f6f541f348a1187222132b055f4e4a528bfd46`；固定已训练集拟合的同一基底，取前256。
- environment：在实际目标主机核对后绑定当前 receipt；仅完全相同时可以使用相同内容 hash。
- resource lease：新的有效时间、目标 host/GPU UUID 与 world4 资源范围；不能继承过期 SP2 lease。

新 run 建议 `SP2_PCA256_fulltrain_seed42_official_5k_v1`，新输出根 `/mnt/project/st-world-policy-20261016/formal/teacher_channel_control_v1/runs`。沿用 global32、micro4、world4、caller accumulation2、seed42、20k cosine、warmup1000、stop5000、save1000。新 run 从相同原 S0 新建 optimizer，不能加载384终点或其 optimizer。384 control 直接引用已完成的原 SP2。

## 2. 可以继承的真实证据

已核对原 `RUN_APPROVAL.json` 与三份角色 review 的 hash/fingerprint 一致，角色独立身份不在原 implementer 列表。原正确性 review 为 `/root/st_correctness`，SHA256 `c25e5f4efcd13d2d441897d8faf45d8a93a59e825e7f8109cf7d51898e4cca67`。

原 `SP2_FORMAL_CORRECTNESS_DECISION.md` / `SP2_FORMAL_REGISTRATION_AUDIT.json` 与其精确源版本支持复用：

1. 原 learned S0 milestone100000 的严格 FP32 master 初始化；没有 diagnostic 权重。
2. 原完整 train 的 scene/log 隔离、current/ego pin、完整 native384 teacher 覆盖与实际访问记录校验；没有必要重复扫描539GB教师 payload。
3. 同一 raw teacher 契约、固定标准化/非白化 PCA 与 channel scope；不重拟合基底，不重生成缓存。
4. 原 Flow repeat8、四卡 ZeRO2/FA2、FP32 master/Adam、全局有效元素归约、tail24 与实际159992曝光的5k路径。
5. 相同 native writer/reader 的真实四 rank model/master/Adam/RNG/cursor 恢复；保留“production FA2 跨运行逐步 bitwise 不保证”的原边界。

上述是继承既有真实证据，不是假称本 reviewer 重跑。新的256末端参数形状不改变主干或恢复算法，但必须保持新 run 自有 state identity，不能跨宽度加载旧完整 state。

## 3. 最小新增检查与首次预算停止

既有独立静态审查已确认：384配置字节一致；256仅channels、派生transform、output_root改变；真实代码的维度参数接通 PCA→dataset→readout→loss分母；shared readout初始化相同，最终output bias前缀不相同且已披露。

本次新增CPU测试目标限定为实际1.59MB PCA统计的256/384前缀、真实尺寸的轻量readout forward/backward、256有效元素分母/空mask、共享初始化RNG及head严格保存恢复。它们使用明确合成W/target，不冒充真实图像或完整Base资格。

第一次执行由256MiB逻辑读取守卫停止（exit91），保留 `CPU_TEST_ATTEMPT1.log`。256有效元素分母/空mask测试已完成；其余没有因此记为通过。根因是 `TeacherPCA.load(channel_scope=...)` 重复校验原大索引：native384的 `identity.json` 是145,979,873字节，原函数先整文件SHA再 `read_text`，单次namespace复验就超出本轮预算。没有读取教师tensor或大checkpoint，没有运行GPU。

受影响测试改为复用原SP2的完整namespace审查，只调用原 `TeacherPCA.load` 的相同 raw-contract/statistics 严格校验路径；不重新传 full-cache channel_scope。**生产训练仍使用原完整scope检查，未修改或绕过。** Lead 随后明确授权一次新的2CPU/2GiB/256MiB/120s窄补测，结果如下，首次失败记录仍保留。

| 新增 CPU 行为证据 | 实际结果 |
|---|---|
| 实际 PCA statistics 1,590,864 bytes：校验原 receipt/hash，分别读取256/384前缀 | 公共256最大绝对误差0；实际投影契约匹配PCA256配置 |
| 原始readout W `[1,144,2048]`、288位置、256输出 forward/backward | 输出 `[1,288,256]`；全部参数梯度有限且存在；W梯度norm `0.004789562895894051` |
| 共享初始化、RNG及head严格保存恢复 | shared层相同，外部RNG恢复；新head strict加载后相同CPU输入输出逐位一致 |
| 256通道全局有效元素分母、padding、全空mask、microbatch合并、过小分母拒绝 | 全部通过；没有改变原训练归约 |

共3项独立测试通过、0失败/skip。测试主体0.336s，含import总wall4.676s；最大RSS845,430,784字节、逻辑读取140,124,284字节，均在新授权范围内。CPU affinity2/3、GPU不可见。完整日志与计数见 `CPU_TEST.log` / `CPU_RESULT.json`，可执行工具为 `run_cpu_review.py`。

W和目标为明确fixture；这不是新的真实图像或完整Base测试。真实教师统计、真实256模块与梯度/保存链路已验证；未变化的真实S0、图像数据、四卡engine及native恢复继承原SP2审批证据。

## 4. 正式启动前所需的最小真实观察

无需单独重跑大模型qualification或恢复全套。独立角色对新registration fingerprint完成核验、资源审查通过后，首个正常正式update即可承担新256链路的在线观察：目标实际256通道、正确valid count、所有损失/共享梯度有限、原FP32 master实际更新、原采样流前进。异常必须按原失败处理退出，不自动改loss/precision。

这些首步观察不是“先跑未经审批训练”的豁免，也不要求额外训练一个diagnostic Base；只是利用已允许的正式作业现有日志确认新维度真实接线。已有384的训练效果不能预测256效果。后续5000端点的teacher-common误差、learned-W probe与固定dev规划/风险结果分别报告。

实际正式审核仍由原 `check_formal_approval` 合并三个独立角色的新 fingerprint；旧审批只作为 evidence artifact 引用。当前没有由本 reviewer 创建新 registration、lease、三项PASS或GPU任务。

## 5. 实际新登记的正确性角色结论

Lead 已创建真实 `formal/spatial/registrations/SP2_PCA256_fulltrain_seed42_official_5k_v1/REGISTRATION.json`，SHA256 `a644ad60477b5dd857b44069568e9b93fde68994b4862ffcbcaaae9b57e29ca6`，并执行原 `verify_registration` 的完整文件hash校验。Lead提供的完整 fingerprint 为 `FINGERPRINT.json`，SHA256 `e9582e4b1002ce9293a6a241e9f0465c274aa0e11ad62afeda5d6c2e508097c9`。

本 reviewer 随后独立核验新 config/target/Base binding/environment/lease等小文件的真实hash与已审身份，确认新 target 正是有效训练配置中的256契约；逐项重建 fingerprint；以原函数独立计算 runtime source fingerprint并检查clean SHA；确认大train/teacher manifest继承原SP2的相同路径与hash。本次不重复读取146MB教师identity，采用Lead完整hash核验和原不可变资产审批链，明确此范围。

正确性结论：**PASS，仅上述精确新登记的PCA256首5000步。** 对应机器可读角色签名为 `CORRECTNESS.json`；reviewer不在本run implementer列表内。不批准其他loss、维度、预算、数据、精度或部署；不承诺256科学收益。

Environment内容与已核实旧runtime一致；新的lease在结构上覆盖world4、5000步和28800秒，但本 reviewer不主张已独立核验实时GPU空闲、锁或空间。此责任由独立效率/资源审查和Lead启动前检查承担。必须取得另两个独立角色签名、运行原正式gate后才可启动。
