# IQE 工程实现

IQE 仅为代码简称。用户后续明确要求复用原轨迹 S0 **框架而非权重**，将 DiT/flow matching 改为 Query。因此本实现分为新 Query S0 的基础训练，以及锁定它之后的增量阶段；不把随机 Query 或旧 64-query 的一个切片冒充已训练 S0。

实际起点为 `VLA-Drive-action-video-source-1493ded` 的 `DDPActionVideoForesight`。继承 Qwen3-VL-2B、三当前视角、16:9/1024×576 输入、history/world/action token、ego 编码规则和原归一化统计。基础训练保留 current DINO、future clip、interaction 辅助监督。唯一明确的主目标变更是 flow matching → 单 Query 轨迹 IL。旧源码、入口、历史权重不改。通用 Qwen 初始化是允许的基础预训练，未读取已学习的驾驶权重。

基础动作分支是 Q-Former 压缩、scene queries、ego encoder、独立 singleton Query、M0 TransformerDecoder、全部中间与最终回归头。最终张量为最后回归头，raw `[T,4]` 为 normalized x/y/sin/cos；physical `[T,3]` 为 rear-axle 自车相对坐标、米/弧度，实际 T=8、dt=.5。所有尺寸均由契约传递。

锁定 Query S0 后，`E0(x)=(F,e,m)` 始终冻结并保持 eval。每个专家深拷贝 Base 动作分支的 Query、Decoder、全部回归头；无跨专家 attention、共享可训练 storage 或 optimizer 状态。专家直接继承 Query S0 的 L1 轨迹损失，含中间监督递推；不进行跨专家 WTA。新增专家可以改变，旧分支参数/buffer 必须逐字节一致，数值输出和实际官方评分另行审计。

数据按 source log 与 parent group 隔离。既有独立 dev 保持独立，允许训练 logs 按稳定 hash 分成 90/5/5 的 fit/stage_val/selector_cal。新 Query 基础训练仅使用 fit，因此此后的两个留出角色未进入其训练。若接入其他已经见过数据的 S0，SceneRecord 的 exposure 必须如实记录，不能从划分名称推断未见过。

每轮对整个已接纳旧池计算质量覆盖。safe high quality 要求有效 R≥.8 且 NC/DAC 满分。已有好候选但未选中是 selection_gap；没有可靠监督是 target_gap。只有 coverage_gap 且目标通过有限性、坐标时间、输入一致性、官方安全分项、质量及收益/修复条件才能进入主难例。失败目标保留审计队列。没有难例时科学阶段报 NO_ELIGIBLE_COVERAGE_GAP；smoke 可显式使用审计通过的 GT probe，记录其非科学身份。

H/A/N/G 默认为 40/40/10/10；没有新观测时是 80/0/10/10 的 resampling_only。缺检索支持再移至 H。G 必须存在，只有显式 anchor=0 配置可关闭。合法 A 存在而 H 为空时，H 份额明确转至已审计的 A。按组累计均衡配额、固定每 observation 的 target_id。检索仅用 fit 中冻结的 pooled scene + ego、训练统计标准化；不把离线 map/actor/未来标签加入部署输入。

评分严格继承 NAVSIM v1.1 PDMS。每条输出单独与 cache.trajectory 的正式参考上下文评分；每次新建进程、解压独立 cache、创建 simulator/scorer。它是原正式非响应交通协议，不能称 v2/EPDMS。旧训练缓存缺参考轨迹时，`prepare-metric-contexts` 从真实日志、地图重建官方 cache，不拿 GT 替代参考。独立 candidate key 不绑定整个池，score key 保留 context、reference、protocol、seed、repetition；当前确定性 v1 的训练标签使用 repetition 0，其他重复记录完整保留。

Scorer 只读 detach 的 F/e/候选实际物理轨迹。每候选的时间 token `[x,y,sin(yaw),cos(yaw)]` 经两层 cross-attention Decoder 得到直接 value 和命名 component logits；只在单候选时间维 attention。K 展平进 batch，不读 expert ID、真实分数或未来环境。损失为 Huber(.1)+有效分项 BCEWithLogits+.2×按场景归一化的 gap-weighted ranking。所有 rank 使用真实全局分母，空标签/空 pair 保持图内零梯度和一致 collective。

Scorer 每轮均匀采样 fit 场景、重放全部旧新候选；第二轮 warm-start 网络权重、重建 optimizer/scheduler。stage_val 依安全退化、selection regret、selected PDMS 选择 checkpoint。只在自然分布 selector_cal 搜索收益 margin 和 NC/DAC 阈值。经验预算同时相对 Base 和上一部署版，包括错误替换、高分转零、新 NC/DAC 违例；默认新增 hard-safety 预算 0。无可行改善继续保留上一 serving bundle，always-Base 不计科学成功。

Router 为效率对照：读取 F/e，按真实分数 soft targets 训练，新增行保留旧输出层初始化；先决定专家再按子 batch 运行被选 Decoder。它独立校准 logit/probability margin。Oracle/uniform random 仅离线评估，无法导出。

checkpoint 在 optimizer-step 边界保存 active weights、AdamW、scheduler、sampler consumed cursor、全部已消费 IDs、各 rank RNG、world/batch/accumulation、环境及依赖。恢复默认拒绝数据/世界大小/配置变化；显式 non-exact 模式重建训练状态。缓存、manifest、状态及权重原子写入，完成标志与 hash 一起决定复用。部署包无 optimizer/标签/metric cache；保留原始模型代码和 tokenizer/generic model construction 依赖，当前不是脱离原框架安装的单文件模型。

科学接纳需 Oracle mean gain≥.005、组 bootstrap 下界>0、新安全成功≥10、至少10组，以及全部冻结审计。发布还需对 S0 和上一部署均无新增真安全违例/高分转零，并有正收益 CI。smoke 永不更新正式 active。资源和科学 gate 限制实际运行/发布，不省略后续 Scorer、多轮、恢复和导出实现。
