# Qwen-World四架构：完整IL100k与Flow-GRPO RL50k

本轮是四组完整主实验；当前仍处于实现、数据完成与真实GPU资格阶段，不能记为100k/50k已完成。所有四组都保留完整预算，不按早期性能淘汰。训练seed42，推理seed2101。当前实际日期2026-10-29，因此隔离分支与目录使用20261029；未修改旧实验工作树。

## 结构与共同输入

|组|动作路径|P/W与R|部署|
|---|---|---|---|
|A|官方Qwen3-VL-2B→C160→独立20×1024 clean-x actor|20×384 WorldStream只训练；后置R读取有梯度的candidate逐层hidden|删除整个WorldStream及readouts|
|B|Qwen→C160→单套20×1024共享P32/A8/R16|P固定动作无关时间；A用sigma；R用clean条件；逐层真实KV|保留P/共享actor，删除R专用参数|
|C|Qwen→C160→20×384 W与20×1024 actor独立流|20层一对一，四组共享world KV、双softmax、共享actor Q/O、小非零gate|保留W和world→actor桥，删除R反向桥/readouts|
|D|原生28×2048 Qwen语言层直接处理C/P/A/R|原Norm/GQA/mRoPE/DeepStack，A-only256维时间FiLM；共享文本主干|保留Qwen/P/连续action与time接口|

三视图仅当前left/front/right、导航、合法ego。没有旧DDP query/history/8×4编码，也不加载旧驾驶权重。A/B/C的scene provider与旧实现共用，只有C160，A明确拒绝旧StudentMemory.actor_memory/predictive拼接。D不使用C160或独立actor。

A/C共用同一WorldStream实现和seed242；scene provider统一seed142；共同readout各组件使用342—346，避免C额外bridge改变共同head的初值。通用Qwen权重revision89644892e4d85e24eaac8bacfd4f463576704203与旧初始化收据相同。四组都是新IL，旧IL40k不是本轮parent。

## 共同监督与数据身份

完整TRAIN90890、DEV1696、Navtest12146；按原封存物理log隔离，Navtest为44个物理log。复用DINO current[3,9,16,1024]和WA future[3,4,8,16,512]离线teacher。四个WA tubelet保存其真实两个帧timestamp，不冒充1/2/3/4秒。

新增未来track为当前可见最多32对象，按稳定track token读取1/2/3/4秒实际timestamp，未来状态[x,y,length,width,sin,cos]固定当前ego frame。Hungarian只匹配当前一次，未来沿同一assignment，不把未知coverage的未匹配query设负样本。

**坐标v1已拒绝**：首次准备用matrix atan2而非原Scene._build_ego_status的Quaternion.yaw_pitch_roll。虽然首个gt_boxes函数oracle通过，未覆盖ego构造；全链复核发现TRAIN未来有效点平均xy差2.67mm、最大28.41cm。v1完整资产保留并旁路标REJECTED，未用于正式训练。正式版本targets_v2_native_yaw重新全量构建/校验，TRAIN manifest4d6e4342b75c8e2f36ab7ec4e8336fce87b30561fbdb00d71ee4b155e235089d，DEV01565ddeb78a7070c08dcf2870148f3555df4194a820fd2da03bc25c196ee2b3。

固定公共候选库每scene16条、共1454240条；不做各组refresh。来源包含实际记录轨迹、公共策略和连续物理扰动。原bank_hash356d24a0e8e9e6095a9dd1f8d2d96fe39394ca51408124f6d51b62cd99f95b7f。原生事件已全部重新提取，evidence bank e15caed2fcf4ac02b30b8ae33cc85447e58f55c3d4bc4010e8025d978c2f8364，manifest0dd4c096a36135dac5f6269e8b87a1b123b2accf7bf9c4a3f3db2d748c6db546。24个CPU worker在rl-zt2实际完成，CPU/失败/复用账本与部署资源分开保存。

NC/TTC首次事件来自官方scorer API；DAC来自官方0.1秒non-drivable-body数组，半秒9类含none。32条真实候选的逐条官方评分 vs chunk1、chunk4逆序、chunk16全部总分/六小分/模拟状态/事件时间零差；另测到真实NC3.6s、DAC.7s、TTC2.7s，与官方一致。每条候选都独立与固定PDM参考归一化，没有G8共同EP池。

## IL数学和有效曝光

sigma=5u/(1+4u)，x_sigma=(1−sigma)*x0+sigma*epsilon，模型直接预测x0，FP32 valid-coordinate L1；没有1/sigma²权重或velocity MSE。部署step为sigma_next/sigma*x+(1−sigma_next/sigma)*x0。

共同总损失为10*x + DINO + .5*WA_pre + .1*object_current + .2*future_track + .25*WA_post_factual + .2*post_evaluation。post_evaluation=factor+.5*total_score+.5*pair_rank+.5*event。factors保留NC=.5 soft label；score SmoothL1、ranking score差≥.02才监督；event9类CE。invalid标签在BCE/CE/差分前mask，FP32 target不先量化为BF16。

每global32只选8个scene做post，每scene1条实际记录动作＋2条bank候选。选择与FM noise/sigma/128个有效teacher query按update与逻辑scene位置固定，micro1/2/4不改变曝光。只有事实轨迹可用真实未来WA监督；非事实候选只用native指标/evidence，不能套日志未来视频。前置readout只读取P/W，后置只读取精确candidate replay后的R，不绕过reasoning。

100k有效AdamW更新含1000接口阶段，global32、repeat1；预训练语言/merger1e-5、vision1e-6、新模块1e-4，3%warmup/cosine，betas(.9,.95)、eps1e-8、wd.01、clip1。参数/optimizer FP32、BF16 forward、state/sigma/loss FP32。D新接口通过冻结/zero-LR的语言层保留梯度，不被旧no_grad包住；阶段边界仅初始化预训练组Adam，不重置新接口。

每5k完整DEV，全部20个点和100k完成后选各自parent，最高canonical PDMS，平分选更早。100k端点和DEV选择结果均报告；不以Test选择parent。

## RL数学与缓存

继承已验证SourceLoss/SDE/relative clamp/四inner数学。G8/K10/shift5，[0..6]ODE，[7,8,9]随机，noise.1/minstd.001/创新±5/逐维relative logprob width7后H/D mean，PPO clip.005，全fresh rollout跨rank原始reward population std、组内中心化、advclip5、tied PG mask但anchor保留。唯一x→v=(state−pred_x0)/positive_sigma在FP32执行一次。reference transition mean weighted MSE beta50、dim[.2,1,1]，不除std²。clip1；无BC/aux/dual/scorer reward。

A只actorattention LoRA；B共享attention；C actor＋在线world attention＋worldKV；D仅原生文本Q/K/V/O。r16/alpha32/dropout0、按对象身份去重，reference永久冻结在本组IL。B/C/D当前P/W/文本prefix每次logprob有梯度重建，只缓存不依赖LoRA的当前输入。D同scene候选packed成隔离的A/R块，共享一份CP；没有上一步A KV累加。

50k有效updates = 12500 fresh global rollout = 100000 scene曝光 = 800000新候选。四inner固定old chain/oldlogp/reward/adv。source裁剪概率与部分随机步骤是经验surrogate，不能写作最终轨迹精确likelihood。

新恢复schema只接受新8×3及四字段current descriptor；checkpoint不保存可训练prefix图。COLLECTING失败只保存证据不推进latest；原生inner2/3、Adam/scheduler/RNG可恢复。旧验证器和旧训练入口未修改。

## 实际验证与下一执行边界

目前CPU针对性测试已覆盖四结构、full-vs-cached输出/梯度、原生D/Qwen tiny、scope、实际sampler mean/std、fresh ratio、四inner、损失分母、all-invalid、strict load/export、DEV选择、恢复事务和调度。最新一次统一执行207项通过；后续补充的集成/控制器检查单列，不能将重跑计成独立证据。FP32 full-vs-micro梯度相对L2约7e-8—2e-7；BF16布局差异约.2%—.67%、cos>.99998，不能宣称跨布局bitwise；RL fresh-ratio门槛仍1e-4未放宽。

全尺寸meta总参数（含训练readout）A2710801843、B2661686323、C2713958855、D2220489267。meta计数不是实际CUDA峰值或吞吐。实际八卡qualification、原生连续4 vs2+resume2、DDP/ZeRO2性能与最终训练结果由下阶段资产填写。

## 真正存在的入口

在该分支worktree且使用已锁定venv Python：

```bash
python tools/qwen_world_compare/campaign.py plan --config configs/qwen_world_compare/campaign.yaml
python tools/qwen_world_compare/campaign.py prepare --config configs/qwen_world_compare/campaign.yaml
python tools/qwen_world_compare/campaign.py verify --config configs/qwen_world_compare/campaign.yaml
python tools/qwen_world_compare/campaign.py run --config configs/qwen_world_compare/campaign.yaml --architectures A B C D --stages il rl evaluate --resume
python tools/qwen_world_compare/campaign.py report --config configs/qwen_world_compare/campaign.yaml
```

正式owner必须从clean immutable source启动；每架构真实GPU资格收据和实际source hash相同才派正式IL。部署/评价/profile/恢复真实入口及CLI帮助随代码提交；运行日志保留实际argv，不把本段命令文本当成已执行证据。当前尚无本轮IL100k、RL50k或Navtest分数。

来源、许可、原样计算/接口适配/主动研究设置见upstream_lock.json与各架构实现报告。UWM固定git树未发现LICENSE/COPYING，因此只参考设计，不复制其源文件。AUGUstus为Qwen3.5参考，本地仍调用锁定Transformers4.57.0的Qwen3-VL；没有迁入其Norm或视频大骨干。所有源权重、缓存、raw场景与大候选库均不入Git。
