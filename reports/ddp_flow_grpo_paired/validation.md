# Action-only F/U Flow-GRPO 修复验收

**工程：NOT_READY。效果：证据不足。** 本轮已修改实际训练/恢复/评估代码，并执行真实完整模型 CPU FP32 诊断和自动测试。8张A800一直由其他作业占用，每卡约77,337MiB；未抢卡、未停止其他进程。因此新BF16/ZeRO-2生产验收、两组100→2000更新、step0/100开发集及最终5seed全navtest成绩均不能宣布完成。正式入口已实际验证会拒绝缺少放行记录的2000更新请求。

本报告区分 **IMPLEMENTED、TESTED/PASS、TESTED/FAIL、NOT_RUN、BLOCKED**。源码实现不等于实测通过。历史BF16 chunk失败仍是FAIL，且不归咎于本轮GPU资源缺失。

## 版本、工作区与来源

- 审核提交：`fc354f8be62798b36330ac55edd7369f33fa949c`，修复分支包含它。
- 修复工作区：`/mnt/project/DriveDreamer-Policy-paired`，分支 `fix/ddp-flow-grpo-paired-visual-frozen`。
- 用户原有 infer.py/QwenPI 补丁单独提交 `4acd8e7`，归属为用户既有工作。新worktree先检查正向可应用，旧worktree反向检查已包含，再仅在新worktree应用一次。没有依赖旧目录的未提交 infer.py。
- 主代码提交：`d73402cf77e76c0f615250e847883dd9b509b349`；后续诊断脚本和证据提交见最终git记录。实际执行指纹保存在各run的source_environment及current_source_comparison，不以HEAD代替工作树摘要。最终runtime SHA256为 `39a764d64f87ba49f2e2a35b44bf1a1a3564a968aa9f81f50ad6608e504855a1`，包含脚本/测试的执行摘要为 `2c11cf4993105cd2ae366d850b28f70f166b733da71104a1dd5f982bf62cd7d5`，见`delivery_source.json`。晚于完整FP32模型运行的唯一内核文件变化是reproducibility.py的有界并行文件哈希；不据此签发GPU证书。
- 源action-only代码：`f9449d55bea6895a7a0bd86d09d7ab85fd353f26`；锁定Flow-GRPO参考：`879042cf5707f8b90daa98d147d7deac2317c5da`；SimWAM：`68b426c162827cb7701396895dbb3572d29f3420`；其他参考及NAVSIM文件锁见`reference_lock.json`/执行源摘要。
- 原工作区、旧Flow-GRPO分支、历史报告和checkpoint均保留；没有push、force-push、PR或依赖升级。
- 最终实现提交 `45a0837a9c18a196f2b12cfa2286e9ba00b81289` 已在干净detached worktree `/mnt/project/DriveDreamer-Policy-paired-clean` 幂等prepare并运行完整回归：**95 passed，0 failed，0 skipped，162.47秒**，见`clean_final_checkout.json`和`clean_full_assets.log/xml`。此前d73402c的干净95passed、主目录回归及最初环境缺链接导致的失败均保留。

## 修复与验证对应

| 问题 | 原因与修改 | 实际验证 |
|---|---|---|
| ratio极值/分数/quantile | trainer过去平均各microbatch/rank数值；新增metrics显式schema，合并numerator/denominator、sum计数、全局min/max、排序全部实际ratio样本后quantile | 真实2进程Gloo，不同rank范围、多microbatch单点极值和手算clip/count PASS |
| 梯度和显存日志含义 | hook累计仅是局部反向贡献，现明确命名；backend pre-clip norm与按后端公式算clip_scale分列；无法读取的post-clip norm为null/unavailable；显存逐rank及global max | 单元/源码版本检查PASS；新ZeRO真实数值NOT_RUN |
| rank0 I/O失败导致peer等待 | 单独Gloo控制面包围无内部collective的I/O/评分/预取；save_state/load_state/分布式forward原样致命退出，不套虚假总try/barrier | 五种真实双进程torchrun故障全部非零、有界退出，约3.4–3.8秒；CUDA通信/OOM故障NOT_RUN |
| exact resume只锁路径不锁内容 | processor/tokenizer/template/model config、数值profile、依赖/后端源、内容资产manifest纳入恢复身份；启动/恢复完整核验文件内容，同路径替换可检测 | processor同路径替换、数值设置/依赖变更、同大小同mtime资产替换、token集合变更均拒绝 |
| inner epoch中断 | checkpoint schema2保存pending完整链/replay/next_inner/policy_version和RNG/流游标；old chain/old logprob/advantage hash跨inner检查 | CPU小模型生产格式精确恢复PASS；真实完整CPU模型结果另见本轮inner报告；新CUDA/ZeRO NOT_RUN |
| 候选数值只报首个失败 | 收集全部参数/模块/层/Adam/master/forward更新误差，近零独立处理；保存原报告 | 两份完整模型G2/G8、K10 FP32对照已执行，详情见数值报告；原BF16FAIL不变 |
| 硬编码BF16妨碍FP32诊断 | Qwen构建允许独立CPU/FP32诊断，条件输出使用真实模型dtype；CPU action-only零辅助项在当前设备创建；保留原CUDA FP32 autocast语义 | 两份真实完整模型FP32运行PASS；新的CUDA原源ODE/SFT oracle需重测 |
| 评估输入/协议/随机性 | 显式rl_dev/navtest、token、root、cache、checkpoint、seed、协议；token噪声键；官方相邻one-stage聚合/缺失权重；逐场景paired/log bootstrap | 真实相邻cache及无相邻cache权重处理PASS，顺序/分片噪声键PASS，训练RewardService仍拒绝navtest |
| 无正式预算门禁 | 诊断<=8更新；正式/100短训需匹配代码/测试/配置/权重/资产/world/profile证据，chunk>=2禁止正式 | 实际train CLI在初始化Accelerator/CUDA前拒绝，退出1；旧代码/配置/证据变动负例PASS |

## 两组参数与优化器

F权重SHA256：`9a26685aa3838a2e1b89ab2d92997664afde4b259fed6683851f42781ad16eb4`。
U权重SHA256：`72e626e152a357ec9c478c4b253500599301b38b87758e0aebfa486ff3be21cc`。

| 初始化 | SFT trainable numel | RL trainable numel | RL tensors | 非视觉源LR→RL LR |
|---|---:|---:|---:|---|
| F frozen_visual step100000 | 2,233,120,260 | 2,233,120,260 |672|1e-5→1e-6|
| U unfrozen_visual step120000 |2,640,077,316|2,233,120,260|672|1e-5→1e-6|

两组完整name/shape/aliases/optimizer_group逐项一致，见`current_parameter_contract_comparison.json`和两套current actor/source manifests；不是只比较numel。visual通过模块参数对象id确定边界，保留各自权重，不把U视觉换成F或基础预训练模型。原tied embedding/lm_head继续冻结，inactive agent_dino_head按原构造保留且冻结，无新辅助任务。源绝对LR、betas(0.9,0.95)、eps1e-8、weight_decay0.001、clip1均相同，详见`paired_parameter_optimizer_contract.json`。

当前完整模型CPU FP32四种独立反向（RL-only、SFT-only、reference-only、total）均PASS：语言309/history4/projector4/action355，672/672存在且非零。RL使用固定正负诊断优势；reference-only使用明确标记的参数扰动；都不混入正式训练。visual/reference无梯度且全张量hash不变。两组old/current初始ratio=[1,1]、current=reference KL最大值0。这些源文件与d73402c运行内核摘要一致；不是生产ZeRO梯度认证。

## 数值证据和支持范围

详见`numerical_protocol.md`和`numerical_*`全参数JSON。原容差2e-6/2e-3未改。早期两组G2/G8全部参数、Adam矩、实际更新及固定噪声ODE对照PASS。G8全模型梯度relativeL2：F约3.85e-6、U约3.46e-6；更新relativeL2：F约6.77e-5、U约6.32e-5；同噪声ODE最大绝对差F3.58e-7/U4.77e-7。原误差与所有非逐位相同元素均保留，PASS不代表逐位一致。

提交后使用相同保存链复核最终源码的结果单独收集到`numerical_committed_*`；不覆盖早期探索记录。完整真实CPU inner_epochs=2及恢复证据单列，不与正式GPU运行混淆。代码未用deterministic保证不同BF16 batch布局逐元素相同，也未把概率FP32/最终grad.float()冒充全反向FP32。


两份完整模型均完成真实CPU Accelerate `inner_epochs=2` 保存恢复：同一G8/K10链、原old log-prob/advantages保持不变；第一次更新前ratio均[1,1]。F两次更新后range分别[0.934116,1.038489]、[0.061584,1.059636]；U分别[0.926491,1.010358]、[0.161461,1.022849]。连续/恢复的全部模型、Adam、scheduler、RNG哈希无差异，更新后同链分布和固定噪声ODE逐位一致，visual/reference保持各自初始化。完整checkpoint路径见 `real_inner_cpu_*/inner_resume.json`。两组实际调用CLI导出原格式成功，F989/U995个权重张量逐位等于checkpoint实际forward权重，原config/normalization/processor一致；见 `export_cpu.json`。CUDA原推理入口等价性仍NOT_RUN。

这批固定官方候选评分全部1，优势全部0，因此该恢复实验更新来自原SFT replay和reference项，不能证明非零官方RL收益；非零RL梯度另由明确标记的signed诊断覆盖。global batch1 CPU FP32不等于生产global16 BF16/ZeRO2。第二次更新前SFT loss上升和较大ratio偏离原样报告，不用两步曲线宣称训练稳定或性能提升。可视化 `cpu_diagnostics.svg` 标记为CPU诊断。

**当前没有已放行生产profile。** 目标是candidate_chunk1/transition_chunk1、BF16 ZeRO2、标准FP32累积/通信、global batch16、4GPU×accum4、inner2；实际GPU缓冲、Adam/master更新、重复/恢复、固定噪声ODE和跨卡缩放缺证据。chunk>=2仍为实验诊断且正式拒绝。历史`candidate_numerics.md`和实验patch逐字保留。

## 数据、评估与统一预算

官方navtrain锁定1192 logs / 103288 tokens。项目原 `navtrain_meta.json` 的10000确为目标子集；早期subset诊断固定dev514/57 logs、train9486，其输入及失败证据保留。初次盘点的756次图像缺失引用全部在另一个已有sensor root找到；不是删去场景或生成替代图像。

本轮已完成全量103288份metadata的本机兼容转换：仅本地可信pickle的NumPy模块名兼容，不升级NumPy；逐场景核对官方raw窗口/command/当前三视角文件。位姿最大差异4.44e-16，原校准场景的实际policy像素/history/prompt与旧metadata完全一致。全部metadata数值保留，仅必要的当前图像路径重定向到已存在原图。见 `full_metadata_loader.json`、`full_target/METADATA_COMPLETE`。全量准备源码、命令及输入锁已交付。

正式训练清单改为全量目标减去固定整log开发集：**RL/replay 101592；dev1696场景/16完整logs**。以预先固定hash排序选log，至少512场景且至少16 logs，排除原数值校准logs；没有按分数选样。dev与RL/replay/校准无交叉，可能已被源SFT看到。锁见 `full_target/split_manifest.json`。完整官方v2 navtrain缓存另目录后台构建，完成后自动逐内容生成不可变资产manifest和两组full配置；只准备数据，不自动启动RL或签发放行。实时进程/日志位置见交付记录。现有50,003文件subset manifest的验证结果作为历史诊断证据保留。

完整navtest **12146场景** 的官方v2缓存本轮已生成，原205场景缓存不动；独立核查token集合、每个pickle的v2字段及文件SHA全部PASS，详见 `navtest_cache_verification.json`；另检查12146份metadata和36438个当前视角原文件存在，零缺失，见`navtest_observation_inventory.json`。该结果只是评分输入就绪，**没有模型推理或完整EPDMS成绩**。

共同预算：G8/K10/noise0.1、clip0.02、advε1e-6/clip5、KL0.01、原action SFT0.1、seed42、global16、两次inner。每组2000实际updates、16000 fresh scene rollouts、128000候选、16000 replay draws/32000 replay exposures；save100、dev每200、step0/100短训评估；dev-best平分取最早；最终42–46五seed评估SFT/last/best。两组源SFT步数不同，不能从F/U绝对分数推断视觉SFT解冻的因果作用。

## 本轮执行阶段

| 阶段 | 状态 | 实際结果/边界 |
|---|---|---|
| A代码修复/CPU回归/干净准备 | TESTED/PASS（已覆盖项） |95测试、五故障、真实官方reward/相邻评分、两组真实模型CPU诊断；GPU专属项不在95中 |
| B生产profile | IMPLEMENTED / NOT_RUN / BLOCKED | GPU被已有作业占满；原BF16FAIL保留；CPU高精度与图连通不代替生产验收 |
| C配对100更新 | NOT_RUN | 未获两组生产放行；不接续历史F50/U2 |
| D配对2000更新 | NOT_RUN | 程序明确拒绝无放行正式启动 |
| E统一5seed dev/navtest | NOT_RUN | 无本轮baseline/last/best模型推理成绩；无性能提升结论 |

本轮不存在配对100/2000完整checkpoint或正式训练曲线；已有两组完整CPU诊断checkpoint、诊断曲线、真实原始权重及历史checkpoint。没有伪造分数、替换候选/GT、挑seed、删零分或放宽测试。GPU资源恢复也必须先通过当前生产profile验收才能跑100→2000。

所有可复制preflight、单卡诊断、实际4卡完整配置诊断、连续/inner恢复对照、配对编排、resume、export、dev/navtest命令在[运行说明](../../docs/ddp_flow_grpo/paired.md)。最终代码SHA、源摘要、工作区保护和未运行矩阵见本目录的最终交付记录。

本次后台仅运行全量navtrain官方缓存与后续资产锁定，PID/有界期限/日志/完成标记见 `background_assets_status.json`；它不会自动启动RL。已经完成的CPU checkpoint和导出路径集中列于 `artifact_locations.json`。修复项的25项验收状态见 `acceptance_matrix_25.json`；机器可读总体结论见 `acceptance_status.json`（不是放行证书）。
