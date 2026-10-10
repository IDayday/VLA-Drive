# Gradient and freeze audit

真实框架接口探针使用原轨迹源 `1493ded`、通用 Qwen3-VL-2B 初始化和一条实际三相机观测；不使用学习好的 DiT/S0 驾驶权重。证据见 `evidence/REAL_FRAMEWORK_PROBE.json`。原 Query 框架入口与 IQE K=1 输出 max abs=0，克隆专家 raw max abs=0；新专家一次 AdamW 更新后 raw max abs变化1.0679937601。Query 梯度norm=0.1987029463，Decoder和最终轨迹头有非零有限梯度；共享链及旧专家参数/buffer hash不变。原配置中间监督权重为0，因此没有把未参与损失的中间头零梯度伪称非零。

真实 Query 基础训练另外执行1个 optimizer update，保留 ego IL、current DINO、future clip、interaction 全部目标。gradient norm=83.95825，IL=3.051523447。该阶段按原框架训练语言/投影等共享部分；冻结保护从这个新 Query S0 被 lock 之后开始。不能把原 DiT 与新 Query 初始化的不同当作“旧函数漂移”。

`test_models.py` 验证 train() 模式强制、BatchNorm/dropout buffer、optimizer参数精确集合、storage隔离、no_grad/inference tensor backward、Query-only/Adapter/residual初始化。`test_training_resume_ddp.py` 以两个真实Gloo进程验证不均匀有效标签的global numerator/denominator、梯度以及某rank缓存异常共同退出；另验证带dropout的两进程 optimizer-step边界精确恢复。

真实双轮的逐场景 raw/physical/ADE/FDE/yaw/PDMS 冻结证据由 `real_cpu_two_round/rounds/round_00{1,2}/frozen_audit.json` 生成。最终执行状态见 REAL_SMOKE_REPORT。GPU/NCCL已另在空闲vla-zt2上实跑通过；证据GPU_QUALIFICATION_TESTS.txt，不用Gloo代替。

真实私有专家 optimizer-step 恢复：**PASS**。证据 `evidence/REAL_EXPERT_EXACT_RESUME.json`；真实 Query base 权重、真实图像缓存和通过本轮官方目标审计的GT，比较连续2步与第1步中断恢复，下一批IDs/loss/梯度tensor/参数/RNG完全相等。第一版诊断误比较了按“每次作业首步”采样的梯度日志有无；已改为比较实际梯度tensor，不放宽数值容差。原始失败日志保留在运行目录，修正版独立目录 `real_resume_probe_v2`。

双轮最终结果：PASS（CPU）。两轮新增专家各2步，参数hash确实变化；每轮Scorer各2步且生成器hash完全不变。第二轮Scorer的训练前参数hash等于第一轮训练后hash，旧/新候选IDs为expert_0/1/2，证明实际warm-start和重放。固定4个stage_val场景中，第一轮Base、第二轮Base与expert_1的raw/physical max abs、ADE/FDE、周期yaw漂移均为0，旧PDMS逐条一致，包括expert_1的非零0.8026790469715103样例。证据是ROUND_1/2_frozen_audit.json，不能外推到未执行的GPU kernel。

后续A800真实框架探针也通过：原Query入口与K=1、克隆raw输出max abs均0，实际私有专家更新1.1438275576，Query梯度norm=0.19270414，旧hash不变。真实32场景overfit执行32步，raw L1 0.694936→0.302886、ADE 6.843978→5.542515m；不代表拟合已收敛或泛化提高。原language reentrant checkpoint与DDP冲突已改为non-reentrant，带dropout的完整输出/梯度等价测试通过，并完成真实8卡基础训练8步。
