# 真实梯度与部署隔离

真实A800、完整参数规模、真实当前图像、真实DINO/WA目标、真实官方PDM标签。
证据目录：`/mnt/project/qwen-world-postcot-20261025/audits/`。

`REAL_QWEN_AND_DEPLOYMENT.json`：Qwen第8/16/28层与官方完整hidden_states
逐位一致，maxabs=0；临时hooks在backward重计算前已移除，有效Qwen梯度>0。
两场景当前视觉token均693，完整序列802。删除全部训练heads后，同输入同噪声
物理轨迹maxabs=0，单次部署Qwen调用恰好一次。

`REAL_GRADIENTS.json`：实际训练样本09d161aae53c5e86，先3次诊断更新打开零门控，
再分别只开启一种loss。以下为**未乘主权重**的L2梯度范数；不是训练效果证据。

| loss | Qwen vision | Qwen language | memory | actor attention | actor FFN | action output |
|---|---:|---:|---:|---:|---:|---:|
| x | 4.89477 | 3.55978 | 14.70686 | 19.99531 | 0.65021 | 18.37535 |
| DINO | 0.04477 | 0.03165 | 0.12995 | 0 | 0 | 0 |
| future | 0.12811 | 0.06583 | 0.34605 | 0 | 0 | 0 |
| object | 6.09031 | 5.80395 | 20.53235 | 0 | 0 | 0 |
| post | 0.48384 | 0.32778 | 1.17661 | 1.66989 | 0.05779 | 0 |

post head梯度范数6.82813；精确缓存轨迹、分项标签和两类教师目标均无梯度。
五个独立loss forward各只调用一次Qwen。post不训练最终action projection符合契约，
由轨迹L1训练该projection。验证范围为真实有限样本，不声称所有数据已做逐项autograd。

官方标签首轮审计：32场景512候选，另132次复算；缓存/重算、单条/batch、
候选重排maxabs=0。包含NC/DAC/TTC的0/1与变化的EP；此小审计没有覆盖到
Comfort/DDC失败和NC0.5。NC0.5保存与软BCE语义有自动化测试，不冒充真实边界覆盖。

教师冻结、无标签输入、候选间隔离、context重复索引梯度、空rank有效数归约、
clean-x端点及严格加载有行为测试。真实ZeRO与保存恢复证据单列，未完成的恢复
不按“load成功”写成exact resume。

后续真实ZeRO2验证已得到完整18文件SHA一致：保存2→新进程恢复3与连续3更新，
全模型、8个Adam shards、8rank RNG/sampler/cursor全部一致。FP32 loss修复后的
source也完成31→32同样的全状态一致检查，分别见外部证据
`ZERO2_EXACT_RESUME_COMPARISON.json`和`ZERO2_FP32LOSS_RESUME31_32.json`。
这些严格结论绑定对应源码、相同硬件和相同缓存身份，不泛化为跨后端bitwise承诺。

真实完整训练模型→独立进程strict部署导出也已通过：2个真实三视图场景，
raw/physical轨迹逐位一致、maxabs=0（预先固定容差0），删除36,852,497辅助参数；
每scene visual/language各一次，教师/训练缓存/GT/模拟器访问次数0。
证据：`audits/full_export_parity_stage02_attempt2/PARITY.json`。

最终银行元数据版本还完成真实两卡保存1→新进程2：正确v3目标、完整银行、
全部五loss；两rank的cursor32→64，后续场景窗口不重叠，三个采样RNG均继续，
Adam step1→2且skipped=0，梯度范数均有限。此项验证银行／恢复接线，不额外
宣称跨代码逐位一致。证据：`audits/BANK_METADATA_RESUME.json`。

完整最终回归为104 passed，22个实际CLI的`--help`退出0。一次回归发现共享文件
系统可能在同尺寸改写后保持粗粒度时间戳；已修复评测summary复用检查，改为
核对小型scene JSON的实际内容SHA，不每步重hash模型或巨型教师shard。
固定源码、真实检查收据及其SHA见`audits/ENGINEERING_ACCEPTANCE.json`。

最终交付的离线RGB工具额外修复同run互斥、禁止覆盖、落后恢复点拒绝和终态零更新。
完整CPU回归增至109 passed；实际完成2000的真实checkpoint也已在CPU-only环境
执行终态resume，optimizer更新0，原checkpoint/receipt/config/status的字节与mtime均
不变。它不修改已运行的RGB快照或正式IL源码；正式训练仍是d54617b的封存身份。
证据：`audits/REAL_RGB_TERMINAL_RESUME.json`及`audits/FINAL_DELIVERY_REGRESSION.json`。
