# IQE real two-round smoke

**PASS on CPU; SCIENCE_UNTESTED.** 真实原轨迹框架、真实三相机图像、真实GT、真实官方metric context已经贯通两轮。后续GPU/NCCL、真实GPU前向及8卡基础训练profile已经独立实跑通过。CPU双轮不能代替GPU双轮，二者未混称。主功能实现覆盖数据、独立专家、官方评分、Scorer、校准、多轮、恢复、Agent和导出；完整任务状态为PARTIAL，具体未完成项见下文。

按用户后续指令，复用原DiT/flow-matching S0的基础框架，改为单Query动作头，**没有加载已有驾驶S0/DiT权重**。通用初始化为本地Qwen3-VL-2B-Instruct。先用实际图像/GT和原current-DINO、future-clip、interaction辅助目标训练新的Query base 1步，再冻结它执行双轮。这仅是接口诊断基础模型；质量很低，不是正式训练S0或科学实验基线。

实际工作区 `/mnt/project/DriveVLA-M0-iqe-20261009`，分支 `feature/s0-incremental-query-experts-20261009`。仓库起点 `6e96cf7321b134c42c2cf0fbbc315cd61c925b11`；原轨迹框架源 `/mnt/project/VLA-Drive-action-video-source-1493ded`，commit `1493deda247107efe076b9294863a6753391298a`。原工作区2处未提交修改完整保留，patch SHA256为`d54c0822b288f047903059c5910401b822c0fddc01f4f994896fd16b1742a896`。没有push、PR、merge或更改其他训练进程。

## 实际资源和数据

- 主机：`training-vla-zt-worker-0`，Intel Xeon Platinum 8378A，128逻辑CPU；本任务CPU训练用2或8线程，特征缓存最多4个8线程分片；GPU使用0。
- 环境：`/root/miniconda3/envs/ddp/bin/python`，Python3.10.20、torch2.5.1+cu124、transformers4.57.0；官方评分使用`/root/miniconda3/envs/navsim/bin/python`和锁定NAVSIM v1.1源码。完整版本见`evidence/ENVIRONMENT.json`。
- Qwen：`/mnt/project/DriveDreamer-Policy/models/Qwen3-VL-2B-Instruct`。输入为原CAM_F0/L0/R0当前帧，原1024×576处理、prompt和ego字段，不减少相机/分辨率/token或辅助任务。
- 数据：`/mnt/project/ddp-foresight-artifacts/20260928/student_train_v1`与`student_dev_v1`；原DINO/future/interaction标签路径在resolved config及S0_CONTRACT中。44条实际smoke记录：fit32/30logs、stage_val4/4logs、selector_cal4/4logs、dev4/1log，共39组、44唯一观测，角色完全隔离。
- 官方训练cache：96条从原raw logs/maps重建，其中40条用于本双轮；开发4条使用已有独立官方cache。拒绝缺少官方reference trajectory的旧训练专用cache，未修改原cache。
- 没有新增观测，实验类型`resampling_only`。没有final_test访问。

## 实际执行结果

| 验证项 | 结果与证据 |
|---|---|
| 原框架新Query基础训练 | 全部原辅助目标启用，CPU 1 optimizer update，global batch1；实际消耗1条fit样本；394.88秒/update、462.59秒作业时间 |
| 真实S0与IQE K=1 | 同一新Query base、真实图像，physical max abs=0；REAL_K1.json |
| 初始化及backward | 独立clone输出一致、storage不共享；真实Query/Decoder/最终回归头有限非零梯度；REAL_FRAMEWORK_PROBE.json |
| 第一轮难例 | 旧池为Base，31条通过完整GT审计；H/A/N/G数量31/0/32/32，probe=false |
| 第一轮专家/Scorer | 各2 optimizer updates，global batch4，参数实际改变；专家12.31秒，Scorer0.28秒训练器内部累计时间；见独立阶段成本说明 |
| 第一轮银行 | 44场景×2候选=88条实际官方标签；复用44条Base标签，新增44条，66.13秒评分墙钟 |
| 第二轮重新挖掘 | 旧池为Base+expert_1，合格H降至27条；不是重复原Base列表；probe=false |
| 第二轮专家/Scorer | 新建expert_2，各2步；专家12.04秒、Scorer0.39秒；Scorer从上一轮权重warm-start、重建optimizer，实际重放expert_0/1/2 |
| 第二轮银行 | 44场景×3候选=132条实际官方标签；复用88条旧标签，只新增44条，71.13秒评分墙钟 |
| 冻结 | 两轮共享链和旧专家参数/buffer hash完全不变；固定4个stage_val场景raw、physical、ADE/FDE、yaw漂移均0；每条旧PDMS重评一致 |
| 校准与部署保护 | 两轮selector_cal均选择always-Base；CANDIDATE_GAIN_NOT_REALIZED；没有把零切换称为算法成功；gate未通过，不修改正式active |
| 导出与真实Agent | 两轮临时不可变bundle落盘；第二轮恢复，真实NAVSIM AgentInput与原像素/prompt/ego一致，官方Trajectory与恢复后缓存路径max abs=0；REAL_AGENT.json |
| 整轮恢复 | 第二轮11个阶段全部校验hash并REUSED；REAL_TWO_ROUND_RESULT.json |
| 真实head exact resume | 连续2步vs第1步中断恢复，下一批IDs/loss/实际梯度tensor/参数/Python及CPU RNG完全一致；REAL_EXPERT_EXACT_RESUME.json |
| 真实Router效率对照 | 实际Base+expert_1银行、2步更新、selector_cal独立校准、dev4评估；hooks核实4条缓存观测只运行4条选中Decoder，额外1条在线观测只运行1次共享编码/1条Decoder；REAL_ROUTER.json |
| 无损验证并发 | 同一真实专家checkpoint与固定stage_val列表，优化后的独立进程并发评分结果与原串行验证JSON完全相同；REAL_ROUTER.json |

训练器elapsed字段包含其内部训练/验证，不包含外层模型装载、hash、特征生成、银行构建、额外旧输出审计和bundle写入。CPU评分和缓存成本另列，不能据0.28/0.39秒宣称整个Scorer流水线成本。特征缓存44条已全部完成，原子恢复到4分片时复用了已完成条目，因此不从混合新建/复用耗时推导吞吐。GPU总成本为0。完整阶段时间与资源记录在运行目录的`*.stage.json`、`status.json`、`scoring_cost.json`。

每轮专家仅8次实际曝光：H/N/G为6/1/1（75%/12.5%/12.5%），原意80/10/10在这么短的计划中存在整数舍入；单bucket仅1或6次draw无法满足0.10的连续比例。曝光与退化事实单列在DATA_AND_LEAKAGE_AUDIT，冻结清单未追改。较完整配方和global batch128未在本次CPU小预算下执行。

stage_val仅4个组，dev仅1个组。第一轮stage_val Oracle边际增益为0.2006697655，第二轮为0.0076635629（零一制）；这些数值是低质量、1步基础模型上的功能观测，**不构成科学收益证据**。两轮dev最终均回退Base，实际selected均值约0.1041667；原始逐案例/安全分项/上一部署比较记录在REAL_ROUND_SUMMARIES.json。没有显著抗遗忘、泛化改善或部署安全保证的声明。

## 运行命令与产物

实际诊断配置`/mnt/project/iqe-runtime-audit-20261009/real_cpu_two_round.yaml`继承独立CPU base配置，所有resolved config均在运行目录落盘。正式main契约为另一个尚未正式训练的Query base身份，不与本诊断base混用。

```bash
cd /mnt/project/DriveVLA-M0-iqe-20261009
export IQE_PYTHON=/root/miniconda3/envs/ddp/bin/python
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  $IQE_PYTHON scripts/iqe/real_cpu_two_round_probe.py \
  --config /mnt/project/iqe-runtime-audit-20261009/real_cpu_two_round.yaml --steps 2
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  $IQE_PYTHON scripts/iqe/real_resume_probe.py \
  --config /mnt/project/iqe-runtime-audit-20261009/real_cpu_two_round.yaml \
  --round 1 --output outputs/iqe/real_resume_reproduction
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  $IQE_PYTHON -m pytest -q tests/iqe
```

测试结果：**85 passed, 1 skipped**，16.03秒；跳过项明确为GPU/NCCL资源测试。两个真实Gloo进程已验证全局有效分母、梯度、零有效rank/异常共同退出和optimizer-step exact resume；不是GPU测试。shell脚本语法和Python编译另行检查。官方入口等价和候选集合不变量在23次真实调用中通过，详见SCORE_CONTEXT_AUDIT。

收尾回归补充验证显式`--resume`一个已完成checkpoint时，不重写原result/阶段receipt；不会抹掉实际更新证据。真实Router校准选择全部Base，真实hooks只覆盖这一分配；混合专家分配的batch分组及顺序还原由单元行为测试覆盖，不能将两者混称。最终测试耗时以evidence/UNIT_TESTS.txt为准。

真实checkpoint：

- Base：`/mnt/project/iqe-runtime-audit-20261009/base_cpu_probe/query_base/step_000001.pt`，SHA256 `ff8f7905315d5d3b64e535c4799b872cd9466847a7f245d73c32ff8e8e1a266c`。
- 专家：`/mnt/project/iqe-runtime-audit-20261009/real_cpu_two_round/rounds/round_001/expert/step_000002.pt`及`round_002/expert/step_000002.pt`。
- Scorer：同目录`round_001/scorer/step_000002.pt`及`round_002/scorer/step_000002.pt`。
- 中断恢复探针：`/mnt/project/iqe-runtime-audit-20261009/real_resume_probe_v2/interrupted/step_000001.pt`。
- 临时部署包：`/mnt/project/iqe-runtime-audit-20261009/real_cpu_two_round/rounds/round_002/bundle`。

以上大权重/图像/metric cache均不进入git。完整第一轮、第二轮、Scorer、评估、恢复、导出、回滚和资源释放后的资格测试命令见`docs/iqe/RUNBOOK.md`。

## 尚未完成的验收

`training-vlawm-zt`正在运行正式structured-world科学训练，保留。`training-vlawm-zt2/3`的GPU仍被占用，但当前SSH容器内看不到对应计算PID；只读管理入口返回403，不能精确定位可终止的压力脚本。没有杀未知进程或做GPU reset。三台实际资源资格检查均以exit2、BLOCKED_GPU_RESOURCES退出，证据已保存。用户已授权训练与停止已确认压力脚本；这些主机的压力脚本未被终止；随后找到同样获准使用的vla-zt2空闲8卡，训练不再依赖解决这些容器的进程可见性。

GPU/NCCL、真实GPU框架probe、K=1/2/3/5时延和8卡吞吐资格均已另行通过。32场景/32步overfit诊断也通过；RAW L1从0.694936降到0.302886，旧输出不变。E3真实新增观测和E12可运行对齐LoRA-DiT集合仍缺失；没有执行正式navtest，科学收益UNTESTED。正式Query基础训练按用户后续授权启动并监督，最终活动PID/进度以FORMAL_TRAINING_REPORT及FINAL_STATUS为准；正式增量专家/全量银行/全量Scorer等待合格Query基础模型，未冒称完成。final_test未用于训练，smoke未更新正式部署。

全量原数据预审计已完成，103288条实际记录（101592 train-domain +1696 dev），1192个source groups，隔离PASS；16线程有序完整hash导入392.85秒。此阶段仅做来源核验，没有启动全量PDMS标签生成。GPU和正式作业成本、启动检查优化及吞吐结果单列，不能把本表CPU两轮的GPU成本0误读为整个开发会话GPU成本0。

独立新进程bundle恢复也已完成：`load-bundle`不传shared adapter，35.69秒从包内真实权重重建Base+expert_1+expert_2+Scorer，bundle hash为`d7648d6074500dae8c6434aa544f7b418624758227c023cee2295241ca2a2191`，见FRESH_REAL_BUNDLE_LOAD.txt。
