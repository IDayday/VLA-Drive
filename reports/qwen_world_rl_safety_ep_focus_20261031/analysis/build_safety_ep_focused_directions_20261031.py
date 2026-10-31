"""Build a CPU-only proposal from existing SafetyEP evidence and primary sources.
No training/evaluation dispatch, journal traversal, model load or process action.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os
R=Path('/mnt/project/qwen-world-rl-optimization-20261030');A=R/'analysis';OUT=A/'SAFETY_EP_FOCUSED_DIRECTIONS_20261031.json'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fp(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
E={name:read(A/name) for name in ['SAFETY_EP_CREDIT_COUNTERFACTUAL_20261031.json','ZERO_RESCUE_NEXT_DECISIONS_20261031.json','TRAIN_PREFERENCE_ZERO_TIE_DIAGNOSTIC_20261031.json','SCENARIO_SAFETY_EFFICIENCY_AUDIT_20261030.json','RL_LR_NATIVE_AUDIT_20261031.json','RL_LR_PRIMARY_RESEARCH_20261031.json']}
cp=A/'SAFETY_EP_PROCESS_DIAGNOSIS_20261031.json';process=read(cp) if cp.exists() else None
if process is not None:E[cp.name]=process
credit=E['SAFETY_EP_CREDIT_COUNTERFACTUAL_20261031.json'];old=E['ZERO_RESCUE_NEXT_DECISIONS_20261031.json'];obs=E['SCENARIO_SAFETY_EFFICIENCY_AUDIT_20261030.json']
S=[
 dict(id='FPO',title='Flow Matching Policy Gradients',version='2507.21053v2',sourceURLs=['https://arxiv.org/html/2507.21053v2'],
  supported_claim_zh='用CFM损失差构建policy-ratio代理，避免计算flow的精确likelihood；可与组相对优势兼容，采样器选择与更新目标可解耦。',
  limits_zh='代理不是端点精确likelihood或精确IS；无优势差异仍无PG方向，也不自动保证ODE部署提升。'),
 dict(id='FPO_PLUS_PLUS',title='Flow Policy Gradients for Robot Control',version='2602.02481v2,2026-10-13',
  sourceURLs=['https://arxiv.org/html/2602.02481v2#S3','https://arxiv.org/html/2602.02481v2#A3.SS6',
   'https://github.com/amazon-far/fpo-control/blob/8f460e4415a2a6debfbeb689093f5e36140bd178/isaaclab_experiments/isaaclab_fpo/isaaclab_fpo/algorithms/fpo.py',
   'https://github.com/amazon-far/fpo-control/blob/8f460e4415a2a6debfbeb689093f5e36140bd178/manipulation_experiments/src/flow_model.py#L550',
   'https://github.com/amazon-far/fpo-control/blob/8f460e4415a2a6debfbeb689093f5e36140bd178/manipulation_experiments/finetune_online_rl.py#L1205'],
  official_commit='8f460e4415a2a6debfbeb689093f5e36140bd178',
  supported_claim_zh='逐MC-sample分别构造ratio再clip；机器人微调中普通PPO分支优于直接迁移ASPO的若干任务。文中也诊断额外去噪噪声使低成功率基座更难收到成功反馈。',
  limits_zh='这是机器人环境实证而非World驾驶保证；自适应LR实验用噪声预测变化代理，不是真实policy KL；不能据此搬数值clip或原样叠全部组件。'),
 dict(id='SIL',title='Self-Imitation Learning',version='1806.05635v1',sourceURLs=['https://arxiv.org/abs/1806.05635'],
  supported_claim_zh='过去的成功经验可辅助后续学习，稀少成功不必在下一批失败后完全丢失。',
  limits_zh='原SIL是off-policy actor-critic/return-gap算法；本文提议verified-positive CFM只借成功复用动机，不冒称原SIL。'),
 dict(id='AWR',title='Advantage-Weighted Regression',version='1910.00177v3',sourceURLs=['https://arxiv.org/abs/1910.00177v3','https://xbpeng.github.io/projects/AWR/index.html'],
  supported_claim_zh='回放经验可用于独立的加权监督policy更新；这是与on-policy PG不同的利用历史经验途径。',
  limits_zh='AWR包含value拟合与优势加权；没有value的positive CFM不是AWR，历史trajectory绝不能伪装为当前GRPO样本。'),
 dict(id='CPO',title='Constrained Policy Optimization',version='1705.10528v1',sourceURLs=['https://arxiv.org/abs/1705.10528'],
  supported_claim_zh='安全约束与任务回报可分别表达，标量回报上升本身不足以证明安全约束改善。',
  limits_zh='其约束/信赖域假设不能直接移植为本工程的逐候选安全保证；下述lambda门控不是CPO，不宣称其理论保证。'),
 dict(id='RUN_THEN_WALK',title='Sometimes You Gotta Run Before You Can Walk',version='2609.25831v2,2026-10-07',
  sourceURLs=['https://arxiv.org/html/2609.25831v2',
   'https://github.com/haha-yuki-haha/AutoDrive-P3_with_Run-then-walk/tree/2f99d85782995f175d5a117eb45356bff34e343a'],
  official_commit='2f99d85782995f175d5a117eb45356bff34e343a',
  supported_claim_zh='驾驶中进度发现与安全修复存在不同优化阶段；有效更新仍需要当前采样支持中的安全进度候选。',
  limits_zh='原方案同时更换reward/reference/阶段LR，不能归因为某一个组件；其Walk gate可允许NC=.5，和本Safety严格NC=DAC=TTC1不同；不建议照搬进度优先放松安全。'),
 dict(id='ASYMMETRIC',title='Asymmetric Actor Critic for Image-Based Robot Learning',version='1710.06542v1',sourceURLs=['https://arxiv.org/abs/1710.06542'],
  supported_claim_zh='训练期privileged信息与推理期图像输入可以严格分离；训练监督的可用性不代表actor推理时观测到真值。',
  limits_zh='原文是full-state critic，本文若做训练期条件监督只是借鉴信息分离，不声称已实现该actor-critic。'),
 dict(id='LBC',title='Learning by Cheating',version='1912.12294v1',sourceURLs=['https://arxiv.org/abs/1912.12294'],
  supported_claim_zh='驾驶teacher可在训练时使用privileged场景信息，student仍以视觉推理。',
  limits_zh='它不证明单帧可恢复被遮挡目标或真实邻车速度；本任务不据此新增推理真值输入或重做整套IL。'),
 dict(id='TRICKS',title='Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning',version='2508.08221v3',sourceURLs=['https://arxiv.org/abs/2508.08221v3'],
  supported_claim_zh='统一配置逐项隔离检验是应迁移的研究方法，基座/数据变化会影响技巧是否有效。',
  limits_zh='LLM token/sequence likelihood、长度与离散entropy的数值经验不能直接当连续flow transition/CFM ratio配方。'),
]
sources={x['id']:x for x in S}
def urls(*ids):return list(dict.fromkeys(u for i in ids for u in sources[i]['sourceURLs']))
common=dict(initialization='same sealed IL70,current+reference initialtrainable closure; noRL10k weights/nativefork',
 parent_model_sha256='7dda5b14b1ea67fc631b06c7fa336950fa07a0b84c969f4e714bdeb5b5436d40',
 fresh_AdamW=True,fresh_RNG_seed=42,initial_optimizer_updates=0,initial_scene_cursor=0,
 data='full103288 eligible uniform,world8 scenes/collection,G8 candidates/scene,8ranks,4inner reuse',
 source_baseline='SafetyEP with W3 rigorousSDE eta.1 steps7/8/9; originalIL70 physicalclean-x mean anchor beta50',
 scope='52,707,840 = actionLoRA5,242,880 +scene/predictiveMemory47,464,960; Qwenfrozen',
 optimizer_groups='action5e-5/Memory5e-6;100optimizerupdate linearwarmup thenconstant; no secondwarmup across native stage boundaries',
 advantage='fixed kappa0,lambdaEP1; group-centered Safety+within-feasibleEP; onceglobal64A populationstd+1e-8; sourceAclip5',
 evaluation='official fullNAVTEST12146,singleODE10 shift5,seed2101,44physical logs paired2000bootstrap seed42; canonicalmetrics unchanged',
 budgets='proposal only: first5k,1k/5k gates; extend to10k only afterrootreview; later2k cadence onlyifexplicitlyapproved',
 control_reuse='Existing freshIL70 SafetyEP same-update results may be reused if scientific/data/inference identities match; no gratuitous rerun or newtrainingseeds.',
 high_retention='predefine IL70score>=.9 NAVTEST diagnosticcohort; historicalSafety10k>=.9 secondarycohort only. NoNAVTEST trainingtoken selection.',
 low_repair='predefine IL70 zero/positive-low TRAIN diagnosticcohorts; distinguish persistent/intermittent/recovered/relapse and never-resampledunknown',
 promotion='sameupdatePDMS gain with44logCI plus NC/DAC/TTC/EP andhighcohort/newzero review; smallgain ortrainingrewardalone insufficient; formal noninferioritymargins rootmustprespecify',
 deployment='singletrajectoryODE,unchangedparameterclosure; nooracle scorer/reranking/teacher at inference')
P0=dict(priority='P0_DIAGNOSIS_BEFORE_CHANGING_REFERENCE_OR_STACKING_METHODS',status='DESIGN_ONLY_NOT_MEASURED_NO_GPU_RUN',
 problem_zh='候选信用修正不等于更新后策略安全；稀疏成功/局部zeroPG、PPO阻塞、不同rank梯度相消和anchor拉回是不同机制。',
 unknown_zh='真实World同批PG与beta50物理anchor梯度的幅度/夹角尚未知；scalar pg_loss/kl_loss及Adam二阶矩不能完成归因。IL70不刷新的pullback可以是假设，不能先称reference错。',
 measurements=[
  'gPG=(1/8)sum_rank grad(LPG_rank), gAnchor=(1/8)sum_rank grad(50*Lphysical_anchor_rank)，保留实际source local-effective reduction。',
  '报告||gPG||、||gAnchor||、cos(gPG,gAnchor)、||gPG+gAnchor||，分别全scope/action/Memory；同时记录总gradclip前后与Adam实际参数步长，loss标量不能替代。',
  '按同一freshbehavior中的mixed/allfail/allpassEPsaturated分层，区分局部贡献与8rank聚合；分层只是诊断，不重分配rank权重。',
  '记录每inner的正/负A实际clip阻塞、Aclip5触及率、ratio有效权重、global8rank全A0与local0比例；4inner共用同A但ratio/梯度可变。',
  '若另拆Asafe/AEP梯度，必须保留Atotal真实clip分支与Aclip后的共同系数，不能用不同分支loss假称精确可加分解。',
 ],
 feasible_execution_zh='先复用已记录字段。若不足，root以后授权的fresh试验在少量正常collection上测量，复用已有forward/行为材料；不额外造GPU资格矩阵，不热改当前Safety。',
 CPU_fact=credit['current'],
 decision_rules_zh=[
  '新1280组allfail正unsafe24个（NC0/.5均0），旧10k1000组有5个NC0正A。频率/窗口不同，分别审原向量；不把旧计数或理论NC.5例子伪装最新主因，不一刀切屏蔽leastcost正A。',
  '若gAnchor明显大且与成功PG方向冲突：才有依据提出独立beta/reference实验；仍须从同IL起点匹配，不能混入LR/teacher/sampler。',
  '若PG有方向但部署仍zero：优先候选到ODE的成功传递/更新目标，而不是把所有zero解释为探索不足。',
  '若costtie且没有成功witness：现有数据不含可靠方向；保持anchor、标记缺证，不靠随机优势或降低安全阈值制造梯度。',
 ])
P0['CPU_fact_window']=credit['window']
D=[
 dict(id='CREDIT_COVERAGE_GATE',priority=1,title_zh='低成本：按同组可行覆盖度调节EP信用',status='CPU_COUNTERFACTUAL_READY_NOT_REGISTERED',
  problem_zh='SafetyEP虽已阻止mixed unsafe获得EP正信用，但少数可行候选与多数失败共存时，低EP可行轨迹可能同时被安全项奖励、被EP项拉低。它是否是新NC退化主因尚未知。',
  evidence_ids=['SAFETY_EP_CREDIT_COUNTERFACTUAL_20261031.json','SAFETY_EP_PROCESS_DIAGNOSIS_20261031.json'],
  mechanism_zh='只在当前组可行覆盖弱时降低EP竞争，保留先获得更多可行解的信用。p=fmean来自已评分候选，是局部采样覆盖proxy，不是已校准场景风险概率或新增actor输入。',
  hypothesis_zh='若弱可行组的EP压力妨碍安全模式巩固，lambdaEP=p可能改善NC/DAC/TTC和新zero，同时allfeasible组仍保留效率学习。',
  single_core_change='Araw=Asafe + p_group*AEP, p_group=(1/8)sum f; same onceglobal64std+eps; allothermath unchanged',
  already_done_not_new_zh='原Safety已严格within-feasibleEP；本方向只改变组条件系数，不重复注册原有gate，也不全局提高EP权重。',
  CPU_effect=credit['candidate'],
  scale_caution_zh='447/1000组数值变化，但只有21/8000个A正负号变化；未必主瓶颈。allfeasible/allfail原始A未变，共享globalstd仍使绝对尺度联动。',
  design=dict(common=common,arm='one proposedfreshIL70 SafetyEP credit-gate versus exactfreshSafetyEPcontrol',
   not_changed=['canonicalreward','samplerW3','dataorder','referenceIL70','beta50','LR/scheduler','scope','noCFM/teacher'],first_go='P0及最新PROCESS支持weak-feasible类值得优先才注册'),
  success_metrics=['canonicalPDMS samebudget pairedCI','NC/DAC/TTC repair/newfailure','fixedhighcohort retention/newzero','EP/rawprogress withinverifiedfeasible opportunities','PG/anchor gradcos andglobalbatchscale/clip'],
  risks_zh=['p由8样本估计，受SDE噪声影响且并非真实可行率。','降EP压力可能强化保守停滞；低EP不能直接判无故停车，须有同场景strictsafe更高进度见证。','原平均deficit allfail排序完全没修；不能拿此门控掩盖该审计。'],
  sourceURLs=urls('CPO','RUN_THEN_WALK','TRICKS'),attribution_zh='本工程自定义的可行覆盖门控，非上述论文原算法；安全理论保证不迁移。'),
 dict(id='FLOW_ENDPOINT_RATIO',priority=2,title_zh='解决更新目标错配：逐MC CFM ratio的普通PPO',status='CPU_LOSS_PROTOTYPE_ONLY_FORMAL_SCALE_UNREGISTERED',
  problem_zh='同theta同roots的TRAIN诊断显示末步SDE扰动显著损伤候选，而去末噪声也会丢掉少数逃逸成功；直接W2的LastDet真实学习已负向。应区分探索成功的发现和把成功轨迹学到部署分布。',
  evidence_ids=['FIXED_THETA_ODE_SDE_ZERO_PROBE.json','FIXED_THETA_NOISE003_ZERO_PROBE.json','FINAL_SDE_STEP_COUNTERFACTUAL_DIAGNOSIS_20261031.json','ZERO_RESCUE_NEXT_DECISIONS_20261031.json','FPO_ENDPOINT_RATIO_CPU_PROTOCOL_20261031.json'],
  mechanism_zh='以clean端点的CFM误差差替代当前均维transition概率proxy，学习轨迹邻域的flow场；逐MC先ratio再clip避免先平均不同MC导致信号抵消。它使后续探索采样器可独立研究，但首版不打包换采样器。',
  hypothesis_zh='在保留相同W3候选与SafetyA的条件下，端点场更新可能比最后几步transitionPG更有效传递安全高EP模式到ODE部署。',
  single_core_change='replace PG estimator only: rho_im=exp(ell_behavior_old_im-ell_current_im); ordinaryPPOclip perMC; noASPO',
  math=dict(state='x_sigma=(1-sigma)*clean_endpoint + sigma*epsilon',target='v*=epsilon-clean_endpoint',
   World_cleanx_to_velocity='v_theta=(x_sigma-x0_theta)/sigma, sigma>0',
   energy='sum_8poses(mean_xyz((v_theta-v*)^2)); explicitly8xmean_all24; notoriginalILcleanxL1',
   old='samecollection CURRENT behavior old,notpermanentIL70reference',reuse='sigma/noise/oldenergy/cleanendpoint fixed4inners,detached; freshcurrentMemory perinner'),
  design=dict(common=common,arm='freshIL70 FPO-PPO versus SafetyEP; first matchedW3rollout/data/SafetyA/anchorwindow',
   not_changed=['W3eta.1','beta50physicalanchor','referenceIL70','uniformfull','LR','fourinner','deploymentODE'],
   later_sampler_test='Onlyif estimator works: separately testODEcollection+FPO againstmatchedFPO-W3 fromsameIL; donotclaim firsttest alreadyfullysampleraligned'),
  blockers_zh=['正式MC数、sigma分布与近零sigma权重、clip/energy尺度需要明确；不能把sourceclip.005或LLMclip.2不校量纲直接当最优。','CPU原型无World真实CFM能量/梯度分布证明；数值clamp/Huber若需加入必须显式注册，不能悄悄复制论文全部tricks。','PG应替换不叠加；原physicalanchor单独保留，不把ILreference当behaviorold。'],
  success_metrics=['samebudget officialODEPDMS/NC/DAC/TTC/EP','positivewitness learnedunderODE notonlySDEcandidate mean','CFMratio tails/perMCclipping/effectivegradient','highcohort/newzero','costperupdate/forwardbudget'],
  risks_zh=['CFM ratio是代理，不是精确endpointlikelihood/IS/KL。','allA0仍无PG方向；无成功支持时还需witness或可靠failcredit。','机器人微调支持普通PPO对照，不支持默认搬ASPO；多MC增加计算量且可能放大outlier。'],
  sourceURLs=urls('FPO','FPO_PLUS_PLUS'),existing_CPU_protocol=str(A/'FPO_ENDPOINT_RATIO_CPU_PROTOCOL_20261031.json')),
 dict(id='VERIFIED_SUCCESS_MEMORY',priority=3,title_zh='把成功变成可复用方向：当前run的verified-positive witness缓冲',status='DESIGN_ONLY_NOT_REGISTERED',
  problem_zh='部署zero与G8全fail/零PG不同；某scene有真实成功端点却在后续collection失败或回退，成功没有成为稳定知识。固定9low只覆盖已知小池，不能代表483低分或103k未知人口。',
  evidence_ids=['ZERO_RESCUE_NEXT_DECISIONS_20261031.json','ZERO_LOW_SCORE_LONGITUDINAL_DIAGNOSIS.json','TRAIN_PREFERENCE_ZERO_TIE_DIAGNOSTIC_20261031.json','SAFETY_EP_PROCESS_DIAGNOSIS_20261031.json'],
  mechanism_zh='保留当前run曾真实评分通过的同scene成功端点；当fresh组全fail或丢失已有更好可行机会时，用独立positiveCFM辅助将曾发现的成功转成可复现模式。历史teacher不进入A/logp/importance ratio。',
  hypothesis_zh='若主要是成功遗忘/轮换zero而非从未见过可行解，成功复用可减少新zero和relapse；无witness场景则不伪造目标。',
  single_core_change='add bounded verified-positive CFM auxiliary onuniformfull freshscenes; sameSafetyPG/anchor/sampler/LR; noforcedlow-scene resampling inthisfirstfactor',
  teacher=dict(start='empty current-run buffer; optionalalready-scoredIL70 initialization wouldbeaseparateexplicitchoice,notoldRL10k',
   accept='TRAINonly,samecanonicalcache/protocol,NC=DAC=TTC=Comfort1; keepactualsingleendpointbestEP/score,notaveragingmultiplemodes',
   refresh='upgrade onlywhenfreshrealverifiedscore>stored+1e-6; bindtoken/cache/behavior/normalizedendpoint/hash; freezechosenmaterials4inners',
   activation='allfailwithwitness OR freshbeststrictsafe scorelowerthanstored; allfreshpass+EP1skip; no-teacherskip',
   loss='squaredCFMvelocityerror onpositiveendpoint; notDPO/SIL/AWRexactalgorithm; coefficientandfinitebufferbudgetexplicitlyprespecify afterP0',
   failure_credit='retaincurrent mean(c)-c whenallfailcostvaries; noautomatictimingbonus; tie+nowitness remainsnoreliablePGdirection'),
  design=dict(common=common,arm='freshIL70 SafetyPG+successmemory versusfreshSafetyPGcontrol; sameuniformscene stream',
   not_changed=['noSafety10kwarmstart','noW2','no80/10/10sampler','noLRchange','nohistoryAdvantage','referenceIL70/beta50same']),
  success_metrics=['TRAINpredefinedpersistent/rotating/relapsezero repair','highcohortpreservation/newzero','same-scene witness retentiongap','auxenabled/skip/no-bank andcoverage','actualPG/anchor/auxgradientnorm+cos,notlossmagnitudeonly','officialODEPDMSsamebudget'],
  risks_zh=['成功端点只在记录协议下有效，不是humanoptimum或绝对安全。','当前run很少重采同scene时buffer覆盖和激活率可能很低；需事实触发，不为生成梯度反复全量扫。','aux过强会复制旧较低效率或单模态，必须最佳fresh升级/高分饱和skip，并实际测梯度注入。','旧Rescue同时RLwarmstart/freshAdam/W2/采样变化，不可当该fresh单变量的已完成证据。'],
  sourceURLs=urls('SIL','AWR','FPO')),
 dict(id='VISIBLE_CONDITION_GROUNDING',priority=4,title_zh='有条件推进：保住当前可见的场景条件表征',status='CPU_LABEL_AND_INFORMATION_AUDIT_BEFORE_ANY_PILOT',
  problem_zh='actor只有三路当前图像、vx/vy/ax/ay和粗导航；没有结构化红灯、车道/route几何、邻车速度或历史。Memory实际训练，但RL不再调用IL的object/future/post辅助任务，可能发生语义漂移，尚未被证明。',
  evidence_ids=['SCENARIO_SAFETY_EFFICIENCY_AUDIT_20261030.json','REPRESENTATION_TRAINABILITY_AUDIT_20261030.json','SAFETY_EP_PROCESS_DIAGNOSIS_20261031.json'],
  mechanism_zh='先确认危险/必要等待类别能否从现有图像+ego/nav分辨。若IL可分而Safety高分退化后不可分，可提出训练期小条件监督保持Memory语义，推理输入和部署参数闭包不变。',
  hypothesis_zh='若失败集中在本来可见的相关信号/队列且Memory辨识能力随RL下降，保留这部分语义可帮助同一policy在可见条件下平衡效率与安全。',
  single_core_change='conditionalfuturepilot: one train-only current-condition supervision loss onexistingMemory; noextraactororacleinput/noQwenunfreeze',
  first_diagnostic=dict(labels='relevantcurrentred joinedto currentrouteconnector or currentlead/queuegeometry; restrictobjects tot0;UNKNOWNnotgreen',
   exclusions=['futureGTvelocity/occupancy asactorinput','absentcachelighttokens interpretedgreen','coarsenavigation alone calledintersectiontruth','lowscore/EP0 alone calledunnecessaryparking'],
   validation='currentimageaudit andsmallcapacityreadout onpredefinedTRAINlogstrata; quantifylabelcoverage/conflicts andILvsSafetyreadoutquality; noGPUruninthisproposal'),
  trigger_zh='必须先有当前标签可信覆盖+可见性证据+IL到Safety语义下降证据；若单帧本身不可辨，训练监督不能创造观测，是否增加历史/传感信息应另做架构/IL协同研究，不作为同closure RL小trick。',
  design=dict(common=common,arm='futurefreshIL70 SafetyEP+oneconditionaux versusfreshSafetyEP; conditionheadtrainonlyandremovedatdeploy',
   scope_note='Any train-onlyreadoutparams andcompute mustbeexplicitlycounted; existing52.7mMemory/actionremainpolicyclosure; noILknowledgepipeline restart'),
  success_metrics=['currentrelevant-conditionlabel calibration/discrimination','necessarywaitvsverifiedsafeprogress opportunity strata','samebudgetNC/DAC/TTC+EP/newzero/highretention','PG/anchor/auxgradientcos','unchangeddeploymentinputs/noprivilegedleak'],
  risks_zh=['没有真实on-route signal/map输入时，privileged标签与可见观测可能不一致。','frozenQwen没保留的小目标线索不能靠Memory一定找回；当前更新到Memory不是表征充分证明。','aux只是训练监督，不是oraclecritic或已学会场景风险推理；不借IL方向名义重新训整套感知。'],
  sourceURLs=urls('ASYMMETRIC','LBC')),
]
# Latest process determines priorities; never promote a trick from change-count alone.
credit_gate=D[0]
credit_gate.update(priority='LOW_COST_CONDITIONAL_ONLY',status='CPU_COUNTERFACTUAL_NOT_PRIORITY_GIVEN_EP_DECLINE')
credit_gate['trigger_zh']='仅当未来同批梯度与条件分层证明弱可行组EP竞争是主要安全损失时再考虑；当前整体EP已回退，不能默认进一步压EP。'
retention=D[2];retention.update(priority=1,title_zh='第一方向：保留当前run已发现的安全进度与高质量模式')
retention['problem_zh']='最新Safety在修复一些旧zero同时制造新zero，高分队列回退；整体TTC改善与高队列TTC/EP退化并存。主要目标是让已获得的安全高进度不被后续更新遗忘，不是继续全面压EP。'
retention['single_core_change']='one bounded positive-CFM success-replay objective from CURRENT freshILrun only; unchangedfreshSafetyPG/anchor/sampler/LR'
retention['mechanism_zh']='从本run已真实评分的TRAIN成功轨迹中保存安全/进度见证，用独立小aux replay minibatch重建该scene当前Memory并学习端点；fresh onpolicy采样/A保持原样。覆盖高质量保持与真实zero见证，不把低分一律等同无解。'
retention['uniform_sampling_activation_blocker']=dict(source='starVLA/rl/qwen_world_research_data/data.py:SceneStream._uniform_order/at',
 population=103288,fresh_scene_exposures_per_optimizer_update=2,first_epoch_optimizer_updates=51644,
 conclusion_zh='full无放回epoch；50k更新仅100000次freshscene曝光，仍未遍历103288。若只在freshtoken碰巧命中此前成功bank才aux，整个初始预算基本不会激活。1280个诊断token也没有重复。',
 required_implementation_zh='必须显式从bounded成功memory取历史scene作独立aux样本，或另注册有覆盖的采样改变；本首版选择前者，不假称同tokenfreshlookup已实现保持。')
retention['teacher']['activation']='draw bounded separate historicalscene auxsample; all-freshsuccess-saturation canlimitauxbudget; perstoredscene skipobsolete/invalid and keepbestverified; no-history/no-witnessskip'
retention['teacher']['condition']='store originalscene/token/cache/frozen-Qwen-feature descriptor; rebuild CURRENT scene/predictiveMemory eachinner; neverreusegraph oroldMemory ascurrent'
retention['teacher']['refresh']='upgrade fromnewrealcanonicalTRAINrollouts orprespecifiedTRAINpanel only; ifcurrentverifiedendpointbetterthenreplace; noNAVTEST-basedteacher choice; samechosenmaterials4inner'
retention['teacher']['known_limit']='beststored isbestOBSERVED,notguaranteedtobebetterthanthe unscoredcurrentpolicy; avoidclaimingautomaticnonregression'
retention['success_metrics'] += ['auxactualactivation anddistinctscene exposure/compute,soanunactivatedbufferisnotdeclaredineffective','nohiddenreplayintoGRPO/nochangefreshstreamRNG']
retention['risks_zh'] += ['独立历史scene需要额外condition/CFM计算；其数据曝光和算力必须记录，不能装成零成本或on-policy样本。','当前run只见到很少zero场景或没有安全见证时buffer不能凭空修复；先看实际coverage与激活。']
retention['alternative_not_combined']=dict(name='fixed-step CURRENT-run reference refresh onlyifP0 supports',
 status='CONDITIONAL_SEPARATE_SINGLE_FACTOR_NOT_REGISTERED',
 hypothesis_zh='若beta50 IL70物理anchor与新成功PG持续反向且幅度重要，可检验固定时点保留本run已学进度的reference。当前仅是假设，不叫reference错误。',
 single_core_change='replaceanchorreferenceonly atonepredeclaredstep withsamefresh-run completecurrentLoRA+scene/predictiveMemory; noaux/LR/reward/samplerchange',
 invariants='freshIL70update0 origin; currentAdam/RNGcontinuity unchanged; behaviorold forPPO remainsseparate; referencecache synchronized andversioned; bet50physicalpenaltynotLMKL',
 protocol='choosefixedstepbeforelaunch usingTRAINrule/approvedbudget,notTestpeak; sealcomplete52.7mclosure+hash/checkpoint; nooldSafety10kexternalreference',
 sources=urls('RUN_THEN_WALK'))
retention['sourceURLs']=list(dict.fromkeys(retention['sourceURLs']+urls('RUN_THEN_WALK')))
fail=dict(id='FAILURE_CREDIT_WHEN_IDENTIFIABLE',priority=3,title_zh='条件方向：少量全失败组的可辨识信用与见证缺口',status='LOW_FREQUENCY_CPU_AUDIT_FIRST_NOT_REGISTERED',
 problem_zh='真正失败且costtie只占最新所查1280组的1组；另外7allfail组已有leastcost方向。不能把110个成功EP1零PG当作失败，也不能把不存在的NC.5正A当主因。',
 mechanism_zh='先逐candidate核NC/DAC/TTC向量与几何/违规时刻，把可比较的违例改善与不可比较的风险权衡分开。只有出现可靠的新排序信息才改变全fail credit；已有costvar保留原fallback。',
 hypothesis_zh='若被覆盖的persistent失败确实被scalar门控/等cost抹去方向，可靠细粒度credit可减少该类zero；如果根本没有见证或现有指标完全同败，则需要成功经验/条件信息而非伪造adv。',
 single_core_change='futureonly: changeallfailcreditforonevalidatedevent/metric-Pareto distinction; mixed/allfeasibleAsafe+AEP unchanged; noextraexplorationorLRchange',
 trigger_zh='必须先确认足够TRAIN曝光与事件/metric可比较性；目前该类仅1/1280，优先级低于高队列保持。crossed安全向量不强排，NC.5/NC0正A仅可解释相对leastcost、不能称安全。',
 evidence_ids=['SAFETY_EP_PROCESS_DIAGNOSIS_20261031.json','TRAIN_PREFERENCE_ZERO_TIE_DIAGNOSTIC_20261031.json'],
 design=dict(common=common,arm='onlyafterTRAINevidence: freshIL70 SafetyEP oneallfailcredit factorversuscontrol',
  tie_rule='ifnowitness/noadditionalvalidinformation retainzeroPG+anchor,notrandomdirection',
  no_scope_expansion='noQcritic/noGTfutureinput/nodefaultnewGPUresampling/noGLMtokentricks'),
 risks_zh=['平均deficit不是碰撞severity；自定义NC优先级也是目标选择，非经验证物理风险。','奖励更晚违规可能学会延迟撞击或无故停车，需事件和进度机会共同核对。','低频机制即使修复全部样例也可能几乎不提高全局分数；不为填资源注册。'],
 success_metrics=['actualactivatedgroupfraction','candidate-pair ordering validatedagainstsourcevectors','fixedTRAINpersistentzero repair notsaturatedsuccessupdates','newzero/highcohort safety/EP','samebudget officialPDMS'],sourceURLs=urls('CPO','RUN_THEN_WALK','SIL'))
D=[retention,D[1],fail,D[3]]
latest=None;at22=None;highlatest=None
if process is not None:
 latest=max(process['evaluation_points'],key=lambda x:x['updates'])
 attr=max(process['component_attribution'],key=lambda x:x['updates'])
 at22=next((x for x in process['component_attribution'] if x['updates']==22000),None)
 highlatest=next(x for x in process['high10k_retention'] if x['updates']==latest['updates'])
 latest_readout=dict(latest_updates=latest['updates'],latest_PDMS=latest['PDMS'],baseline10k_PDMS=next(x['PDMS'] for x in process['evaluation_points'] if x['updates']==10000),
  latest_vs10k=attr,latest_high10k_retention=highlatest,secondary22k=at22,
  zero_longitudinal=process['zero_longitudinal'],rollout_aggregate=process['rollout_aggregate'],rollout_sampling_limits=process['rollout_sampling_limits'],
  interpretation_zh='总体修复部分旧zero/TTC提升，与固定高队列各项回退可并存；高cohort为条件选择指标不是无偏总体估计。Shapley仅账面分解不是梯度因果。')
else:latest_readout=dict(status='WAITING_LATEST_PROCESS')
if process is not None:
 windows=process['rollout_windows']
 P0['recorded_scalar_guards']=dict(windows=len(windows),rank0_optimizer_updates=sum(w['journal']['updates'] for w in windows),preclip_above1_updates=int(sum(w['journal']['updates']*w['journal']['preclip_above1_fraction'] for w in windows)),Aclip5_candidates=process['rollout_aggregate']['normalized_A_clip5_candidates'],candidates=process['rollout_aggregate']['candidates'],interpretation_zh='所查窗口没有总gradnorm>1触发；Aclip5也仅2/10240，不优先修未见的爆炸。总norm不回答PG/anchor夹角或跨rank抵消。')
proposal=dict(schema='world_SafetyEP_problem_driven_freshIL_research_directions_v1',
 status='COMPLETE_PROPOSAL_NOT_REGISTERED' if process is not None and process.get('status','').startswith('COMPLETE') else 'DRAFT_WAITING_LATEST_SAFETY_PROCESS',
 created_at=datetime.now(timezone.utc).isoformat(),
 focus=dict(current_allowed_RL='PPO_SAFETY_EP only; root ownscooperative停止/撤销其它eval',other_routes='stopped/stoprequestedbyuserfocuschange,notalgorithmfailure',new_pilots_started=False,new_registrations=False,new_GPU_work=0,
  all_future_start='sameIL70,freshAdam/RNG42/update0; noRL10kfork',historical_LR_study='nativeforkresultsnotyetavailable; userstoppedfocusnotfutility; preupdatehistoryguardfailuresremainoperational'),
 main_points_zh=[
  '最新24k相对10k整体TTC改善而EP明确下降；固定高8850队列NC/DAC/TTC/EP均负，不能概括为过度追快。优先保护已学安全进度和高分，再修可靠的少量失败信用。',
  'P0先测真实PG/物理anchor梯度与全局8rank信号，不以技巧数量、loss标量或局部zeroPG决定主因。',
  'SafetyEP已经实现严格within-feasible EP。有可行候选的组，unsafe不可能因EP获得正A；allfail正A是leastcost方向，需逐NC/DAC/TTC向量审计，不是安全证书。',
  '新10个窗口1280组中111局部zeroPG=110成功EP1饱和+1allfailcosttie；旧10k125个完整collection无一全rank A0。两种窗口口径不混，均不证明optimizer因zeroPG停摆。',
  '候选质量不等于ODE学习收益；LastDet1k实测负结果阻止把“去末噪声”直接当改进证据。FPO方向应先隔离PG估计器，后谈采样器。',
  '成功witness/teacher刷新、参数步长LR、reference约束分别改变不同机制；暂不组合、暂不从RL10k分叉，所有未来对照从同IL70开始。',
 ],
 baseline=common,P0_gradient_and_credit_diagnosis=P0,
 evidence=dict(latest_Safety_process=dict(path=str(cp),status=process.get('status') if process else 'PENDING',file_sha256=sha(cp) if process else None,summary_reference='priority_basis_latest_process; completeoriginalprocesslinkedbySHA,notduplicatedhere'),
  historical10k_cached_credit=credit,
  historical_zero_observation_scope=dict(Safety_updates=[0,1000,5000,10000],TRAIN1024=old['coverage']['per_TRAIN1024_route']['PPO_SAFETY_EP'],warning='historical4observationsonly; latestPROCESSsupersedes persistentcounts'),
  fixedtheta_selected15TRAIN=old['observed_mechanisms']['fixed_theta_selected_TRAIN'],
  finalnoise_caution_zh=old['observed_mechanisms']['final_noise_interpretation_zh'],
  lastdet_actual1k=old['observed_mechanisms']['lastdet_complete_1k'],
  actor_information=obs['actor_current_inputs'],available_fields=obs['available_fields_and_limits'],canonical_semantics=obs['reward_readout_semantics'],
  trainability=old['observed_mechanisms']['trainability']),
 directions=D,priority_basis_latest_process=latest_readout,
 deferred_single_factor_controls=[credit_gate,
  dict(id='FRESH_LR_CONTROL',status='DEFERRED_NOT_REGISTERED',problem_zh='当前100opt warmup后constant确实未衰减；这不是已证退化根因。',
   design='IfP0 showsoverlarge lateupdates: freshIL70constant baseline versusoneexplicitgroup-relative schedule withsame100optwarmup,source50kidentityandregistereddecayhorizon; noRL10kfork/Adamresetconfound/nootherchange',
   cautions=['LRchangesphysicalPGandanchorupdatestogether,notteacherrefresh','action/memory10:1ratioretained; vendorabsoluteeta_minwouldalterratio','paperRun-then-Walkchangesreward/ref/LR,notcausalLRproof','adaptiveflowchangeproxyisnottrueKL'],
   sourceURLs=['https://arxiv.org/html/2609.25831v2#A3','https://arxiv.org/html/2602.02481v2#A3.SS3'],primary_inventory=str(A/'RL_LR_PRIMARY_RESEARCH_20261031.json')),
  dict(id='FAIL_CREDIT_PRIORITY_AUDIT',status='CPU_RECORD_REVIEW_FIRST',fact='old10k7allfailgroups contain28positiveunsafe candidates,5NC0; exactNCvsDAC/TTCtradeoffrequiresoriginalcanonicalvectors',
   caution='mean deficit isnotcrashseverity. costvar alreadyhasdirection; extraeventtiming onlyaddedonecosttie distinction intheolderADAPT170allzeroexample,notSafety-specificproof.',
   potential_followup='OnlyiflatestSafetyfailureconcentration supportsit: oneexplicitallfail component-priority/verifiedPareto-creditchange,keepingwithin-feasibleEP andallotherrecipe; no blanketunsafeA ban',sourceURLs=urls('CPO','RUN_THEN_WALK')),
  dict(id='REFERENCE_OR_BETA',status='DEFER_UNTIL_P0_GRADIENT_EVIDENCE',caution='IL70physicalmeananchor isnotLMtokenKL. Scalar kl_loss cannotshow wrongreference/pullback; do notrefreshreference orremoveanchorwithoutmatchedgradientandhighretentionevidence.'),
 ],
 LLM_to_flow_transfer=dict(methods_transfer=['controlledsinglefactorablation','groupcentering/globalnormalizationdiagnostics','positive/negativeeffectiveclipmaskanalysis','recorddata/model/sampleridentities'],
  not_automatic=['LLMtokenlosslevel/lengthpenalty','tokenclipnumerics copiedto24Ddimensionmeantransitionproxy','sequenceIS/entropy interpretations appliedtoW3transition averages','removeKLrecommendation appliedtophysicalILanchor','dynamicresamplingtoavoidall0 appliedtosuccessEP1saturation'],
  current_implementation='source likelihood isclamped dimension-mean transitionproxy,ratio[8,W3],notexactendpoint/fullchainIS; global64A normalization present; beta50isphysicalclean-x anchor',sourceURLs=urls('TRICKS','FPO_PLUS_PLUS')),
 sources=S,input_evidence={str(A/n):dict(file_sha256=sha(A/n),status=v.get('status')) for n,v in E.items()},
 sourceURLs=list(dict.fromkeys(u for s in S for u in s['sourceURLs'])),
 limitations_zh=['文献与CPU反事实产生假设，不构成新算法收益证明。','TRAIN反馈面板非heldout；固定15场景探针是条件选中小cohort，不能估103k人口收益。','paired44logCI只反映固定训练/eval seed下场景log不确定性，不是多seed训练方差；不追加种子。','不从NAVTEST失败名单生成训练bank；未来设计只用TRAIN当前材料。','所有新pilot须root整合后安排，本文件不执行训练/评测/停止。'],
 builder=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)))
proposal['sha256']=fp(proposal)
p=OUT.with_name(OUT.name+'.tmp.'+str(os.getpid()));p.write_text(json.dumps(proposal,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(p,OUT)
print(json.dumps(dict(output=str(OUT),status=proposal['status'],directions=len(D),sources=len(S),CPU_only=True),ensure_ascii=False))
