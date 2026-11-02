"""Own-current-run sampled witness -> actual B1 nominal improvement, not extra PG.

The auxiliary replaces a fixed share of the beta50 IL budget. Gradient-derived
scales are detached, rank-local, checkpointed, and frozen after100 effective
teacher-slot loss calls. No critic or external/cross-algorithm teacher is used.
"""
import copy,math
import torch
import torch.nn.functional as F
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.rl.simwam_recipe import checkpoint
from starVLA.rl.simwam_recipe.contracts import seed_for
from starVLA.rl.qwen_world_safety_exploration.retention_material import digest,sealed,verify_seal,validate_feedback
from .ode_target import probe_nominal,validate_nominal,strict_high,condition_for_batch,predict_clean_x
from .adaptive_group import collect_extra_if_needed,validate_extra

SCHEMA='world_performance_own_run_nominal_improvement_v1'
CALIBRATION_SCHEMA='world_performance_rank_local_aux_gradient_calibration_v1'
CONFIG=dict(schema=SCHEMA,start_epoch=.5,alpha=.05,min_teacher_score=.9,gain_margin=.02,
    cfm_mc=2,sigma_min=.1,sigma_max=.9,cfm_weight=.75,ode_weight=.25,seed=42)
QUALITY_SCHEMA='world_verified_quality_nominal_improvement_v2'
QUALITY_CONFIG=dict(CONFIG,schema=QUALITY_SCHEMA,start_epoch=0.,alpha=.05,
    strict_high_nominal_alpha=.02,allow_strict_high_nominal_teacher=True,
    repair_gain_margin=.02,safe_high_gain_margin=.005)
QUALITY_CONFIG.pop('gain_margin')
CALIBRATION_PROTOCOL=dict(schema=CALIBRATION_SCHEMA,target='norm of current same-microbatch unscaled physical ILmean-MSE gradient',
    ratio='norm(gIL)/max(norm(gcomponent),1e-12)',minimum_scale=1e-4,maximum_scale=1e4,
    EMA_decay=.9,calibration_calls=100,measurement_calls=500,clock='effective per-rank teacher-slot loss calls, including inner reuse; not Adam updates',
    gradients='all current trainable parameters; local norm, no conditional cross-rank collective; no .grad mutation',
    freeze='after100 calls keep last EMA; only diagnose until500, then stop gradient probes; last_measured labels remain explicit')


def validate_config(config):
    if config!=CONFIG and config!=QUALITY_CONFIG:raise ValueError('Explicit fixed nominal-improvement config required')
    return config


def _draws(rollout,rank,config=None):
    config=CONFIG if config is None else validate_config(config)
    v=rollout['behavior_version'];seed=seed_for(config['schema'],42,v['run'],v['update'],v.get('adapter_sha256'),rollout['token'],rank)
    g=torch.Generator().manual_seed(seed)
    return seed,.1+.8*torch.rand(2,generator=g,dtype=torch.float32),torch.randn((2,8,3),generator=g,dtype=torch.float32)


def _select_teacher(rollout,nominal,extra,config=None):
    config=CONFIG if config is None else validate_config(config)
    if strict_high(nominal['feedback']) and not config.get('allow_strict_high_nominal_teacher',False):return None,None,'nominal_already_strict_high'
    candidates=[]
    for origin,pool in [('primary_G16',rollout),('extra_G16',extra)]:
        if pool is None:continue
        for i,record in enumerate(pool['records']):
            if strict_high(record):candidates.append((float(record['metrics']['score']),origin,i,pool,record))
    if not candidates:return None,None,'no_strict_high_teacher'
    _,origin,i,pool,record=max(candidates,key=lambda x:x[0])
    margin=(config['safe_high_gain_margin'] if strict_high(nominal['feedback']) else config['repair_gain_margin']) if config==QUALITY_CONFIG else config['gain_margin']
    if float(record['metrics']['score'])<float(nominal['feedback']['metrics']['score'])+margin:
        return None,None,'teacher_gain_below_margin'
    clean=pool['chain'][i,-1].detach().float().cpu().clone()
    requested=torch.as_tensor(record['requested_trajectory']).float().cpu()
    validate_feedback(record,rollout['token'],rollout['log'],requested,rollout['protocol_sha256'])
    decoded=ActionCodec.denorm_odo(clean)
    bound=8*torch.finfo(torch.float32).eps*(decoded.abs()+decoded.new_tensor([66.74,42.,3.53]))
    if clean.shape!=(8,3)or not torch.isfinite(clean).all()or bool(((decoded-requested).abs()>bound).any()):
        raise ValueError('Teacher endpoint does not correspond to actually scored FP32 action')
    target=dict(origin=origin,index=i,token=rollout['token'],log=rollout['log'],source_version=copy.deepcopy(rollout['behavior_version']),
        feedback=checkpoint.cpu_tree(record),clean_sha256=digest(clean),requested_sha256=digest(requested),
        nominal_score=float(nominal['feedback']['metrics']['score']),score=float(record['metrics']['score']),
        gain=float(record['metrics']['score'])-float(nominal['feedback']['metrics']['score']),PG_eligible=False)
    return clean,target,None


def active_alpha(material,config=None):
    config=CONFIG if config is None else validate_config(config)
    if not material['enabled']:return 0.
    if config==QUALITY_CONFIG and strict_high(material['nominal']['feedback']):return config['strict_high_nominal_alpha']
    return config['alpha']


@torch.no_grad()
def attach_material(actor,rollout,conditions,service,*,exposure_epoch,rank,config=None,calibration_state=None,nominal=None,extra=None,teacher_selector=None):
    config=CONFIG if config is None else validate_config(config)
    if 'nominal_improvement'in rollout:raise ValueError('Nominal targets attached once per fresh behavior version')
    if rollout.get('group_size')!=16:raise ValueError('Primary group is G16, extraG16 is teacher-only')
    epoch=float(exposure_epoch)
    if not math.isfinite(epoch)or epoch<0:raise ValueError('Actual TRAIN exposure epoch required')
    base=dict(schema=config['schema'],protocol=copy.deepcopy(config),token=rollout['token'],log=rollout['log'],rank=rank,
        behavior_version=copy.deepcopy(rollout['behavior_version']),condition_identity=rollout['cond_bg']['identity'],
        exposure_epoch=epoch,enabled=False,skip_reason='before_rescue_start',nominal=None,extra=None,
        actual_B1_queries=0,actual_extra_teacher_queries=0,teacher=None,clean=None,sigma=None,noise=None,selection_seed=None)
    if epoch>=config['start_epoch'] and rollout.get('valid_scene',True):
        audit=actor._probability_audit
        try:
            nominal=probe_nominal(actor,conditions,rollout['token'],rollout['log'],service,rollout['behavior_version'])if nominal is None else nominal
            validate_nominal(nominal,token=rollout['token'],log=rollout['log'],version=rollout['behavior_version'],protocol_sha256=rollout['protocol_sha256'],condition_identity=rollout['cond_bg']['identity'])
            extra=collect_extra_if_needed(actor,rollout,nominal,service,rank=rank)if extra is None else extra
            validate_extra(extra,rollout,nominal,rank=rank)
            clean,teacher,skip=(teacher_selector or _select_teacher)(rollout,nominal,extra,config)
            base.update(nominal=nominal,extra=extra,skip_reason=skip,enabled=skip is None,
                actual_B1_queries=1,actual_extra_teacher_queries=0 if extra is None else 16,teacher=teacher,clean=clean)
            if skip is None:
                seed,sigma,noise=_draws(rollout,rank,config);base.update(selection_seed=seed,sigma=sigma,noise=noise)
        finally:actor._probability_audit=audit
    elif not rollout.get('valid_scene',True):base['skip_reason']='invalid_padding_scene'
    elif nominal is not None or extra is not None:raise ValueError('No hidden auxiliary scoring before exposure-start')
    if calibration_state is not None:validate_calibration_state(calibration_state)
    if config==QUALITY_CONFIG:base['active_alpha']=active_alpha(base,config)
    rollout['nominal_improvement']=sealed(base,'material_sha256')
    return rollout


def validate_material(rollout,*,rank,expected_epoch=None,config=None,teacher_selector=None):
    config=CONFIG if config is None else validate_config(config);p=rollout['nominal_improvement'];verify_seal(p,'material_sha256')
    if(p.get('schema')!=config['schema'] or p.get('protocol')!=config or p['rank']!=rank or p['token']!=rollout['token']or p['log']!=rollout['log']
       or p['behavior_version']!=rollout['behavior_version']or p['condition_identity']!=rollout['cond_bg']['identity']
       or(expected_epoch is not None and p['exposure_epoch']!=expected_epoch)):
        raise ValueError('Nominal auxiliary current-run/scene/config identity changed')
    if config==QUALITY_CONFIG and p.get('active_alpha')!=active_alpha(p,config):raise ValueError('Verified-quality conditional alpha changed')
    active=p['exposure_epoch']>=config['start_epoch']and rollout.get('valid_scene',True)
    if not active:
        skip='before_rescue_start'if rollout.get('valid_scene',True)else'invalid_padding_scene'
        if p['skip_reason']!=skip or p['enabled']or any(p[k]is not None for k in ('nominal','extra','teacher','clean','sigma','noise','selection_seed'))or p['actual_B1_queries']or p['actual_extra_teacher_queries']:
            raise ValueError('Skipped auxiliary must retain no teacher/query material')
        return p
    validate_nominal(p['nominal'],token=rollout['token'],log=rollout['log'],version=rollout['behavior_version'],protocol_sha256=rollout['protocol_sha256'],condition_identity=rollout['cond_bg']['identity'])
    if p['nominal']['condition']!=rollout['cond_bg']:raise ValueError('B1 did not use this exact legal current observation descriptor')
    validate_extra(p['extra'],rollout,p['nominal'],rank=rank)
    clean,teacher,skip=(teacher_selector or _select_teacher)(rollout,p['nominal'],p['extra'],config)
    if digest(p['teacher'])!=digest(teacher)or p['skip_reason']!=skip or p['enabled']!=(skip is None)or p['actual_B1_queries']!=1 or p['actual_extra_teacher_queries']!=(0 if p['extra']is None else 16):
        raise ValueError('Teacher selection/gain/gate/query accounting changed')
    if skip is not None:
        if any(p[k]is not None for k in ('clean','sigma','noise','selection_seed')):raise ValueError('Inactive teacher retained loss material')
    else:
        seed,sigma,noise=_draws(rollout,rank,config)
        if p['selection_seed']!=seed:raise ValueError('Auxiliary privateRNG seed changed')
        for k,want in [('clean',clean),('sigma',sigma),('noise',noise)]:
            x=p[k]
            if x.requires_grad or x.dtype!=torch.float32 or x.shape!=want.shape or not torch.equal(x.cpu(),want):raise ValueError('Detached auxiliary '+k+' changed')
    return p


def compute_nominal_losses(model,rollout,config=None):
    """Raw CFM velocityMSE and normalized-action B1 last-head SmoothL1."""
    config=CONFIG if config is None else validate_config(config)
    p=rollout['nominal_improvement'];zero=next(model.parameters()).detach().new_zeros(())
    if p.get('schema')!=config['schema'] or p.get('protocol')!=config:raise ValueError('Nominal component protocol changed')
    alpha=active_alpha(p,config)
    if config==QUALITY_CONFIG and p.get('active_alpha')!=alpha:raise ValueError('Verified-quality saved alpha changed')
    metrics=dict(nominal_aux_enabled=zero+int(p['enabled']),nominal_aux_alpha=zero+alpha,
        nominal_B1_queries=zero+p['actual_B1_queries'],nominal_extra_teacher_queries=zero+p['actual_extra_teacher_queries'],
        nominal_aux_raw_cfm=zero,nominal_aux_raw_ode=zero,nominal_aux_teacher_gain=zero)
    if not p['enabled']:return None,None,metrics
    clean=p['clean'].detach().to(model.device);sigma=p['sigma'].detach().to(model.device);noise=p['noise'].detach().to(model.device)
    teacher=clean.unsqueeze(0).expand(2,-1,-1);state=(1-sigma[:,None,None])*teacher+sigma[:,None,None]*noise
    audit=model._probability_audit
    try:
        raw=condition_for_batch(model,rollout['cond_bg'],2)
        with model._context_call(raw)as condition:velocity=model.action_velocity(state,sigma*1000,condition)
        cfm=(velocity.float()-(noise-teacher)).square().mean()
        nominal=p['nominal'];prediction=predict_clean_x(model,rollout['cond_bg'],nominal['last_state'].to(model.device),nominal['last_sigma'].to(model.device))
        ode=F.smooth_l1_loss(prediction.float(),teacher[:1].detach().float())
    finally:model._probability_audit=audit
    if not torch.isfinite(cfm)or not torch.isfinite(ode):raise FloatingPointError('Nonfinite nominal improvement component')
    metrics.update(nominal_aux_raw_cfm=cfm.detach(),nominal_aux_raw_ode=ode.detach(),nominal_aux_teacher_gain=zero+p['teacher']['gain'])
    return cfm,ode,metrics


def new_calibration_state():
    return dict(schema=CALIBRATION_SCHEMA,protocol=copy.deepcopy(CALIBRATION_PROTOCOL),effective_updates=0,calibration_updates=0,
        measured_updates=0,frozen=False,scale_cfm=None,scale_ode=None,last_norms=None,scale_clip_hits_cfm=0,scale_clip_hits_ode=0)


def validate_calibration_state(state):
    if state.get('schema')!=CALIBRATION_SCHEMA or state.get('protocol')!=CALIBRATION_PROTOCOL:raise ValueError('Auxiliary calibration protocol changed')
    n=state['effective_updates']
    if type(n)is not int or n<0 or state['calibration_updates']!=min(100,n)or state['frozen']!=(n>=100)or state['measured_updates']!=min(500,n):
        raise ValueError('Auxiliary calibration effective-slot counters changed')
    for k in ('scale_cfm','scale_ode'):
        v=state[k]
        if (v is None)!=(n==0)or(v is not None and(not math.isfinite(v)or not 1e-4<=v<=1e4)):raise ValueError('Invalid frozen/EMA auxiliary scale')
    if n and(state['last_norms']is None or any(not math.isfinite(v)or v<0 for v in state['last_norms'].values())):raise ValueError('Invalid measured component gradient norm')
    return state


def _gradient_norm(loss,params):
    if not loss.requires_grad:return 0.
    gradients=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    norms=[]
    for g in gradients:
        if g is None:continue
        value=g.detach().float()
        norms.append(value.norm().double())
    if not norms:return 0.
    norm=torch.stack(norms).norm()
    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite auxiliary component gradient')
    return float(norm)


def calibrate_components(model,pg,IL,cfm,ode,state):
    validate_calibration_state(state);params=[p for p in model.parameters()if p.requires_grad]
    if not params:raise ValueError('No actual trainable scope for gradient calibration')
    measured=state['effective_updates']<500
    if measured:
        norms={k:_gradient_norm(v,params)for k,v in [('pg',pg),('IL',IL),('cfm',cfm),('ode',ode)]};state['last_norms']=norms;state['measured_updates']+=1
        if not state['frozen']:
            for name in ('cfm','ode'):
                raw=norms['IL']/max(norms[name],1e-12);clipped=max(1e-4,min(1e4,raw));key='scale_'+name
                state['scale_clip_hits_'+name]+=int(raw!=clipped)
                state[key]=clipped if state[key]is None else max(1e-4,min(1e4,.9*state[key]+.1*clipped))
            state['calibration_updates']+=1
    state['effective_updates']+=1;state['frozen']=state['effective_updates']>=100
    validate_calibration_state(state)
    out=dict(cfm=cfm*state['scale_cfm'],ode=ode*state['scale_ode']);out['rescue']=.75*out['cfm']+.25*out['ode']
    z=cfm.detach().new_zeros(());metrics={f'nominal_gradient_norm_{k}_last_measured':z+v for k,v in state['last_norms'].items()}
    metrics.update(nominal_gradient_norm_measured=z+int(measured),nominal_calibration_effective_updates=z+state['effective_updates'],
        nominal_calibration_frozen=z+int(state['frozen']),nominal_scale_cfm=z+state['scale_cfm'],nominal_scale_ode=z+state['scale_ode'],
        nominal_scale_cfm_clip_hits=z+state['scale_clip_hits_cfm'],nominal_scale_ode_clip_hits=z+state['scale_clip_hits_ode'],
        nominal_normed_cfm=out['cfm'].detach(),nominal_normed_ode=out['ode'].detach())
    return out,metrics


class NominalLossMixin:
    """Root's ComponentCapture must provide actual source PG and betaIL terms."""
    def _policy_loss(self,model,rollout):
        total,metrics=super()._policy_loss(model,rollout)
        if 'nominal_improvement'not in rollout:return total,metrics
        config=getattr(self,'nominal_config',CONFIG)
        cfm,ode,values=compute_nominal_losses(model,rollout,config);metrics.update(values)
        if cfm is None:return total,metrics
        state=self.auxiliary_state;beta=float(model.grpo_kl_beta)
        if beta!=50.:raise ValueError('Nominal study replaces fixed beta50 budget')
        calibrate=getattr(self,'calibrate_nominal_components',calibrate_components)
        parts,values=calibrate(model,self.pg_component,self.weighted_anchor_component/beta,cfm,ode,state)
        alpha=active_alpha(rollout['nominal_improvement'],config)
        replacement=alpha*(beta*parts['rescue']-self.weighted_anchor_component)
        metrics.update(values);metrics.update(nominal_aux_budget_delta=replacement.detach(),
            nominal_aux_beta_rescue=(beta*parts['rescue']).detach(),nominal_aux_IL_share=total.new_tensor(1-alpha),
            nominal_aux_rescue_share=total.new_tensor(alpha))
        return total+replacement,metrics
