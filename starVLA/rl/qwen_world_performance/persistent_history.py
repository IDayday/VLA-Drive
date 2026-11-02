"""Proactive own-run B1 success retention; no extra PG or cross-run teachers."""
import copy,math
import torch
from starVLA.rl.simwam_recipe import checkpoint
from starVLA.rl.simwam_recipe.contracts import seed_for
from starVLA.rl.qwen_world_safety_exploration.retention_material import digest,sealed,verify_seal
from .ode_target import probe_nominal,validate_nominal,strict_high,condition_for_batch,predict_clean_x
from .nominal_improvement import validate_calibration_state
import torch.nn.functional as F

SCHEMA='world_performance_proactive_own_run_history_v1'
PROTOCOL=dict(schema=SCHEMA,capacity=512,seed=42,minimum_score=.9,strict='NC/DAC/TTC/Comfort all1',
    admission='own-run actualTRAIN B1 from phase2 nominal probes; runtime stage admits, never NAV or sampled teacher',
    target='prior policy-version actualB1 normalized clean endpoint; no oldpolicy density or PG replay',
    selection='one proactive private-RNG historical scene per rank/collection, only teacher.update<current.update',
    floor=.05,regression_extra=.20,regression_margin=.01,upgrade_margin=1e-6,
    dominated='if currentB1 strict-high andscore>teacher+1e-6: h0 thiswindow, upgradebank forlaterpolicyversion',
    loss='same normalized .75CFMvelocityMSE+.25B1-last-stateSmoothL1; currentMemory rebuilt; scales read-only fromnominal calibration, fallback1',
    CFM=dict(mc=2,sigma_min=.1,sigma_max=.9),PG_eligible=False)


def new_history_state(runsha,rank):
    return dict(schema=SCHEMA,protocol=copy.deepcopy(PROTOCOL),run=runsha,rank=rank,entries=[],
        rng_state=torch.Generator().manual_seed(seed_for(SCHEMA,42,runsha,rank)).get_state(),condition_identity=None,protocol_sha256=None,
        admissions=0,replacements=0,not_retained=0,upgrades=0,windows=0,selected_count=0,
        regression_count=0,dominated_teacher_count=0,active_count=0,history_B1_queries_local=0,
        hshare_sum=0.,forward_seconds_local=0.,scoring_seconds_local=0.)


def _check_witness(nominal,runsha,require_high=False):
    v=nominal['source_version']
    if v.get('run')!=runsha or type(v.get('update'))is not int or v['update']<0:
        raise ValueError('No external-run or invalid-version historical actor')
    validate_nominal(nominal,token=nominal['token'],log=nominal['log'],version=v,
        protocol_sha256=nominal['protocol_sha256'],condition_identity=nominal['condition']['identity'])
    if require_high and not strict_high(nominal['feedback']):raise ValueError('Only real strict-high B1 may become historical teacher')


def admit_nominal(bank,nominal):
    """Mutate/return the bank; runtime calls only phase2/3 fresh valid TRAIN probes."""
    if bank.get('schema')!=SCHEMA or bank.get('protocol')!=PROTOCOL:raise ValueError('Historybank protocol changed')
    _check_witness(nominal,bank['run'])
    if not strict_high(nominal['feedback']):return bank
    for key,value in [('condition_identity',nominal['condition']['identity']),('protocol_sha256',nominal['protocol_sha256'])]:
        if bank[key]is None:bank[key]=value
        elif bank[key]!=value:raise ValueError('Historical teacher observation/scorer binding changed')
    entries=list(bank['entries']);bank['entries']=entries
    index=next((i for i,e in enumerate(entries)if e['token']==nominal['token']),None)
    if index is not None:
        old=entries[index]
        if nominal['source_version']['update']<old['source_version']['update']:raise ValueError('Historical teacher cannot be replaced by an older policy')
        if float(nominal['feedback']['metrics']['score'])>float(old['feedback']['metrics']['score'])+1e-6:
            entries[index]=checkpoint.cpu_tree(nominal);bank['upgrades']+=1
        return bank
    bank['admissions']+=1
    if len(entries)<512:entries.append(checkpoint.cpu_tree(nominal))
    else:
        rng=torch.Generator();rng.set_state(bank['rng_state'].cpu());j=int(torch.randint(bank['admissions'],(1,),generator=rng));bank['rng_state']=rng.get_state()
        if j<512:entries[j]=checkpoint.cpu_tree(nominal);bank['replacements']+=1
        else:bank['not_retained']+=1
    return bank


def _pick(bank,version):
    ids=[i for i,e in enumerate(bank['entries'])if e['source_version']['update']<version['update']]
    rng=torch.Generator();rng.set_state(bank['rng_state'].cpu())
    return (ids[int(torch.randint(len(ids),(1,),generator=rng))]if ids else None),rng.get_state()


def _regression(teacher,current):
    return (any(current['feedback']['metrics'][k]!=1 for k in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','comfort'))
        or float(current['feedback']['metrics']['score'])<float(teacher['feedback']['metrics']['score'])-.01)


def _decision(teacher,current):
    if teacher is None:return False,False,0.
    dominated=strict_high(current['feedback'])and float(current['feedback']['metrics']['score'])>float(teacher['feedback']['metrics']['score'])+1e-6
    regression=_regression(teacher,current)
    return bool(regression),bool(dominated),0. if dominated else .05+.20*int(regression)


def _advance(before,current,version):
    bank={k:(list(v)if k=='entries'else v)for k,v in before.items()};index,state=_pick(bank,version);bank['rng_state']=state;bank['windows']+=1
    teacher=None if index is None else before['entries'][index]
    if (teacher is None)!=(current is None):raise ValueError('A selected historical scene needs exactly one currentB1 verification')
    regression,dominated,h=_decision(teacher,current)
    if teacher is not None:
        bank['selected_count']+=1;bank['regression_count']+=int(regression);bank['dominated_teacher_count']+=int(dominated)
        bank['active_count']+=int(h>0);bank['history_B1_queries_local']+=1;bank['hshare_sum']+=h
        for key in ('forward','scoring'):bank[key+'_seconds_local']+=float(current['seconds'][key])
        #Teacher material stays the selected older version; any improved current
        #B1 is admitted only to the bank for a later collection/policy version.
        admit_nominal(bank,current)
    return bank,index,teacher,regression,dominated,h


@torch.no_grad()
def attach_history_window(bank,actor,conditions,service,version):
    if version.get('run')!=bank['run']:raise ValueError('Cross-algorithm history retention is forbidden')
    before={k:(list(v)if k=='entries'else v)for k,v in bank.items()};index,_=_pick(before,version);teacher=None if index is None else before['entries'][index]
    current=None;audit=actor._probability_audit
    try:
        if teacher is not None:current=probe_nominal(actor,conditions,teacher['token'],teacher['log'],service,version)
    finally:actor._probability_audit=audit
    after,index,teacher,regression,dominated,h=_advance(before,current,version)
    sigma=noise=seed=None
    if h>0:
        seed=seed_for(SCHEMA,42,bank['run'],version['update'],version.get('adapter_sha256'),teacher['token'],bank['rank'])
        rng=torch.Generator().manual_seed(seed);sigma=.1+.8*torch.rand(2,generator=rng,dtype=torch.float32);noise=torch.randn((2,8,3),generator=rng,dtype=torch.float32)
    value=dict(schema=SCHEMA,protocol=copy.deepcopy(PROTOCOL),run=bank['run'],rank=bank['rank'],version=copy.deepcopy(version),window=after['windows'],
        selection_index=index,teacher=teacher,current=current,enabled=h>0,hshare=h,regression=regression,dominated_teacher=dominated,
        actual_B1_queries=int(current is not None),selection_seed=seed,sigma=sigma,noise=noise,
        bank_before=before,bank_after_sha256=digest(after),PG_eligible=False)
    bank.clear();bank.update(after)
    return sealed(value,'material_sha256')


def validate_history_state(bank,runsha,rank,allowed_tokens=None,maximum_update=None):
    if(bank.get('schema')!=SCHEMA or bank.get('protocol')!=PROTOCOL or bank.get('run')!=runsha or bank.get('rank')!=rank
       or len(bank['entries'])>512 or len({e['token']for e in bank['entries']})!=len(bank['entries'])or bank['admissions']<len(bank['entries'])):
        raise ValueError('Historicalbank lineage/rank/capacity changed')
    rng=torch.Generator();rng.set_state(bank['rng_state'].cpu())
    for e in bank['entries']:
        _check_witness(e,runsha,require_high=True)
        if e['condition']['identity']!=bank['condition_identity']or e['protocol_sha256']!=bank['protocol_sha256']:
            raise ValueError('Historicalbank mixes observation/scorer identities')
        if maximum_update is not None and e['source_version']['update']>maximum_update:raise ValueError('Future actor in historicalbank')
        if allowed_tokens is not None:
            if e['token']not in allowed_tokens or isinstance(allowed_tokens,dict)and allowed_tokens[e['token']]!=e['log']:
                raise ValueError('Historical teacher is outside registered TRAIN population')
    for k in ('admissions','replacements','not_retained','upgrades','windows','selected_count','regression_count','dominated_teacher_count','active_count','history_B1_queries_local'):
        if type(bank[k])is not int or bank[k]<0:raise ValueError('Historicalbank counter changed '+k)
    if not(bank['history_B1_queries_local']==bank['selected_count']<=bank['windows']and bank['active_count']+bank['dominated_teacher_count']==bank['selected_count']and bank['regression_count']<=bank['active_count']):
        raise ValueError('Historicalwindow/query/gate accounting inconsistent')
    for k in ('hshare_sum','forward_seconds_local','scoring_seconds_local'):
        if not math.isfinite(bank[k])or bank[k]<0:raise ValueError('Invalid historical loss/cost counter '+k)
    if abs(bank['hshare_sum']-(.05*bank['active_count']+.20*bank['regression_count']))>1e-8:raise ValueError('Historical beta-budget shares changed')
    return bank


def validate_history_material(material,bank_after=None,*,allowed_tokens=None):
    p=material;verify_seal(p,'material_sha256')
    if p.get('schema')!=SCHEMA or p.get('protocol')!=PROTOCOL or p['version']['run']!=p['run']or p.get('PG_eligible')is not False:
        raise ValueError('Historical material protocol/lineage changed')
    before=p['bank_before'];validate_history_state(before,p['run'],p['rank'],allowed_tokens,maximum_update=p['version']['update'])
    if p['current']is not None:
        _check_witness(p['current'],p['run'])
        if p['current']['source_version']!=p['version']:raise ValueError('Historical regression gate was not scored at currentbehavior')
    expected,index,teacher,regression,dominated,h=_advance(before,p['current'],p['version'])
    if(index!=p['selection_index']or digest(teacher)!=digest(p['teacher'])or p['window']!=expected['windows']or p['enabled']!=(h>0)
       or p['hshare']!=h or p['regression']!=regression or p['dominated_teacher']!=dominated
       or p['actual_B1_queries']!=int(teacher is not None)or p['bank_after_sha256']!=digest(expected)):
        raise ValueError('Historical selection/privateRNG/gate/budget transaction changed')
    if teacher is not None:
        if teacher['source_version']['update']>=p['version']['update']or p['current']['token']!=teacher['token']or p['current']['log']!=teacher['log']:
            raise ValueError('Teacher must come from an older own-run actor for this same scene')
    if h>0:
        seed=seed_for(SCHEMA,42,p['run'],p['version']['update'],p['version'].get('adapter_sha256'),teacher['token'],p['rank']);g=torch.Generator().manual_seed(seed)
        sigma=.1+.8*torch.rand(2,generator=g,dtype=torch.float32);noise=torch.randn((2,8,3),generator=g,dtype=torch.float32)
        if p['selection_seed']!=seed:raise ValueError('HistoricalCFM private seed changed')
        for k,want in [('sigma',sigma),('noise',noise)]:
            x=p[k]
            if x.requires_grad or x.dtype!=want.dtype or x.shape!=want.shape or not torch.equal(x.cpu(),want):raise ValueError('Historical detached '+k+' changed')
    elif any(p[k]is not None for k in ('sigma','noise','selection_seed')):raise ValueError('Inactive historical auxiliary retained trainingdraws')
    if bank_after is not None and digest(bank_after)!=p['bank_after_sha256']:raise ValueError('Historical window not committed exactly once')
    return p


def history_loss(model,material,calibstate):
    p=material;zero=next(model.parameters()).detach().new_zeros(())
    metrics=dict(history_enabled=zero+int(p['enabled']),history_hshare=zero+p['hshare'],history_regression=zero+int(p['regression']),
        history_dominated_teacher=zero+int(p['dominated_teacher']),history_B1_queries=zero+p['actual_B1_queries'],history_raw_cfm=zero,history_raw_ode=zero,
        history_scale_fallback=zero,history_scale_cfm=zero,history_scale_ode=zero)
    if not p['enabled']:return None,metrics  #Root must execute the genuine freshleaf*0 fallbackforward on thisrank.
    target=p['teacher']['clean'].detach().to(model.device);sigma=p['sigma'].detach().to(model.device);noise=p['noise'].detach().to(model.device)
    expanded=target.expand(2,-1,-1);state=(1-sigma[:,None,None])*expanded+sigma[:,None,None]*noise
    descriptor=p['current']['condition'];audit=model._probability_audit
    try:
        raw=condition_for_batch(model,descriptor,2)
        with model._context_call(raw)as condition:velocity=model.action_velocity(state,sigma*1000,condition)
        cfm=(velocity.float()-(noise-expanded)).square().mean()
        prediction=predict_clean_x(model,descriptor,p['current']['last_state'].to(model.device),p['current']['last_sigma'].to(model.device))
        ode=F.smooth_l1_loss(prediction.float(),target)
    finally:model._probability_audit=audit
    fallback=calibstate is None or calibstate.get('effective_updates',0)==0
    if calibstate is not None:validate_calibration_state(calibstate)
    sc=1. if fallback else calibstate['scale_cfm'];so=1. if fallback else calibstate['scale_ode'];total=.75*sc*cfm+.25*so*ode
    if not torch.isfinite(total):raise FloatingPointError('Nonfinite proactive historical auxiliary')
    metrics.update(history_raw_cfm=cfm.detach(),history_raw_ode=ode.detach(),history_scale_fallback=zero+int(fallback),history_scale_cfm=zero+sc,history_scale_ode=zero+so)
    return total,metrics
