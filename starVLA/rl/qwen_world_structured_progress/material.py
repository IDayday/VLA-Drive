"""Typed synthetic pool and exact teacher selection/recovery; never PG replay."""
import copy
import numpy as np
import torch
from starVLA.rl.qwen_world_performance import nominal_improvement as N
from starVLA.rl.qwen_world_safety_exploration.retention_material import digest,sealed,verify_seal,validate_feedback
from starVLA.rl.qwen_world_performance.ode_target import strict_high
from .recipe import CONFIG
from .geometry import propose_family,InvalidGeometry

_READER=None


def bind_reader(reader):
    global _READER
    _READER=reader


def get_reader():
    global _READER
    if _READER is None:
        from starVLA.rl.qwen_world_research_data import ResearchSceneData
        from .gt_reader import GTReader
        _READER=GTReader(ResearchSceneData('/mnt/project/qwen-world-rl-optimization-20261030/data/NAVTRAIN',role='NAVTRAIN'))
    return _READER


def trigger(leaf,nominal,extra):
    return bool(leaf.get('valid_scene',True) and nominal is not None and not strict_high(nominal['feedback'])
        and not any(strict_high(r) for r in leaf['records']) and extra is not None
        and not any(strict_high(r) for r in extra['records']))


def generate(leaf,nominal,reader,*,saved_GT_source=None):
    """Pure deterministic CPU geometry; no random draw or model call."""
    ego=reader.current_ego(leaf['token'],leaf['log'])
    gt=reader.read(leaf['token'],leaf['log']) if saved_GT_source is None else reader.validate_saved(saved_GT_source,leaf['token'],leaf['log'])
    requested=[];clean=[];descriptors=[];status={}
    bases=[]
    if gt['valid']:bases.append(('GT_path',gt['requested']))
    else:status['GT_path']='GT_source_rejected:'+str(gt['reason'])
    bases.append(('B1_path',nominal['requested']))
    for family,base in bases:
        try:
            r,c,d=propose_family(torch.as_tensor(base).detach().float().cpu().reshape(8,3).numpy(),float(ego[0]),family)
            requested.extend(r);clean.extend(c);descriptors.extend(d);status[family]='generated8'
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:status[family]='geometry_rejected:'+type(exc).__name__+':'+str(exc)
    requests=torch.from_numpy(np.stack(requested)) if requested else torch.empty(0,8,3,dtype=torch.float32)
    targets=torch.stack(clean) if clean else torch.empty(0,8,3,dtype=torch.float32)
    if len({digest(x) for x in requests})!=len(requests):
        # Deduplicate rather than spend the scoring budget twice.
        keep=[];seen=set()
        for i,x in enumerate(requests):
            key=digest(x)
            if key not in seen:seen.add(key);keep.append(i)
        requests=requests[keep];targets=targets[keep];descriptors=[descriptors[i] for i in keep]
    return dict(ego=ego,GT_source=gt,requests=requests,clean=targets,descriptors=descriptors,family_status=status)


@torch.no_grad()
def collect_pool(leaf,nominal,extra,service,reader,rank):
    active=trigger(leaf,nominal,extra)
    value=dict(schema=CONFIG['schema'],protocol=copy.deepcopy(CONFIG),token=leaf['token'],log=leaf['log'],rank=rank,
        source_version=copy.deepcopy(leaf['behavior_version']),protocol_sha256=leaf['protocol_sha256'],
        triggered=active,reader_identity=copy.deepcopy(reader.identity),generation=None,records=[],queries=0,
        PG_eligible=False,model_policy_density=None)
    if active:
        value['generation']=generate(leaf,nominal,reader);requests=value['generation']['requests']
        if len(requests)>16:raise ValueError('Synthetic teacher-only query cap')
        value['records']=[] if not len(requests) else service.score((leaf['token'],leaf['log'],requests.numpy()))
        value['queries']=len(requests)
        if len(value['records'])!=len(requests):raise ValueError('Scorer omitted synthetic proposals')
        for request,record in zip(requests,value['records']):
            validate_feedback(record,leaf['token'],leaf['log'],request,leaf['protocol_sha256'])
            if record.get('cache')!=nominal['feedback'].get('cache'):raise ValueError('Synthetic/B1 official reference cache mismatch')
    return sealed(value,'material_sha256')


def select_teacher(leaf,nominal,extra,config,pool):
    if config!=N.QUALITY_CONFIG:raise ValueError('S7 keeps the exact S6 loss-component config')
    if not pool['triggered']:return N._select_teacher(leaf,nominal,extra,config)
    good=[i for i,r in enumerate(pool['records']) if strict_high(r)]
    if not good:return None,None,'no_strict_high_teacher'
    i=max(good,key=lambda i:float(pool['records'][i]['metrics']['score']));record=pool['records'][i]
    score=float(record['metrics']['score']);b1=float(nominal['feedback']['metrics']['score'])
    if score<b1+.02:return None,None,'teacher_gain_below_margin'
    clean=pool['generation']['clean'][i].clone();requested=pool['generation']['requests'][i]
    target=dict(origin='structured_progress_'+pool['generation']['descriptors'][i]['family'],index=i,
        token=leaf['token'],log=leaf['log'],source_version=copy.deepcopy(leaf['behavior_version']),
        feedback=copy.deepcopy(record),clean_sha256=digest(clean),requested_sha256=digest(requested),
        nominal_score=b1,score=score,gain=score-b1,PG_eligible=False,
        generation='explicit_synthetic_not_a_model_sample',structured_pool_sha256=pool['material_sha256'])
    return clean,target,None


def validate_pool(pool,leaf,nominal,extra,rank,reader=None):
    verify_seal(pool,'material_sha256');reader=get_reader() if reader is None else reader
    expected={'schema','protocol','token','log','rank','source_version','protocol_sha256','triggered','reader_identity',
        'generation','records','queries','PG_eligible','model_policy_density','material_sha256'}
    if (set(pool)!=expected or pool['schema']!=CONFIG['schema'] or pool['protocol']!=CONFIG
            or pool['token']!=leaf['token'] or pool['log']!=leaf['log'] or pool['rank']!=rank
            or pool['source_version']!=leaf['behavior_version'] or pool['protocol_sha256']!=leaf['protocol_sha256']
            or pool['reader_identity']!=reader.identity or pool['PG_eligible'] is not False
            or pool['model_policy_density'] is not None or pool['triggered']!=trigger(leaf,nominal,extra)):
        raise ValueError('Wrong typed structured teacher pool/source/rank/gate')
    if not pool['triggered']:
        if pool['generation'] is not None or pool['records'] or pool['queries']:raise ValueError('Inactive pool has hidden queries/teacher')
        return pool
    generation=pool['generation']
    if not isinstance(generation,dict):raise ValueError('Missing detached synthetic generation')
    for key,tail in (('requests',(8,3)),('clean',(8,3))):
        value=generation.get(key)
        if not torch.is_tensor(value) or value.dtype!=torch.float32 or value.ndim!=3 or tuple(value.shape[1:])!=tail or value.requires_grad or not torch.isfinite(value).all():
            raise ValueError('Invalid detached synthetic '+key)
    generated=generate(leaf,nominal,reader,saved_GT_source=generation['GT_source'])
    if digest(generated)!=digest(pool['generation']):raise ValueError('Saved synthetic geometry/source/actual request changed')
    n=len(generated['requests'])
    if pool['queries']!=n or len(pool['records'])!=n or not 0<=n<=16:raise ValueError('Synthetic actual query count changed')
    for request,record in zip(generated['requests'],pool['records']):
        validate_feedback(record,leaf['token'],leaf['log'],request,leaf['protocol_sha256'])
        if record.get('cache')!=nominal['feedback'].get('cache'):raise ValueError('Saved synthetic reference cache changed')
    return pool


def validate_nominal(leaf,*,rank,expected_epoch=None,config=None):
    p=leaf['nominal_improvement']
    if 'structured_progress' not in p:raise ValueError('S7 material missing its typed pool, even when inactive')
    pool=validate_pool(p['structured_progress'],leaf,p['nominal'],p['extra'],rank)
    return N.validate_material(leaf,rank=rank,expected_epoch=expected_epoch,config=N.QUALITY_CONFIG,
        teacher_selector=lambda a,b,c,d:select_teacher(a,b,c,d,pool))


def local_counts(window):
    result=dict(queries=0,triggered=0,enabled=0,GT_rejected=0,GT_qualified=0,B1_qualified=0)
    for leaf in window['samples']:
        p=leaf['nominal_improvement'];pool=p['structured_progress']
        result['queries']+=pool['queries'];result['triggered']+=int(pool['triggered'])
        result['enabled']+=int(p['enabled'] and str(p['teacher']['origin']).startswith('structured_progress_'))
        if pool['generation'] is not None:
            result['GT_rejected']+=int(not pool['generation']['GT_source']['valid'])
            for d,r in zip(pool['generation']['descriptors'],pool['records']):
                if strict_high(r) and r['metrics']['score']-p['nominal']['feedback']['metrics']['score']>=.02:
                    result['GT_qualified' if d['family']=='GT_path' else 'B1_qualified']+=1
    return result
