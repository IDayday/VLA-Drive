"""D6 TRAIN-only self-history and deterministic distinct-scene mixture.

Archive labels are real own-run B1 replays, not predictions or external search.
The first observed nominal is a self-history baseline, not an IL44 census.
"""
import copy
import hashlib
import json
import numpy as np
import torch
from .native_cleanx_aux import verified_index
from .safety_credit import FACTORS

SCHEMA='D6_committed_TRAIN_nominal_history_v1'


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def initial(run):
    return dict(schema=SCHEMA,run=run,rows={},teachers={},teacher_order=[],commits=0,digest=digest(dict(run=run,empty=True)))


def update(history,records,legal_tokens,*,capacity=8192):
    """All ranks apply the same globally finalized, token-sorted delta."""
    tokens=[r['token']for r in records]
    if len(tokens)!=len(set(tokens)):raise ValueError('D6 history update has duplicate scenes')
    for value in sorted(records,key=lambda r:r['token']):
        token=value['token'];version=value['version']
        if token not in legal_tokens or version['run']!=history['run']:raise ValueError('Foreign/non-TRAIN self-history')
        score=float(value['record']['metrics']['score'])
        if not np.isfinite(score)or not 0<=score<=1:raise ValueError('Invalid actual nominal score')
        row=history['rows'].setdefault(token,dict(first_nominal_score=score,latest_nominal_score=score))
        row['latest_nominal_score']=score
        if verified_index([value['record']],0.,improvement=0.)is not None:
            old=history['teachers'].get(token)
            if old is None or score>old['score']:
                clean=value['normalized_clean_action']
                if np.asarray(clean).shape!=(8,3)or not np.isfinite(clean).all():raise ValueError('Invalid native B1 teacher coordinates')
                history['teachers'][token]=dict(score=score,normalized_clean_action=clean,
                    record=value['record'],version=version,source='own_previous_committed_single_B1_ODE')
                if token in history['teacher_order']:history['teacher_order'].remove(token)
                history['teacher_order'].append(token)
        while len(history['teacher_order'])>capacity:
            history['teachers'].pop(history['teacher_order'].pop(0))
    history['digest']=digest(dict(previous=history['digest'],delta=sorted(records,key=lambda r:r['token'])))
    history['commits']+=1
    return history


def pools(history):
    hard=sorted(t for t,v in history['rows'].items()if v['latest_nominal_score']<.9)
    regression=sorted(t for t,v in history['rows'].items()if v['first_nominal_score']-v['latest_nominal_score']>=.02)
    return hard,regression


def select(history,uniform,*,collection,enabled):
    """60/25/15 stochastic roles; unavailable/duplicate pool entries fall back.

    Each input item has token+padding, from the ordinary deterministic sampler.
    Padding remains padding. Sampling has no test-scene feedback or labels.
    """
    if len({x['token']for x in uniform})!=len(uniform):raise ValueError('Uniform base batch must be distinct')
    if not enabled:return [dict(x,source='uniform')for x in uniform]
    hard,regression=pools(history);generator=torch.Generator().manual_seed(42000000+collection)
    roles=torch.multinomial(torch.tensor([.60,.25,.15]),len(uniform),replacement=True,generator=generator).tolist()
    # Reserve base tokens for future fallback, so a pool draw cannot consume
    # another position's unique fallback and silently duplicate a scene.
    reserved={x['token']for x in uniform};used=set();result=[]
    for base,role in zip(uniform,roles):
        token=base['token'];source='uniform';pool=hard if role==1 else regression if role==2 else []
        if pool and not base['padding']:
            order=torch.randperm(len(pool),generator=generator).tolist()
            match=next((pool[k]for k in order if pool[k]not in used and pool[k]not in reserved),None)
            if match is not None:token=match;source='hard_nominal'if role==1 else'regression_self_history'
        used.add(token);result.append(dict(token=token,padding=base['padding'],source=source))
    if len(used)!=len(result):raise ValueError('Mixture selection duplicated a global scene')
    return result


def historical_teacher(history,token,current_nominal,current_version):
    candidate=history['teachers'].get(token)
    if candidate is None:return None
    if candidate['version']['run']!=current_version['run']or candidate['version']['update']>=current_version['update']:
        raise ValueError('Teacher is not an earlier own-run committed policy')
    return candidate if candidate['score']-current_nominal>=.02 else None


def delta_from_scene(row):
    if row['padding']:return None
    # Keep exact simulator-derived metric labels but not large physical rollout
    # arrays. Their full evidence remains in the behavior file for this version.
    record=dict(metrics={k:float(v)for k,v in row['nominal_record']['metrics'].items()},status='OK')
    return dict(token=row['token'],record=record,version=copy.deepcopy(row['behavior_version']),
        normalized_clean_action=row['nominal_chain'][0,-1].detach().cpu().tolist())
