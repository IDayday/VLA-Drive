"""Committed, own-run high-quality B1 targets, including verified policy zero."""
import copy
import hashlib
import json
import numpy as np

SCHEMA='d46_FPO_verified_B1_bank_v1'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def strict(record):
    m=record['metrics']
    return (record.get('status')=='OK'and m['score']>=.90 and
            all(m[k]==1 for k in('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','comfort')))


def initial(run):
    return dict(schema=SCHEMA,run=run,commits=0,teachers={},order=[],rows={},digest=fingerprint(dict(run=run,empty=True)))


def validate_teacher(teacher,run,update):
    v=teacher['version'];a=np.asarray(teacher['action'])
    if(v['run']!=run or not 0<=v['update']<=update or not strict(teacher['record'])
            or a.shape!=(8,3)or not np.isfinite(a).all()):
        raise ValueError('Foreign/unverified/future B1 history teacher')
    return teacher


def best_teacher(*candidates,run,update):
    values=[validate_teacher(t,run,update)for t in candidates if t is not None]
    return max(values,key=lambda t:(t['record']['metrics']['score'],t['version']['update']))if values else None


def commit(bank,deltas,legal_tokens,*,capacity):
    tokens=[d['token']for d in deltas]
    if len(tokens)!=len(set(tokens)):raise ValueError('Duplicate global TRAIN history delta')
    for d in sorted(deltas,key=lambda x:x['token']):
        token=d['token'];version=d['version']
        if token not in legal_tokens or version['run']!=bank['run']:raise ValueError('Non-TRAIN or foreign history update')
        if not 0<=d['nominal_score']<=1:raise ValueError('Invalid B1 score')
        bank['rows'][token]=dict(latest_nominal=d['nominal_score'],version=version)
        teacher=best_teacher(bank['teachers'].get(token),d.get('teacher'),run=bank['run'],update=version['update'])
        if teacher is not None:
            if token not in bank['teachers']:bank['order'].append(token)
            bank['teachers'][token]=copy.deepcopy(teacher)
    while len(bank['order'])>capacity:bank['teachers'].pop(bank['order'].pop(0))
    bank['digest']=fingerprint(dict(previous=bank['digest'],delta=sorted(deltas,key=lambda x:x['token'])))
    bank['commits']+=1


def retention_share(teacher,nominal_score,*,floor=.05,regression_extra=.20):
    return 0. if teacher is None else floor+regression_extra*float(teacher['record']['metrics']['score']-nominal_score>=.02)
