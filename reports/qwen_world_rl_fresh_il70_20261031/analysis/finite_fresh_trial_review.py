"""One finite, complete-only CPU review of a sealed fresh-trial registration.

Uses existing metric loaders/pair math only. Never calls an earlier review's
analyze_ready, dispatches work, watches logs, requests STOP, or changes training.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import tempfile
import uuid
import numpy as np

from rescue_pair_analyzer import value, contrast, METRICS
from safety_fresh_pair_review import il_value, matched
from safety_evidence_reviewer import atomic, sha, now, sealed

R=Path('/mnt/project/qwen-world-rl-optimization-20261030')
SCHEMA='world_finite_fresh_IL70_trial_matched_review_v1'
DEFAULT_OUT=R/'analysis/FPO_FULLTIME_MATCHED_REVIEW_20261031.json'


def fp(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def read(p):return json.loads(Path(p).read_text())


def stable_provenance(value):
    """Cache-read flags are execution details, not changed scientific evidence."""
    if isinstance(value,dict):
        return {k:stable_provenance(v) for k,v in value.items() if k!='cache_reused'}
    if isinstance(value,list):return [stable_provenance(v) for v in value]
    return value


def registration(path,root):
    p=Path(path).resolve();reg=sealed(p)
    if not reg.get('schema'):raise ValueError('Registration schema required')
    rows=reg.get('arms',[])
    if isinstance(rows,dict):rows=[dict(v,id=k) for k,v in rows.items()]
    arms=[r if isinstance(r,str) else r.get('id',r.get('arm')) for r in rows]
    gates=reg.get('gates',[])
    if (not arms or any(not isinstance(a,str) or not a for a in arms)
            or len(set(arms))!=len(arms) or not gates
            or any(type(u) is not int or u<=0 for u in gates)
            or sorted(set(gates))!=gates):
        raise ValueError('Explicit distinct registered arms and increasing gates required')
    comparisons=reg.get('comparisons',{})
    primary=comparisons.get('primary')
    secondary=comparisons.get('secondary',[])
    if (not isinstance(primary,str) or not primary or not isinstance(secondary,list)
            or any(not isinstance(a,str) or not a for a in secondary)
            or len(set([primary,*secondary]))!=len(secondary)+1
            or set(arms)&set([primary,*secondary])):
        raise ValueError('Explicit primary/secondary same-budget controls required')
    reviewer=reg.get('reviewer',{})
    if reviewer.get('sha256') not in (None,sha(__file__)):
        raise ValueError('Registration reviewer source hash changed')
    if reviewer.get('script') and Path(reviewer['script']).resolve()!=Path(__file__).resolve():
        raise ValueError('Registration selected a different reviewer')
    output=Path(reviewer.get('output',str(DEFAULT_OUT)))
    if not output.is_absolute():output=root/output
    forbidden={R/'analysis/SAFETY_FRESH_PAIR_MATCHED_REVIEW_20261031.json',
               R/'analysis/RESCUE_PAIR_MATCHED_REVIEW.json',R/'analysis/LR_STUDY_MATCHED_REVIEW_20261031.json'}
    if output.resolve() in forbidden:raise ValueError('Never overwrite an existing running study review')
    return reg,arms,gates,[primary,*secondary],output.resolve()


def signature(a,b,il):
    return fp(dict(target=a['label'],baseline=b['label'],
        target_provenance=stable_provenance(a['provenance']),
        baseline_provenance=stable_provenance(b['provenance']),
        IL70_provenance=stable_provenance(il['provenance']),
        token_order=fp(a['tokens']),physical_log_order=fp(a['logs'].tolist()),
        protocol=a['protocol'],high='IL70 score>=.9',low='IL70 score<.9',bootstrap=2000,seed=42))


def finite_result(result):
    for key in ('delta_PDMS','target_PDMS','baseline_PDMS'):
        if not math.isfinite(result[key]):raise ValueError('Unidentifiable/nonfinite paired statistic')
    if not all(math.isfinite(x) for x in result['CI95_PDMS']):
        raise ValueError('Nonfinite paired CI; never publish an imputed interval')
    return result


def annotate(a,b,il,cohort):
    matched(a,b);matched(a,il)
    if cohort=='NAVTEST' and len(set(a['logs']))!=44:
        raise ValueError('Full NAVTEST must have44 physical logs')
    result=finite_result(contrast(a,b))
    result.update(status='COMPLETE',completed_at=now(),cohort=cohort,
        evidence_signature=signature(a,b,il),
        provenance=dict(target=a['provenance'],baseline=b['provenance'],IL70=il['provenance']),
        canonical_target=dict(target_PDMS_ge92=bool(result['target_PDMS']>=92),target=92.0),
        matching=dict(tokens_logs_protocol_matched=True,
            canonical_cache_hashes_matched='hashes' in a and 'hashes' in b))
    high=il['values'][:,0]>=.9
    for name,mask,definition in (
        ('fixed_IL70_high_ge09',high,'fixed originalIL70 canonical score>=.9'),
        ('fixed_IL70_low_lt09',~high,'fixed originalIL70 canonical score<.9')):
        if mask.any():
            part=finite_result(contrast(a,b,mask))
            part.update(status='COMPLETE',definition=definition,
                baseline_population='same declared target/control pair within this fixed IL cohort',
                cohort_sha256=fp([t for t,keep in zip(a['tokens'],mask) if keep]))
            result[name]=part
        else:
            result[name]=dict(status='NOT_ESTIMABLE_EMPTY_FIXED_IL_COHORT',scenes=0,
                              definition=definition,delta_PDMS=None,CI95_PDMS=None)
    return result


def analyze(registration_path,*,root=R):
    reg,arms,gates,controls,outpath=registration(registration_path,root)
    regpath=Path(registration_path).resolve();regsha=sha(regpath)
    outpath.parent.mkdir(parents=True,exist_ok=True)
    with outpath.with_suffix(outpath.suffix+'.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        if outpath.exists():
            d=read(outpath)
            if (d.get('schema')!=SCHEMA or d.get('registration',{}).get('seal')!=reg['sha256']
                    or d.get('source_sha256')!=sha(__file__)):
                raise ValueError('Output belongs to a different registered review/source')
        else:
            d=dict(schema=SCHEMA,status='PARTIAL_ACTUAL_COMPLETE_RESULTS_ONLY',
                created_at=now(),contrasts={},TRAIN_diagnostics={},completed_scores={},
                completed_TRAIN_scores={},arms=arms,gates=gates,controls=controls,
                registration=dict(path=str(regpath),file_sha256=regsha,seal=reg['sha256'],schema=reg['schema']),
                source_sha256=sha(__file__),source_file=str(Path(__file__).resolve()),
                initialization='registered sameIL70 freshAdam/seed42/update0; not RL10k warmstart/nativefork',
                target_PDMS=92.0,cohort_definitions=dict(high='fixed IL70 score>=.9',low='fixed IL70 score<.9'),
                uncertainty='paired scene-weighted physical-log bootstrap2000 seed42; NAVTEST44logs; TRAIN actual physical-log count; fixed one training/inference seed, no across-seed claim',
                interpretation='NAVTEST diagnostics inform model review only, never TRAIN token/teacher selection; TRAIN1024 is part of eligible training population, not independent heldout.',
                scope=dict(CPU_only=True,GPU_dispatch=0,new_scores=0,process_signals=0,STOP_files=0,
                    journals_read=0,partial_scores_imputed=False,old_review_or_owner_edited=False))
        changed=False;pending=[];scores=0;train_scores=0
        for cohort,key,scorekey in [('NAVTEST','contrasts','completed_scores'),
                                   ('TRAIN_DIAGNOSTIC','TRAIN_diagnostics','completed_TRAIN_scores')]:
            il=None
            for u in gates:
                vals={a:value(a,u,cohort) for a in arms}
                if not any(v is not None for v in vals.values()):
                    for arm in arms:pending.append(dict(arm=arm,updates=u,cohort=cohort,missing='target_complete_evaluation'))
                    continue
                if il is None:il=il_value(cohort)
                bases={a:value(a,u,cohort) for a in controls};bases['IL70']=il
                for arm,a in vals.items():
                    if a is None:
                        pending.append(dict(arm=arm,updates=u,cohort=cohort,missing='target_complete_evaluation'));continue
                    matched(a,il)
                    point=dict(PDMS=float(a['values'][:,0].mean()*100),updates=u,cohort=cohort,
                        ge92=bool(a['values'][:,0].mean()*100>=92),provenance=a['provenance'])
                    if a['label'] not in d[scorekey]:changed=True
                    d[scorekey][a['label']]=point
                    if cohort=='NAVTEST':scores+=1
                    else:train_scores+=1
                    for control,b in bases.items():
                        name=f'{arm}_u{u:06}_minus_{control}'
                        if b is None:
                            pending.append(dict(arm=arm,updates=u,cohort=cohort,control=control,missing='samebudget_control_complete_evaluation'));continue
                        sig=signature(a,b,il)
                        if name in d[key]:
                            if d[key][name].get('evidence_signature')!=sig:
                                raise ValueError('Previously calculated pair evidence changed: '+name)
                            continue  # No repeated bootstrap for an existing actual-complete pair.
                        d[key][name]=annotate(a,b,il,cohort);changed=True
        expected=len(arms)*len(gates)*(len(controls)+1)
        d['progress']=dict(complete_NAVTEST_contrasts=len(d['contrasts']),
            complete_TRAIN_contrasts=len(d['TRAIN_diagnostics']),expected_each=expected,
            complete_new_NAVTEST_models=scores,complete_new_TRAIN_models=train_scores,
            expected_models_each=len(arms)*len(gates),pending=pending)
        d['status']='COMPLETE' if (len(d['contrasts'])==expected and len(d['TRAIN_diagnostics'])==expected
            and scores==len(arms)*len(gates) and train_scores==len(arms)*len(gates)) else 'PARTIAL_ACTUAL_COMPLETE_RESULTS_ONLY'
        d['updated_at']=now();d['registration']['file_sha256']=regsha
        d.pop('sha256',None);d['sha256']=fp(d);atomic(outpath,d)
        if changed:
            atomic(root/'reporting/refresh_request.json',dict(id='finitefreshtrial-'+uuid.uuid4().hex,
                scientific_result=False,evidence_kind='completed_supplemental_analysis',
                at=now(),evidence=str(outpath)))
        return d,outpath


def self_test():
    """Actual pair math on balanced synthetic cohorts; never opens live jobs."""
    global value,il_value,contrast
    old=(value,il_value,contrast)
    availability={'late':False};calls={'contrast':0}
    def synthetic(label,cohort,delta=0.):
        groups=44 if cohort=='NAVTEST' else 6
        tokens=[f't{i:03}' for i in range(groups*2)]
        logs=np.array([f'physical{i//2:03}' for i in range(groups*2)])
        ep=np.tile([1.,.5],groups);ep=ep+np.where(ep==1,-delta,delta)
        m=np.ones((len(tokens),7));m[:,4]=ep;m[:,0]=(5*ep+7)/12
        return dict(label=label,values=m,tokens=tokens,logs=logs,protocol='protocol',
                    provenance=dict(summary=label,summary_sha256=fp(label),cache_reused=False))
    def fake_value(arm,u,cohort):
        if not availability['late'] and (cohort!='NAVTEST' or u!=1000 or arm=='CONTROL_SECONDARY'):return None
        return synthetic(f'{arm}_u{u:06}_{cohort}',cohort,.02 if arm=='TRIAL' else .01)
    def counted(*args,**kw):
        calls['contrast']+=1;return old[2](*args,**kw)
    try:
        value=fake_value;il_value=lambda c:synthetic('IL70_'+c,c);contrast=counted
        with tempfile.TemporaryDirectory(prefix='finite-fresh-review-') as temporary:
            root=Path(temporary);regpath=root/'registration.json';out=root/'analysis/review.json'
            reg=dict(schema='synthetic_fresh_review_registration_v1',arms=[dict(id='TRIAL')],gates=[1000,5000],
                comparisons=dict(primary='CONTROL_PRIMARY',secondary=['CONTROL_SECONDARY']),reviewer=dict(output=str(out)))
            reg['sha256']=fp(reg);atomic(regpath,reg)
            first,_=analyze(regpath,root=root)
            assert len(first['contrasts'])==2 and not first['TRAIN_diagnostics'] and first['status']!='COMPLETE'
            assert not any('005000' in k for k in first['contrasts'])
            for entry in first['contrasts'].values():
                assert entry['physical_log_clusters']==44
                assert entry['fixed_IL70_high_ge09']['delta_PDMS']<0
                assert entry['fixed_IL70_low_lt09']['delta_PDMS']>0
                assert 'provenance' in entry and 'components' in entry and 'zeros' in entry
            before=calls['contrast'];analyze(regpath,root=root);assert calls['contrast']==before
            availability['late']=True
            complete,_=analyze(regpath,root=root)
            assert complete['status']=='COMPLETE' and len(complete['contrasts'])==6 and len(complete['TRAIN_diagnostics'])==6
            assert all(x['physical_log_clusters']==6 for x in complete['TRAIN_diagnostics'].values())
            before=calls['contrast'];analyze(regpath,root=root);assert calls['contrast']==before
            refresh=read(root/'reporting/refresh_request.json')
            assert refresh['scientific_result'] is False and refresh['id']
            bad=copy.deepcopy(reg);bad['sha256']='invalid';atomic(regpath,bad)
            try:analyze(regpath,root=root)
            except ValueError:pass
            else:raise AssertionError('Unsealed registration accepted')
            return dict(status='PASS',partial_inputs_not_imputed=True,all_pairs_complete_required=True,
                high_low_directions_separated=True,already_computed_bootstraps_reused=True,
                NAVTEST44clusters=True,TRAIN_actual_clusters=6,administrative_refresh_false=True,
                invalid_registration_rejected=True,live_files_opened=False,new_GPU=0)
    finally:value,il_value,contrast=old


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration');p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:print(json.dumps(self_test()))
    else:
        if not args.registration:p.error('--registration is required')
        result,path=analyze(args.registration)
        print(json.dumps(dict(output=str(path),status=result['status'],progress=result['progress']),ensure_ascii=False))
