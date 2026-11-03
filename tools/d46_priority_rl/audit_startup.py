#!/usr/bin/env python3
"""Finite CPU-only audit of published D46 native startup material.

Loads only native0 trainable/Adam containers and tiny saved rank states. No
actor, parent tensor, model forward, scoring, CUDA, dispatch or owner mutation.
Re-running reuses native0 summaries keyed by immutable checkpoint inventory.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time

ROOT = Path('/mnt/project/d46-priority-rl-20261103')
SOURCE = ROOT / 'runtime/formal_v1'
SCIENCE_SHA = 'ff3013c4045e9505ae9cd39b9418247cc03a596d'
PARENT_SHA = '386a0a27bd197cb002501bcb90c343659001d2e6d4f7bad0cbd5e31a91f8477d'
REG_SHA = '584dbcf1103fb4f052baa843add861a33bca90023f47827e2a882ef5a66a9632'
ARMS = ('D0','D1','D2','D3_FPO','D3_AWM','D5')
ZERO_COUNTERS = ('updates','collections','scene_cursor','next_inner','valid_scene_exposures',
    'raw_scene_exposures','new_candidates','valid_PG_candidates','extra_B1_queries',
    'adaptive_G32_queries','padding_scene_exposures')
CACHE = ROOT/'qualification/.actual_startup_proof_cache'


def load(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
    return h.hexdigest()


def fp(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def atomic(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.writing-'+str(os.getpid()))
    with t.open('w') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(t,p)


def ref(p):
    p=Path(p)
    return {'path':str(p),'file_sha256':sha(p),'bytes':p.stat().st_size}


def verify_identity(v):
    b=dict(v);s=b.pop('sha256',None)
    checks={'self_seal':fp(b)==s,'world8':v.get('world')==8,
        'parent_model':v.get('parent_model_sha256')==PARENT_SHA,
        'parent_registration_v2':v.get('recipe',{}).get('parent_registration',{}).get('file_sha256')==REG_SHA}
    checks['all_pass']=all(checks.values())
    return checks


_source_cache={}
def source_check(v):
    files=v.get('source_files',{})
    key=fp(files)
    if key not in _source_cache:
        missing=[];changed=[]
        for n,s in files.items():
            p=SOURCE/n
            if not p.is_file():missing.append(n)
            elif sha(p)!=s:changed.append(n)
        _source_cache[key]={'source':str(SOURCE),'source_git_commit_expected':SCIENCE_SHA,
            'identity_source_files_sha256':key,'files_checked':len(files),'missing':missing,'changed':changed,
            'status':'PASS' if files and not missing and not changed else 'REVIEW_REQUIRED'}
    return _source_cache[key]


def seal(p,actual_payload_hashes=False):
    p=Path(p)
    if not (p/'COMPLETE').is_file():return {'status':'PENDING','checkpoint':str(p)}
    c=load(p/'COMPLETE');files=load(p/'files.json');bad=[]
    required={'identity.json','transaction.json','optimizer.pt'}|{f'rank{r}.pt'for r in range(8)}
    if c.get('world')!=8 or c.get('files_sha256')!=sha(p/'files.json'):bad.append('COMPLETE/inventory/world')
    if not required.issubset(files):bad.append('missing required inventory names')
    for n,s in files.items():
        q=p/n
        if q.parent.resolve()!=p.resolve() or not q.is_file():bad.append('missing/escaped '+n)
        elif (actual_payload_hashes or not n.endswith('.pt')) and sha(q)!=s:bad.append('changed '+n)
    return {'status':'PASS'if not bad else 'REVIEW_REQUIRED','checkpoint':str(p),'complete':c,
        'complete_file':ref(p/'COMPLETE'),'files_json':ref(p/'files.json'),
        'required_files_present':required.issubset(files),'full_payload_hashes_checked':actual_payload_hashes,
        'inventory_files':len(files),'problems':bad}


def tensor_hash(v):
    import torch
    x=v.detach().cpu().contiguous()
    return hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()


def native0(p):
    p=Path(p)
    if not (p/'COMPLETE').is_file():return {'status':'PENDING','checkpoint':str(p)}
    CACHE.mkdir(exist_ok=True)
    cache=CACHE/(sha(p/'files.json')+'.json')
    if cache.is_file():
        v=load(cache)
        if v.get('checkpoint')==str(p) and v.get('builder_sha256')==sha(__file__):return v
    import torch
    from starVLA.rl.d44_native_rl.recovery import Boundary
    s=seal(p,True);i=load(p/'identity.json');c=torch.load(p/'optimizer.pt',map_location='cpu',mmap=True,weights_only=False)
    ad=c['adapters'];adapter_records=[];nonfinite=[];b_nonzero=[]
    for n,t in sorted(ad.items()):
        finite=bool(torch.isfinite(t).all())
        if not finite:nonfinite.append(n)
        if n.endswith('lora_B') and bool(torch.count_nonzero(t)):b_nonzero.append(n)
        adapter_records.append({'name':n,'shape':list(t.shape),'dtype':str(t.dtype),'sha256':tensor_hash(t)})
    opt=c['optimizer'];sched=c['scheduler'];ranks=[];summaries=[];problems=[]
    for rank in range(8):
        own=torch.load(p/f'rank{rank}.pt',map_location='cpu',mmap=True,weights_only=False)
        state=own['state'];rng=own['rng'];rn={}
        for k,v in rng.items():
            if torch.is_tensor(v):rn[k]={'sha256':tensor_hash(v),'dtype':str(v.dtype),'numel':v.numel()}
            else:rn[k]={'sha256':hashlib.sha256(pickle.dumps(v,protocol=4)).hexdigest(),'type':type(v).__name__}
        expected=torch.Generator('cpu').manual_seed(420000+rank).get_state()
        good=(own.get('identity_sha256')==i['sha256'] and own['rollout'] is None
            and state['phase']=='READY' and all(state.get(k)==0 for k in ZERO_COUNTERS)
            and all(k in rng and rng[k] is not None for k in ('python','numpy','torch','cuda','rollout_generator'))
            and torch.equal(expected,rng['rollout_generator']))
        summary=Boundary.validate_boundary(state,own['rollout'],i,opt,sched)
        summaries.append(summary)
        ranks.append({'rank':rank,'status':'PASS'if good else 'REVIEW_REQUIRED',
            'identity_sha256':own.get('identity_sha256'),'state':state,'rollout_is_none':own['rollout']is None,
            'rng':rn,'private_generator_seed':420000+rank,'private_generator_matches_seed':torch.equal(expected,rng['rollout_generator'])})
    Boundary.validate_ranks(summaries)
    tx=load(p/'transaction.json')
    txok=tx.get('ranks')==summaries
    groups=[{k:v for k,v in g.items()if k!='params'}for g in opt['param_groups']]
    passes=(s['status']=='PASS' and verify_identity(i)['all_pass'] and source_check(i)['status']=='PASS'
        and s['complete']['updates']==0 and s['complete']['next_inner']==0 and len(opt['state'])==0
        and sched.get('last_epoch')==0 and not nonfinite and not b_nonzero and txok
        and all(r['status']=='PASS'for r in ranks))
    v={'schema':'d46_actual_native0_cpu_proof_v1','status':'PASS'if passes else 'REVIEW_REQUIRED',
        'checkpoint':str(p),'builder_sha256':sha(__file__),'seal':s,'identity_sha256':i['sha256'],
        'identity_checks':verify_identity(i),'source_check':source_check(i),'qualification':i['qualification'],
        'optimizer_state_entries':len(opt['state']),'optimizer_is_empty':len(opt['state'])==0,
        'scheduler':sched,'optimizer_groups':groups,'adapter_tensor_count':len(ad),
        'adapter_numel':sum(t.numel()for t in ad.values()),'adapter_material_sha256':fp(adapter_records),
        'adapter_nonfinite':nonfinite,'nonzero_LoRA_B_names':b_nonzero,
        'zero_LoRA_B_count':sum(n.endswith('lora_B')for n in ad),
        'transaction_matches_all8_actual_native_boundary_summaries':txok,'ranks':ranks,
        'scope':{k:v for k,v in i['scope'].items()if k not in ('trainable_names','adapters')},
        'trainable_names_sha256':fp(i['scope']['trainable_names'])}
    atomic(cache,v)
    return v


def metrics(root,identity,limit=10):
    rows=[];files=[];errors=[]
    for rank in range(8):
        p=Path(root)/f'metrics_rank{rank}.jsonl';current=[]
        if p.exists():
            with p.open()as f:
                for line in f:
                    if not line.strip():continue
                    try:d=json.loads(line)
                    except json.JSONDecodeError:
                        errors.append(f'rank{rank}:incomplete last JSON row');break
                    current.append(d)
                    if len(current)>=limit:break
            files.append({'rank':rank,'path':str(p),'bounded_rows_read':len(current),
                'prefix_sha256':fp(current)})
        rows.append(current)
    common=min(map(len,rows));present=sum(bool(r)for r in rows);fresh=[];grad=[];roles={};finite=True
    for rank,rr in enumerate(rows):
        for expected,r in enumerate(rr[:common],1):
            if r.get('updates')!=expected or r.get('run')!=identity['sha256']:errors.append(f'rank{rank}:wrong run/update {expected}')
            lo=r.get('losses',{});g=r.get('gradient_norm');
            if not isinstance(g,(int,float)) or not math.isfinite(g):finite=False
            else:grad.append(g)
            if any(isinstance(v,(int,float))and not math.isfinite(v)for v in lo.values()):finite=False
            if r.get('inner_completed')==1:fresh.append(lo.get('fresh_ratio_error'))
            for name,v in r.get('role_gradient_norms',{}).items():
                roles.setdefault(name,[]).append(v)
                if v is not None and not math.isfinite(v):finite=False
    first=[{'rank':rank,'updates':rr[0]['updates'],'losses':rr[0].get('losses'),
        'gradient_norm':rr[0].get('gradient_norm'),'role_gradient_norms':rr[0].get('role_gradient_norms'),
        'lr':rr[0].get('lr'),'scene_exposures':rr[0].get('scene_exposures'),
        'at':rr[0].get('at')}for rank,rr in enumerate(rows)if rr]
    arm=identity['recipe']['arm']
    objective={'kind':'CFM_energy_ratio'if arm=='D3_FPO'else'positive_advantage_weighted_matching'if arm=='D3_AWM'else'native_joint_Gaussian_transfer',
        'Gaussian_ratio_is_optimized_objective':arm not in('D3_FPO','D3_AWM'),
        'Gaussian_fresh_ratio_is_also_reported_as_initial_behavior_diagnostic':True}
    own_proxy_ok=True
    if arm=='D3_FPO':
        own_proxy_ok=(len(first)==8 and all(v['losses'].get('fpo_ratio_mean')==1. and v['losses'].get('fpo_ratio_max')==1.
            and v['losses'].get('fpo_joint_logratio_min')==0. and v['losses'].get('fpo_joint_logratio_max')==0. for v in first))
        objective['actual_first_FPO_proxy_ratio_all8_exact_identity']=own_proxy_ok
    if arm=='D3_AWM':objective['PPO_clip_is_not_part_of_matching_loss']=True
    ready=common>0 and not errors and finite and bool(fresh)and all(v==0 for v in fresh)and own_proxy_ok
    return {'optimized_objective':objective,'status':'PASS'if ready else ('PENDING'if common==0 else 'REVIEW_REQUIRED'),
        'rank_files_present':present,'all8_committed_prefix_updates_read':common,'per_rank_rows_read':list(map(len,rows)),
        'max_requested_updates':limit,'source_files':files,'first_actual_updates':first,
        'fresh_inner1_observations':len(fresh),'fresh_ratio_error_all_exact_zero':bool(fresh)and all(v==0for v in fresh),
        'actual_loss_and_gradient_finite':finite if common else None,
        'gradient_norm_range':[min(grad),max(grad)]if grad else None,
        'role_gradient_ranges':{k:{'min':min(x for x in v if x is not None)if any(x is not None for x in v)else None,
            'max':max(x for x in v if x is not None)if any(x is not None for x in v)else None,
            'inactive_none_count':sum(x is None for x in v)}for k,v in roles.items()},'problems':errors,
        'gradient_unit_note':'gradient_norm is native pre-clipping total; role_gradient_norms native recording may be post-clipping; no cross-unit strength inference'}


def attempts(jobs):
    out=[]
    for j in jobs:
        p=Path(j['job']);rr=p/'RESULT.json';run=p/'RUNNING.json';result=load(rr)if rr.is_file()else None
        log=p/'stdout.log';oom=False;excerpt=[]
        if result and result.get('returncode')not in(0,None)and log.is_file():
            with log.open(errors='replace')as f:
                for line in f:
                    if 'out of memory'in line.lower():oom=True;excerpt.append(line.strip()[:1200])
        out.append({'job':str(p),'host':j['host'],'source':j['source'],'source_sha':j['source_sha'],
            'worker_source':j.get('worker_source'),'worker_source_sha':j.get('worker_source_sha'),
            'command':j['command'],'resume':j['command'][j['command'].index('--resume')+1]if '--resume'in j['command']else None,
            'has_qualification_flag':'--qualification'in j['command'],'actual_result':ref(rr)if result else None,
            'returncode':result.get('returncode')if result else None,'result_status':result.get('status')if result else None,
            'actual_running':ref(run)if run.is_file()else None,
            'child_not_started':result.get('child_not_started')if result else None,'reason':result.get('reason')if result else None,
            'classification':'PRECHILD_RESOURCE_REJECTION_PRESERVED'if result and result.get('returncode')==75 and result.get('child_not_started')is True else ('STARTUP_OOM_PRESERVED'if oom else ('EXITED0'if result and result.get('returncode')==0 else ('FAILED_PRESERVED'if result else 'IN_PROGRESS_OR_ADMITTING'))),
            'oom_excerpt':excerpt[:2]})
    return out


def invocations(root,identity):
    rr=[]
    for p in sorted((Path(root)/'invocations').glob('*/rank*.json')):
        d=load(p);rr.append({'evidence':ref(p),'id':d.get('id'),'rank':d.get('rank'),
            'run':d.get('run'),'start_update':d.get('start_update'),'resume':d.get('resume'),
            'status':d.get('status'),'end_update':d.get('end_update')})
    ids={d['id']for d in rr};fresh=[d for d in rr if d['start_update']==0 and d['resume']is None and d['run']==identity['sha256']]
    ranks=sorted({d['rank']for d in fresh})
    return {'records':rr,'fresh_start_ranks':ranks,'all8_actual_fresh_invocations':ranks==list(range(8)),
        'invocation_ids':sorted(ids)}


def audit(root,kind,jobs):
    root=Path(root);p=root/'identity.json';out={'root':str(root),'job_attempts':attempts(jobs)}
    if not p.is_file():return dict(out,status='PENDING',reason='No actual run identity/native0 published yet')
    identity=load(p);out.update(identity_file=ref(p),identity_checks=verify_identity(identity),
        identity_sha256=identity['sha256'],qualification=identity['qualification'],source_check=source_check(identity),
        initial_native=native0(root/'checkpoints/update000000_inner0'),
        actual_metrics=metrics(root,identity),actual_invocations=invocations(root,identity))
    if kind=='qualification':
        parity=[];results=[]
        for rank in range(8):
            pp=root/f'initial_native_parity_rank{rank}.json';rp=root/f'result_rank{rank}.json'
            if pp.is_file():
                d=load(pp);good=(d.get('status')=='PASS'and d.get('ODE_chain_max_abs')==0
                    and d.get('original_cleanx_deployment_max_abs')==0 and d.get('parent')==PARENT_SHA
                    and d.get('reference_frozen')is True)
                parity.append({'rank':rank,'status':'PASS'if good else 'REVIEW_REQUIRED','evidence':ref(pp),
                    **{k:v for k,v in d.items()if k!='trainable'},'actual_trainable_numel':d.get('trainable',{}).get('trainable_numel')})
            if rp.is_file():
                d=load(rp);good=(d.get('rank')==rank and d.get('updates')==10 and d.get('next_inner')==0
                    and d.get('qualification')is True and d.get('status')=='QUALIFICATION_STOPPED'
                    and Path(d.get('checkpoint','')).resolve()==(root/'checkpoints/update000010_inner0').resolve()
                    and bool(d.get('frozen_before'))and d['frozen_before']==d.get('frozen_after'))
                results.append({'rank':rank,'check':'PASS'if good else 'REVIEW_REQUIRED','evidence':ref(rp),
                    **{k:v for k,v in d.items()if k in('updates','collections','next_inner','scene_cursor',
                        'valid_scene_exposures','new_candidates','status','qualification','checkpoint','frozen_before','frozen_after','host')}})
        out['initial_deployment_parity']=parity;out['actual_q10_results']=results
        out['all8_same_frozen_hash_before_and_after']=(len(results)==8 and len({d['frozen_before']for d in results})==1
            and all(d['frozen_before']==d['frozen_after']for d in results))
        out['q10_checkpoint']=seal(root/'checkpoints/update000010_inner0',True)
        passed=(len(parity)==len(results)==8 and all(d['status']=='PASS'for d in parity)
            and all(d['check']=='PASS'for d in results)and out['all8_same_frozen_hash_before_and_after']
            and out['q10_checkpoint']['status']=='PASS'
            and out['initial_native']['status']=='PASS'and out['actual_metrics']['status']=='PASS'
            and out['actual_metrics']['all8_committed_prefix_updates_read']>=10)
        out['status']='PASS'if passed else 'RUNNING_PARTIAL'
    else:
        cmdok=any(j['source_sha']==SCIENCE_SHA and '--resume'not in j['command'] and '--qualification'not in j['command']for j in jobs)
        out['actual_command_fresh_no_resume_or_qualification']=cmdok
        passed=cmdok and out['initial_native']['status']=='PASS'and out['actual_invocations']['all8_actual_fresh_invocations']and out['actual_metrics']['status']=='PASS'
        out['status']='PASS'if passed else 'RUNNING_PARTIAL'
    return out


def main():
    started=time.time();sys.path.insert(0,str(SOURCE))
    campaign=load(ROOT/'CAMPAIGN.json');arms={}
    for arm in ARMS:
        run=campaign['runs'][arm]
        q=audit(run['qualification'],'qualification',run.get('qualification_jobs',[]))
        f=audit(run['output'],'formal',run.get('jobs',[]))
        compare={'status':'PENDING'}
        if q.get('initial_native',{}).get('status')=='PASS'and f.get('initial_native',{}).get('status')=='PASS':
            equal=q['initial_native']['adapter_material_sha256']==f['initial_native']['adapter_material_sha256']
            compare={'status':'PASS'if equal else 'REVIEW_REQUIRED','actual_trainable_native0_q_vs_formal_bit_equal':equal,
                'qualification_native0_adapter_sha256':q['initial_native']['adapter_material_sha256'],
                'formal_native0_adapter_sha256':f['initial_native']['adapter_material_sha256'],
                'note':'Compare actual native0 trainable tensor bytes; qualified10 adapters/Adam are not formal initialization.'}
        arms[arm]={'campaign_status_at_snapshot':run.get('status'),'qualification':q,'formal':f,
            'actual_initial_trainable_comparison':compare}
        print(arm,'qualification',q['status'],'formal',f['status'],flush=True)
    reg=ROOT/'PARENT_REGISTRATION_v2.json';rv=load(reg);body=dict(rv);bodysha=body.pop('sha256')
    complete=all(r['qualification']['status']=='PASS'and r['formal']['status']=='PASS'for r in arms.values())
    output={'schema':'d46_actual_qualification_and_fresh_formal_proof_v1',
        'status':'COMPLETE'if complete else 'RUNNING_PARTIAL','generated_utc':datetime.now(timezone.utc).isoformat(),
        'audit_started_unix':started,'audit_finished_unix':time.time(),'builder':ref(__file__),
        'campaign_snapshot':{'path':str(ROOT/'CAMPAIGN.json'),'read_body_sha256':fp(campaign),
            'operations_source':campaign.get('operations_source'),'operations_source_sha':campaign.get('operations_source_sha')},
        'scientific_source':str(SOURCE),'scientific_source_git_sha':SCIENCE_SHA,
        'parent':{'model_sha256':PARENT_SHA,'registration':ref(reg),'registration_file_expected_sha256':REG_SHA,
            'registration_actual_matches':sha(reg)==REG_SHA,'body_seal_valid':bodysha==fp(body),
            'contract_revision':rv.get('contract_revision'),'IL_updates':rv.get('IL_updates'),
            'readouts_available':False,'parent_hash_recomputed_here':False,
            'model_hash_evidence':'parent_audit/AUDIT.json: previously actual-hashed pinned 8.76GB deployment; not rehashed by this bounded startup audit'},
        'counts':{'arms_registered':len(ARMS),'qualification10_all8_pass':sum(v['qualification']['status']=='PASS'for v in arms.values()),
            'formal_native0_all8_pass':sum(v['formal'].get('initial_native',{}).get('status')=='PASS'for v in arms.values()),
            'formal_actual_first_updates_all8_pass':sum(v['formal'].get('actual_metrics',{}).get('status')=='PASS'for v in arms.values())},
        'arms':arms,'boundaries_zh':[
            '仅记录本次读取时已经存在的真实结果；RUNNING_PARTIAL不等于其余路线失败。',
            'fresh证据来自实际native0空Adam/scheduler0/八rank零曝光与保存RNG、实际无resume命令和invocation；不以fresh_Adam注册字段代替。',
            'qualification内从native0恢复属于资格任务自身的精确恢复；正式任务另建fresh native0，没有继承qualification10或旧RL的Adam。',
            '冻结before/after哈希来自已执行资格任务；本CPU审计校验其全八rank记录和封存，不重新载入/哈希冻结2B模型。',
            'CUDA RNG仅只读其已保存的CPU ByteTensor；本审计无CUDA调用、forward、评分、GPU调度或worker变更。',
            'native Gaussian fresh ratio是启动行为诊断；FPO另核真正energy-proxy ratio，AWM实际优化matching且没有PPO clip。有限梯度不证明最终提分，也不比较异单位梯度。']}
    old=ROOT/'operations/USER_PRIORITY_20261103/D44_ACTUAL_NATIVE_SAVED.json'
    if old.is_file():output['previous_D44_saved_exit_evidence']=ref(old)
    output['sha256']=fp(output)
    path=ROOT/'qualification/ACTUAL_QUALIFICATION_AND_FRESH_FORMAL_PROOF.json';atomic(path,output)
    print(json.dumps({'path':str(path),'status':output['status'],'counts':output['counts'],'file_sha256':sha(path)}),flush=True)

if __name__=='__main__':main()
