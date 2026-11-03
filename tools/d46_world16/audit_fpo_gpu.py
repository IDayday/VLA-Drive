"""Read-only actual FPO16 GPU qualification audit, using its frozen validators.

The audit tool may be newer than the scientific source. It re-executes itself
with frozen science first on sys.path; no training, GPU, SSH, model load, state
mutation or source migration. Only an optional new audit JSON is published.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)

def exact(a,b,path='root'):
    """Compare complete RNG/state trees without administrative exclusions."""
    import numpy as np
    import torch
    require(type(a)is type(b),'Different type: '+path)
    if torch.is_tensor(a):require(a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b),'Different Tensor: '+path)
    elif isinstance(a,np.ndarray):require(a.dtype==b.dtype and np.array_equal(a,b),'Different NumPy: '+path)
    elif isinstance(a,dict):
        require(a.keys()==b.keys(),'Different keys: '+path)
        for k in a:exact(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(list,tuple)):
        require(len(a)==len(b),'Different length: '+path)
        for j,(x,y)in enumerate(zip(a,b)):exact(x,y,path+'.'+str(j))
    else:require(a==b,'Different value: '+path)


def arg(command,key):
    require(command.count(key)==1,'Expected one '+key)
    return command[command.index(key)+1]


def job_review(root,science,source_commit,qroot,resume,stop):
    rows=[]
    for node in range(2):
        job=Path(root)/f'node{node}';launch=read(job/'LAUNCH.json');result=read(job/'RESULT.json')
        command=launch['command']
        require(result.get('returncode')==0 and result.get('child_pid',0)>0,'Node did not actually exit0: '+str(job))
        require(result['command']==command and Path(result['cwd']).resolve()==science,'Node command/cwd changed: '+str(job))
        require(Path(launch['source']).resolve()==science and launch['source_sha']==source_commit,'Node scientific source changed')
        require('tools.d46_world16.train'in command and '--qualification'in command and '--nnodes=2'in command
            and '--nproc_per_node=8'in command and f'--node_rank={node}'in command,'Not exact two-node qualification entry')
        require(Path(arg(command,'--output')).resolve()==qroot and Path(arg(command,'--resume')).resolve()==resume,'Different qualification/native resume')
        require(int(arg(command,'--stop-after-update'))==stop,'Different finite update target')
        require(sorted(result['gpus'])==list(range(8))and len(set(result['gpu_uuids']))==8,'Node did not lease exactly8 physical GPUs')
        rows.append(dict(node=node,host=result['host'],job=str(job),returncode=0,child_pid=result['child_pid'],gpu_uuids=result['gpu_uuids'],launch_sha256=sha(job/'LAUNCH.json'),result_sha256=sha(job/'RESULT.json'),start_unix=result['start_unix'],end_unix=result['end_unix']))
    require(rows[0]['host']!=rows[1]['host'],'Two-node run must use distinct hosts')
    require(not(set(rows[0]['gpu_uuids'])&set(rows[1]['gpu_uuids'])),'Physical UUIDs overlap between nodes')
    return rows


UNCHANGED_COUNTERS=('collections','scene_cursor','valid_scene_exposures','raw_scene_exposures','new_candidates','valid_PG_candidates','extra_B1_queries','adaptive_G32_queries','padding_scene_exposures','evaluation_gates')


def fresh_review(losses):
    keys=('fpo_joint_logratio_min','fpo_joint_logratio_max','fpo_ratio_mean','fpo_ratio_max','fresh_ratio_error')
    require(all(k in losses and math.isfinite(losses[k])for k in keys),'Missing/nonfinite fresh FPO/Gaussian metrics')
    require(max(abs(losses[k])for k in keys[:2])<=1e-4,'Logged aggregate FPO energy logratio not fresh')
    require(max(abs(losses[k]-1)for k in keys[2:4])<=1e-4,'Logged aggregate FPO energy ratio not1')
    require(abs(losses['fresh_ratio_error'])<=1e-4,'Gaussian fresh diagnostic exceeded guard')
    return {k:losses[k]for k in keys}


def audit(args):
    import torch
    from tools.d46_world16.native_review import review
    from starVLA.rl.d46_world16.native import World16Boundary,FPO_PROTOCOL
    require(not torch.cuda.is_available(),'Audit must be CPU-only')
    science=Path(args.science_source).resolve();qroot=Path(args.qroot).resolve();u=args.origin_update
    require(u>=0 and u%2==0,'Origin must be the original READY/inner0 update')
    require(Path.cwd().resolve()==science,'Validator must execute from frozen scientific source')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=science,text=True).strip()
    require(head==args.expected_source_commit,'Frozen source commit mismatch')
    cps={name:qroot/'checkpoints'/f'update{step:06}_inner{inner}'for name,step,inner in [('origin',u,0),('pending',u+3,1),('ready',u+4,0)]}
    reviews={name:review(cp)for name,cp in cps.items()}
    identity=read(cps['origin']/'identity.json');identity_sha=identity['sha256']
    require(identity['qualification']is True and identity['world']==16 and identity['recipe']['arm']=='D3_FPO','Not FPO16 qualification')
    require((identity['recipe']['global_scenes'],identity['recipe']['group'],identity['recipe']['inner_updates'],identity['recipe']['mc_samples'],identity['recipe']['clip'])==(64,16,2,4,.01),'FPO recipe changed')
    require(identity['native_world_transition']['flow_energy_protocol']==FPO_PROTOCOL,'FPO energy protocol changed')
    require(identity['native_world_transition']['source_qualification']is False,'Qualification must originate from own formal8 policy')
    for cp in cps.values():exact(read(cp/'identity.json'),identity,'checkpoint_identity')
    for name,digest in identity['source_files'].items():require(sha(science/name)==digest,'Scientific file changed: '+name)
    first=job_review(args.first_job_root,science,head,qroot,cps['origin'],u+3)
    resumed=job_review(args.resume_job_root,science,head,qroot,cps['pending'],u+4)
    for a,b in zip(first,resumed):require(a['host']==b['host']and a['gpu_uuids']==b['gpu_uuids'],'Host/UUID layout changed between legs')
    common=torch.load(cps['ready']/'optimizer.pt',map_location='cpu',weights_only=False)
    require(common['scheduler']['last_epoch']==u+4,'Scheduler count changed')
    steps=[int(v['step'])for v in common['optimizer']['state'].values()]
    require(bool(steps)and set(steps)=={u+4},'Adam states do not all match final update')
    require(set(common['adapters'])==set(identity['scope']['trainable_names']),'Trainable closure changed')
    require(all(x.dtype==torch.float32 and torch.isfinite(x).all()for x in common['adapters'].values()),'Invalid current trainable bytes')
    require(sha(cps['origin']/'optimizer.pt')==identity['native_world_transition']['common_optimizer_sha256'],'Original Adam/weights not byte-identical at transition')
    ranks=[];frozen_hashes=[];gradient_by_update={step:[]for step in range(u+1,u+5)}
    for rank in range(16):
        own={name:torch.load(cp/f'rank{rank}.pt',map_location='cpu',weights_only=False)for name,cp in cps.items()}
        before=own['pending'];after=own['ready'];start=own['origin']
        require(set(before['rng'])=={'python','numpy','torch','cuda','rollout_generator'},'Incomplete RNG bundle')
        exact(before['rng'],after['rng'],f'rank{rank}.full_RNG')
        exact(before['state']['world16_virtual_sampler'],after['state']['world16_virtual_sampler'],f'rank{rank}.virtual8')
        for key in UNCHANGED_COUNTERS:exact(before['state'][key],after['state'][key],f'rank{rank}.{key}')
        require(after['rollout']is None and before['rollout']is not None,'Pending/READY material phase changed')
        require(before['state']['new_candidates']-start['state']['new_candidates']==2048,'First leg must generate exactly two B64G16 collections')
        require(before['state']['collections']-start['state']['collections']==2 and before['state']['raw_scene_exposures']-start['state']['raw_scene_exposures']==128,'First leg collection/scene count changed')
        batch=before['rollout'];require(len(batch['scenes'])==4,'Expected4 distinct scene leaves/rank')
        World16Boundary.validate_boundary(before['state'],batch,identity)
        fm=[]
        for scene in batch['scenes']:
            fm.append({key:list(scene[key].shape)for key in('fm_state','fm_sigma','fm_target','old_energy')})
        parity=[]
        for step in(u,u+3):
            path=qroot/f'native_B1_restore_parity_rank{rank}_u{step:06}.json';p=read(path)
            require(p['status']=='PASS'and p['rank']==rank and p['updates']==step and p['single_B1']is True and p['seed']==2101 and p['steps']==10 and p['shift']==5.,'B1 restore protocol changed')
            require(p['max_abs']=={'normalized_action':0.,'requested_trajectory':0.},'B1 restore mismatch')
            parity.append(dict(path=str(path),sha256=sha(path),max_abs=p['max_abs']))
        terminal=read(qroot/f'result_rank{rank}.json')
        require(terminal['qualification']is True and terminal['updates']==u+4 and terminal['rank']==rank and terminal['status']=='QUALIFICATION_STOPPED','Final qualification result changed')
        require(terminal.get('frozen_before')and terminal['frozen_before']==terminal.get('frozen_after'),'Frozen parameter mutation')
        frozen_hashes.append(terminal['frozen_before'])
        records=[json.loads(line)for line in(qroot/f'metrics_rank{rank}.jsonl').read_text().splitlines()if line]
        require([r['updates']for r in records]==list(range(u+1,u+5)),'Missing/duplicate/extra qualification Adam rows')
        fresh=[]
        for row in records:
            require(row['run']==identity_sha and row['active_batch']=={'scenes':64,'group':16,'stage':None},'Journal identity/batch changed')
            require(math.isfinite(row['gradient_norm'])and row['gradient_norm']>=0,'Nonfinite gradient')
            gradient_by_update[row['updates']].append(row['gradient_norm'])
            require(all(v is None or math.isfinite(v)for v in row['role_gradient_norms'].values()),'Nonfinite role gradient')
            require(all(math.isfinite(v)for v in row['losses'].values()),'Nonfinite loss/ratio')
            if row['updates']in(u+1,u+3):fresh.append(dict(updates=row['updates'],metrics=fresh_review(row['losses'])))
        # Invocation counters measure real generation, independently of state rollback.
        inv={}
        for path in(qroot/'invocations').glob(f'*/rank{rank}.json'):
            value=read(path)
            if value['run']==identity_sha:inv[value['start_update']]=(path,value)
        require(u in inv and u+3 in inv,'Missing actual leg invocations')
        for start_u,end_u in[(u,u+3),(u+3,u+4)]:
            v=inv[start_u][1];require(v['status']=='QUALIFICATION_STOPPED'and v['end_update']==end_u,'Invocation not completed at exact boundary')
        resume_cost=inv[u+3][1]
        for key in('physical_candidates','candidate_requests','scorer_candidate_attempts'):
            require(resume_cost.get(key,0)==0,'Pending resume performed new query: '+key)
        require(inv[u][1]['physical_candidates']==128,'First-leg actual generated candidate count not128/rank')
        ranks.append(dict(rank=rank,full_RNG_exact=True,virtual8_exact=True,unchanged_counters={k:after['state'][k]for k in UNCHANGED_COUNTERS},FM_shapes=fm,B1_restore=parity,frozen_parameters_unchanged=True,fresh_aggregated_FPO_and_Gaussian=fresh,resume_physical_queries=0,actual_first_leg_candidates=128,invocations=[dict(path=str(inv[k][0]),sha256=sha(inv[k][0]))for k in(u,u+3)]))
    require(len(set(frozen_hashes))==1,'Ranks differ in frozen policy bytes')
    require(all(max(values)-min(values)<=1e-8*max(1.,max(values))for values in gradient_by_update.values()),'DDP gradient norms differ across ranks')
    return dict(schema='d46_actual_FPO_world16_GPU_qualification_audit_v1',status='PASS',qualification=True,qweights_may_initialize_formal=False,origin_updates=u,final_updates=u+4,world=16,B=64,G=16,M=2,MC=4,scientific_source=str(science),scientific_commit=head,auditor_path=str(Path(__file__).resolve()),auditor_sha256=sha(__file__),checkpoints=reviews,node_results={'first':first,'resume':resumed},Adam_steps=u+4,scheduler_steps=u+4,first_leg_new_PG_candidates=2048,resume_new_queries=0,gradient_norms_by_update=gradient_by_update,ranks=ranks,limits=['Actual full-model CUDA execution is evidenced by the bound node receipts/checkpoints; this audit itself is CPU only.','FPO fresh ratios/energy extrema are per-rank averages of4 leaf metrics in existing logs, not a saved per-leaf maximum. Gaussian fresh ratio has its separate actor hard guard.','No uninterrupted full-model parameter-bitexact comparator was run; pending RNG/material/Adam continuity is checked, not crossworld bitexactness.','No long-run speed, score gain or algorithm superiority claim.'])


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--qroot',required=True);p.add_argument('--first-job-root',required=True);p.add_argument('--resume-job-root',required=True);p.add_argument('--origin-update',type=int,required=True);p.add_argument('--expected-source-commit',required=True);p.add_argument('--science-source');p.add_argument('--output');p.add_argument('--_in-frozen-source',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
    if not a._in_frozen_source:
        science=Path(a.science_source or read(Path(a.first_job_root)/'node0/LAUNCH.json')['source']).resolve()
        environment=os.environ.copy();environment.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',WORLD_COMPARE_ROOT='/mnt/project/d46-priority-rl-20261103',WORLD_COMPARE_TASK='d46_priority_rl_20261103',D_NATIVE_CAMPAIGN_ROOT='/mnt/project/d46-priority-rl-20261103')
        environment.pop('PYTHONPATH',None)
        bootstrap="import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_path(sys.argv.pop(1),run_name='__main__')"
        command=[sys.executable,'-c',bootstrap,str(science),str(Path(__file__).resolve())]+sys.argv[1:]+['--_in-frozen-source']
        if a.science_source is None:command+=['--science-source',str(science)]
        raise SystemExit(subprocess.run(command,cwd=science,env=environment,timeout=600).returncode)
    try:value=audit(a)
    except Exception as error:
        import traceback
        value=dict(schema='d46_actual_FPO_world16_GPU_qualification_audit_v1',status='PENDING'if isinstance(error,FileNotFoundError)else'FAIL',error=repr(error),traceback=traceback.format_exc(),qualification=True,qweights_may_initialize_formal=False)
    encoded=json.dumps(value,indent=2,allow_nan=False)+'\n'
    if a.output:
        with Path(a.output).open('x')as stream:stream.write(encoded)
    print(encoded,end='')
    if value['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
