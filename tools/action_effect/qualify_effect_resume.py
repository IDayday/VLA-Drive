"""Real G1 16 vs8+8 qualification using the existing trainer and actual pairs.

Exactly 32 scheduled optimizer updates maximum across two independent runs. The
original 1000-update scheduler/warmup is retained; qualification checkpoints at 9
allow comparison of the first update after the step 8 interruption. No simulator,
VLM, common trainer or collector implementation is replaced.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import time
import numpy as np
import torch

from .common import artifact,digest,load_config,lock,read_json,verify_artifact,write_json

TOTAL_PER_RUN=16
SPLIT_AT=8
FIRST_RESUMED_UPDATE=9
SCHEDULER_HORIZON=1000
FIELDS=('model','optimizer','scheduler','sampler','stateful','rng','consumed_scenes',
        'optimizer_step','run_identity','termination_reason','terminal')


def compare_exact(left,right,prefix=''):
    """Complete byte comparisons, including Adam and signed floating zero."""
    bad=[]
    def visit(a,b,path):
        if type(a) is not type(b):bad.append(path+':type');return
        if isinstance(a,torch.Tensor):
            if a.dtype!=b.dtype or a.shape!=b.shape:bad.append(path+':tensor_schema');return
            if a.is_floating_point() and (not torch.isfinite(a).all() or not torch.isfinite(b).all()):
                bad.append(path+':nonfinite');return
            if not torch.equal(a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                               b.detach().cpu().contiguous().reshape(-1).view(torch.uint8)):bad.append(path+':tensor_bytes')
        elif isinstance(a,np.ndarray):
            if np.issubdtype(a.dtype,np.floating) and (not np.isfinite(a).all() or not np.isfinite(b).all()):
                bad.append(path+':nonfinite');return
            if a.dtype!=b.dtype or a.shape!=b.shape or a.tobytes()!=b.tobytes():bad.append(path+':numpy_bytes')
        elif isinstance(a,dict):
            if a.keys()!=b.keys():bad.append(path+':keys');return
            for k in a:visit(a[k],b[k],path+'/'+str(k))
        elif isinstance(a,(tuple,list)):
            if len(a)!=len(b):bad.append(path+':length');return
            for i,(x,y) in enumerate(zip(a,b)):visit(x,y,path+'/'+str(i))
        elif isinstance(a,float):
            if not math.isfinite(a) or not math.isfinite(b) or struct.pack('!d',a)!=struct.pack('!d',b):bad.append(path+':float_bytes')
        elif a!=b:bad.append(path+':value')
    visit(left,right,prefix)
    return bad


def state_hash(value):
    h=hashlib.sha256()
    def add(x):
        h.update(type(x).__name__.encode()+b':')
        if isinstance(x,torch.Tensor):
            h.update(str(x.dtype).encode());h.update(str(list(x.shape)).encode())
            h.update(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,np.ndarray):h.update(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes())
        elif isinstance(x,dict):
            for key in sorted(x,key=lambda k:(type(k).__name__,str(k))):add(key);add(x[key])
        elif isinstance(x,(tuple,list)):
            for item in x:add(item)
        elif isinstance(x,float):h.update(struct.pack('!d',x))
        else:h.update(repr(x).encode())
    add(value);return h.hexdigest()


def checkpoints(run):
    run=Path(run)
    if not (run/'IDENTITY.json').is_file():return []
    ih=digest(read_json(run/'IDENTITY.json'));valid=[]
    for receipt_path in sorted(run.glob('step_*.COMPLETE.json')):
        try:
            receipt=read_json(receipt_path);path=verify_artifact(receipt['checkpoint'])
            state=torch.load(path,map_location='cpu',weights_only=False)
            if receipt['run_identity']!=ih or state['run_identity']!=ih or state['optimizer_step']!=receipt['optimizer_step']:
                continue
            valid.append((state['optimizer_step'],receipt.get('created_unix_ns',0),path,state,receipt_path))
        except Exception:continue
    return sorted(valid,key=lambda x:(x[0],x[1],str(x[2])))


def at_step(run,step):
    options=[x for x in checkpoints(run) if x[0]==step]
    if not options:raise ValueError('Missing complete qualification checkpoint at step '+str(step)+' in '+str(run))
    return options[-1]


def compare_checkpoint(left,right):
    components={};mismatches=[]
    for name in FIELDS:
        if name not in left or name not in right:raise ValueError('Incomplete real training state: '+name)
        wrong=compare_exact(left[name],right[name],name);mismatches.extend(wrong)
        components[name]={'exact':not wrong,'left_sha256':state_hash(left[name]),'right_sha256':state_hash(right[name])}
    # History wall times are intentionally excluded, but actual update outputs
    # and each stream's final state must match exactly.
    for step in (FIRST_RESUMED_UPDATE,TOTAL_PER_RUN):
        if step>min(left['optimizer_step'],right['optimizer_step']):continue
        a=next((x for x in left['history'] if x['optimizer_step']==step),None)
        b=next((x for x in right['history'] if x['optimizer_step']==step),None)
        if a is None or b is None:raise ValueError('Missing actual update history at step '+str(step))
        for name in ('losses','gradient_norm','lr','consumed_scenes'):
            mismatches.extend(compare_exact(a[name],b[name],f'update_{step}/{name}'))
    return {'exact':not mismatches,'components':components,'mismatch_count':len(mismatches),'mismatches':mismatches[:64]}


def draw_batch(data,sampler,pair_rng,batch_size):
    indices=torch.randint(len(data.scenes),(batch_size,),generator=sampler).tolist();out=[]
    for index in indices:
        scene=data.scenes[index];choices=data.by_scene[scene]
        pair=choices[int(torch.randint(len(choices),(1,),generator=pair_rng))]
        out.append({'scene_id':scene,'pair_identity':pair['identity']})
    return out


def next_batch_proof(data,split_state8,continuous_state9,resumed_state9,*,seed,batch_size):
    sampler=torch.Generator().manual_seed(seed+101);pair_rng=torch.Generator().manual_seed(seed+202)
    for _ in range(SPLIT_AT):draw_batch(data,sampler,pair_rng,batch_size)
    before={'sampler':sampler.get_state(),'pair_rng':pair_rng.get_state()}
    boundary={'sampler':split_state8['sampler'],'pair_rng':split_state8['stateful']['pair_rng']}
    errors=compare_exact(before,boundary,'boundary8')
    expected=draw_batch(data,sampler,pair_rng,batch_size)
    restored_sampler=torch.Generator();restored_sampler.set_state(split_state8['sampler'])
    restored_pair=torch.Generator();restored_pair.set_state(split_state8['stateful']['pair_rng'])
    actual=draw_batch(data,restored_sampler,restored_pair,batch_size)
    errors.extend(compare_exact(expected,actual,'next_batch9'))
    after={'sampler':sampler.get_state(),'pair_rng':pair_rng.get_state()}
    for name,state in [('continuous9',continuous_state9),('resumed9',resumed_state9)]:
        errors.extend(compare_exact(after,{'sampler':state['sampler'],'pair_rng':state['stateful']['pair_rng']},name))
    return {'exact':not errors,'scene_pair_ids':expected,'scene_pair_ids_sha256':digest(expected),
        'actual_checkpoint_after_next_update':FIRST_RESUMED_UPDATE,'mismatches':errors}


def run_segment(cfg_path,args,run,target,*,resume_requested,output):
    progress=checkpoints(run)
    if progress and progress[-1][0]>TOTAL_PER_RUN:raise ValueError('Qualification output has exceeded the total update budget')
    complete=[x for x in progress if x[0]==target]
    if complete:
        return {'status':'REUSED_COMPLETE','target_step':target,'checkpoint':artifact(complete[-1][2]),'new_updates':0}
    current=progress[-1][0] if progress else 0
    if current and not resume_requested:raise ValueError('Partial qualification exists; use --resume')
    if current>=target:raise ValueError('Missing requested historical boundary; never roll back current progress')
    if str(args.device).startswith('cuda') and torch.cuda.device_count()!=1:
        raise ValueError('Expose exactly one admitted GPU with CUDA_VISIBLE_DEVICES before qualification')
    cmd=[sys.executable,'-m','tools.action_effect.train_effect_model','--config',str(cfg_path),
        '--context-manifest',str(args.context_manifest),'--pair-dataset',str(args.pair_dataset),
        '--arm',args.arm,'--device',args.device,'--seed',str(args.seed),'--max-updates',str(SCHEDULER_HORIZON),
        '--stop-after',str(target-current),'--output',str(run)]+(['--resume'] if current else [])
    # Reserve before launching: a failed child may have performed work after its
    # latest durable checkpoint. Never silently spend those updates a second time.
    budget_path=Path(output)/'DISPATCH_BUDGET.json'
    budget=read_json(budget_path) if budget_path.exists() else {'scheduled_updates':0,'reservations':[]}
    if budget['scheduled_updates']+target-current>2*TOTAL_PER_RUN:
        raise ValueError('32-update qualification dispatch budget exhausted; preserve progress and request a new explicit budget')
    budget['scheduled_updates']+=target-current
    budget['reservations'].append({'run':str(run),'from_step':current,'target_step':target,'new_update_ceiling':target-current,'command':cmd})
    write_json(budget_path,budget,immutable=False)
    stamp=time.time_ns();stem=Path(output)/f'{Path(run).name}_target{target:02d}_{stamp}'
    with stem.with_suffix('.stdout.log').open('w') as stdout,stem.with_suffix('.stderr.log').open('w') as stderr:
        result=subprocess.run(cmd,cwd=Path(__file__).resolve().parents[2],stdout=stdout,stderr=stderr,
            env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    receipt={'command':cmd,'returncode':result.returncode,'stdout':artifact(stem.with_suffix('.stdout.log')),
        'stderr':artifact(stem.with_suffix('.stderr.log')),'starting_step':current,'target_step':target}
    write_json(stem.with_suffix('.COMMAND.json'),receipt)
    if result.returncode:raise RuntimeError('Real qualification training child failed; full logs preserved')
    completed=at_step(run,target)
    return {**receipt,'status':'COMPLETE','checkpoint':artifact(completed[2]),'new_updates':target-current}


def reuse_complete(output):
    """Verify completed evidence without rewriting its immutable result pin."""
    done=read_json(Path(output)/'COMPLETE.json')
    verify_artifact(done['identity']);result=read_json(verify_artifact(done['result']))
    if done['status']!='PASS' or result['status']!='PASS' or not result['exact_resume_this_runtime']:
        raise ValueError('Invalid qualification completion')
    for pin in result['checkpoints'].values():verify_artifact(pin)
    for pin in (result['identity'],result['qualification_config']):verify_artifact(pin)
    for entry in result['commands']:
        for key in ('stdout','stderr','checkpoint'):
            if key in entry:verify_artifact(entry[key])
    return {**result,'reused_complete':True,'new_updates_this_invocation':0}


def qualify(args):
    cfg=load_config(args.config)
    if cfg['training']['max_updates']!=SCHEDULER_HORIZON:
        raise ValueError('Qualification must preserve the actual1000-update scheduler horizon')
    if args.seed not in cfg['training']['training_seeds']+cfg['training']['finalist_seeds']:raise ValueError('Unregistered seed')
    output=Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:raise ValueError('Existing evidence requires --resume; never overwrite it')
    output.mkdir(parents=True,exist_ok=True)
    with lock(output/'QUALIFICATION.lock'):
        identity={'schema':'action_effect_real_resume_qualification_v1','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip(),
            'original_config':artifact(args.config),'context':artifact(args.context_manifest),'pairs':artifact(args.pair_dataset),
            'seed':args.seed,'arm':args.arm,'device':args.device,'scheduler_horizon':SCHEDULER_HORIZON,
            'continuous_updates':TOTAL_PER_RUN,'split_updates':[SPLIT_AT,TOTAL_PER_RUN-SPLIT_AT],
            'total_new_update_ceiling':2*TOTAL_PER_RUN,'qualification_save_validation_every':FIRST_RESUMED_UPDATE,
            'production_recipe_changed':False,'Base_changed':False}
        write_json(output/'IDENTITY.json',identity)
        qcfg=deepcopy(cfg);qcfg['training']['validation_every']=FIRST_RESUMED_UPDATE
        config=output/'QUALIFICATION_CONFIG.json';write_json(config,qcfg)
        if args.dry_run:return {'status':'NOT_RUN','identity':identity,'new_updates':0,'new_simulation_calls':0}
        if (output/'COMPLETE.json').exists():return reuse_complete(output)
        # Real dataset contract is checked before any GPU training dispatch.
        from .context_data import ContextReader
        from .train_effect_model import PairData
        reader=ContextReader(args.context_manifest,Path(cfg['artifact_root'])/'CAMPAIGN_BINDING.json')
        data=PairData(args.pair_dataset,reader)
        commands=[];full=output/'continuous16';split=output/'split8plus8'
        for run,target in ((full,TOTAL_PER_RUN),(split,SPLIT_AT),(split,TOTAL_PER_RUN)):
            commands.append(run_segment(config,args,run,target,resume_requested=args.resume or bool(commands),output=output))
        f9=at_step(full,FIRST_RESUMED_UPDATE);r9=at_step(split,FIRST_RESUMED_UPDATE)
        f16=at_step(full,TOTAL_PER_RUN);r16=at_step(split,TOTAL_PER_RUN);r8=at_step(split,SPLIT_AT)
        comparisons={'first_resumed_update9':compare_checkpoint(f9[3],r9[3]),'final_update16':compare_checkpoint(f16[3],r16[3])}
        next_ids=next_batch_proof(data,r8[3],f9[3],r9[3],seed=args.seed,batch_size=qcfg['training']['batch_scenes'])
        exact=all(r['exact'] for r in comparisons.values()) and next_ids['exact']
        result={'schema':'action_effect_real_resume_result_v1','status':'PASS' if exact else 'FAIL',
            'identity':artifact(output/'IDENTITY.json'),'qualification_config':artifact(config),'exact_resume_this_runtime':exact,
            'checkpoints':{'continuous9':artifact(f9[2]),'resumed9':artifact(r9[2]),'continuous16':artifact(f16[2]),'resumed16':artifact(r16[2]),'split8':artifact(r8[2])},
            'comparisons':comparisons,'next_batch':next_ids,'commands':commands,
            'new_updates_this_invocation':sum(r['new_updates'] for r in commands),'total_completed_updates':2*TOTAL_PER_RUN,
            'dispatch_budget':read_json(output/'DISPATCH_BUDGET.json'),
            'fit_unique_scenes':len(data.scenes),'actual_batch_scenes':qcfg['training']['batch_scenes'],
            'runtime':read_json(full/'IDENTITY.json')['runtime'],'new_VLM_calls':0,'new_simulation_calls':0,
            'scientific_gain':'UNTESTED: this is engineering continuation qualification only'}
        write_json(output/'RESULT.json',result,immutable=False)
        write_json(output/f'INVOCATION_{time.time_ns()}.json',result)
        if exact:write_json(output/'COMPLETE.json',{'status':'PASS','identity':artifact(output/'IDENTITY.json'),
            'result':artifact(output/'RESULT.json'),'checkpoint_comparisons':{k:v['exact'] for k,v in comparisons.items()},'next_ids_exact':next_ids['exact']})
        return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--context-manifest',required=True);p.add_argument('--pair-dataset',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--arm',default='delta_feedback',choices=('absolute_score','terminal_pdms','absolute_feedback','delta_feedback','trajectory_only','shuffled_context'))
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    try:
        result=qualify(a);print(json.dumps(result));return 0 if result['status'] in ('PASS','NOT_RUN') else 1
    except Exception as exc:
        print(json.dumps({'status':'FAIL','error_type':type(exc).__name__,'error':str(exc),'exact_resume':False}));return 1


if __name__=='__main__':raise SystemExit(main())
