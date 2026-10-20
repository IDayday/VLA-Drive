"""Real original-head B0 CUDA16 versus8+8 exact-resume qualification.

Reuse train_effect_flow and the shared trainer; never run VLM or a simulator.
The original1000 scheduler/warm50/batch32/micro8 is retained. Saving at9 is the
only recipe change. At most32 new optimizer updates are reserved before launch.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

import torch

from .audit_effect_resume import exact_hash
from .common import artifact,digest,load_config,lock,read_json,verify_artifact,write_json
from .qualify_effect_resume import at_step,checkpoints,reuse_complete,validate_training_identity

TOTAL=16
BOUNDARY=8
FIRST=9
HORIZON=1000
FIELDS=('model','optimizer','scheduler','sampler','stateful','rng','consumed_scenes',
        'optimizer_step','run_identity','termination_reason','terminal')
IDLE_AUX=('adapter.effect_projection.','adapter.effect_head.')


def validate_recipe(cfg):
    t=cfg['training']
    if any(t[k]!=v for k,v in {'max_updates':1000,'warmup_updates':50,'batch_scenes':32,'microbatch_scenes':8}.items()):
        raise ValueError('Keep original1000 scheduler, warm50, batch32 and micro8')
    q=deepcopy(cfg);q['training']['validation_every']=FIRST
    return q


def validate_state(state, identity, step, private_order=None):
    """Adam IDs follow actual private insertion order; dormant B0 aux may lack state."""
    if (state.get('schema')!='action_effect_private_checkpoint_v1'
        or state.get('run_identity')!=digest(identity) or state.get('optimizer_step')!=step
        or state.get('terminal') is not False or state.get('termination_reason') not in ('OPERATOR_PAUSED','RUNNING_CHECKPOINT')):
        raise ValueError('Wrong real B0 checkpoint/identity/boundary')
    model=state['model'];names=list(model)
    if private_order is not None and names!=private_order:
        raise ValueError('Private parameter insertion order differs from reconstructed adapter names')
    if sorted(names)!=identity['private_names'] or state['consumed_scenes']!=step*identity['recipe']['batch_scenes']:
        raise ValueError('Private names or consumed sampler cursor changed')
    if [r['optimizer_step'] for r in state['history']]!=list(range(1,step+1)):
        raise ValueError('Missing or duplicate update history')
    if set(state['stateful'])!={'flow_noise','flow_time'}:
        raise ValueError('Both separate Flow RNG streams must be saved')
    if not {'python','numpy','cpu','cuda'}<=set(state['rng']) or not state['rng']['cuda']:
        raise ValueError('Actual Python/NumPy/CPU/CUDA training RNG evidence required')
    params=[i for group in state['optimizer']['param_groups'] for i in group['params']]
    if params!=list(range(len(names))) or not set(state['optimizer']['state'])<=set(params):
        raise ValueError('Adam parameter-ID mapping differs from stable private names')
    absent=[];active=[]
    for i,name in enumerate(names):
        tensor=model[name]
        if tensor.dtype!=torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError('Private parameter precision/nonfinite failure: '+name)
        moment=state['optimizer']['state'].get(i)
        if moment is None:
            if not name.startswith(IDLE_AUX):raise ValueError('Missing active velocity Adam state: '+name)
            absent.append(name);continue
        for key in ('exp_avg','exp_avg_sq'):
            value=moment[key]
            if value.dtype!=torch.float32 or value.shape!=tensor.shape or not torch.isfinite(value).all():
                raise ValueError('Wrong Adam moment/name/shape/precision: '+name)
        if int(moment['step'])!=step:raise ValueError('Adam step does not match this boundary: '+name)
        active.append(name)
    # Hashing recursively rejects nonfinite states, preserving signed zero and
    # exact tensor/scalar types without assuming every allocated head is used.
    for field in FIELDS:exact_hash(state[field])
    return {'active_Adam_parameter_names':active,'dormant_aux_without_Adam_names':absent,
        'parameter_order_sha256':digest(names),'private_parameters':len(names)}


def compare_states(left,right,step):
    comparisons={}
    for name in FIELDS:
        a,b=exact_hash(left[name]),exact_hash(right[name]);comparisons[name]={'exact':a==b,'left_sha256':a,'right_sha256':b}
    for boundary in (FIRST,TOTAL):
        if boundary>step:continue
        for key in ('losses','gradient_norm','lr','consumed_scenes'):
            a=exact_hash(left['history'][boundary-1][key]);b=exact_hash(right['history'][boundary-1][key])
            comparisons[f'update{boundary}/{key}']={'exact':a==b,'left_sha256':a,'right_sha256':b}
    return {'exact':all(r['exact'] for r in comparisons.values()),'components':comparisons}


def next_scene_proof(pool,state8,continuous9,resumed9,*,seed,batch_size):
    sampler=torch.Generator().manual_seed(seed+101)
    def draw(generator):
        indices=torch.randint(len(pool),(batch_size,),generator=generator).tolist()
        return [{'reference_index':i,'scene_id':pool[i]['scene_id'],'target_kind':pool[i]['target_kind']} for i in indices]
    for _ in range(BOUNDARY):draw(sampler)
    boundary_exact=exact_hash(sampler.get_state())==exact_hash(state8['sampler'])
    expected=draw(sampler)
    restored=torch.Generator();restored.set_state(state8['sampler']);actual=draw(restored)
    after=exact_hash(sampler.get_state())
    states={name:after==exact_hash(s['sampler']) for name,s in [('continuous9',continuous9),('resumed9',resumed9)]}
    return {'exact':boundary_exact and exact_hash(expected)==exact_hash(actual) and all(states.values()),
        'boundary8_exact':boundary_exact,'next9_scene_ids':expected,'next9_scene_ids_sha256':digest(expected),
        'after_update9_sampler_exact':states,'pool_unique_scenes':len({r['scene_id'] for r in pool}),
        'virtual_reference_count':len(pool),'replacement':True}


def validate_run(run,expected,head,device=None):
    validate_training_identity(run,expected)
    path=Path(run)/'IDENTITY.json'
    if not path.exists():return
    identity=read_json(path);actual=identity['head']
    if any(actual.get(k)!=v for k,v in head.items()):
        raise ValueError('Canonical original head receipt/strict-load provenance changed')
    runtime=identity['runtime']
    if (device is not None and runtime.get('device')!=device) or any(runtime.get(k)!=v for k,v in
            {'trainable_dtype':'float32','autocast':'disabled','deterministic':True,'matmul_tf32':False,
             'cudnn_tf32':False,'cudnn_benchmark':False}.items()):
        raise ValueError('Actual qualification training runtime changed')


def boundary(run,step):
    row=at_step(run,step);path,state,receipt_path=row[2:]
    receipt=read_json(receipt_path)
    if (path.resolve().parent!=Path(run).resolve() or receipt_path.resolve()!=path.with_suffix('.COMPLETE.json').resolve()
        or receipt.get('terminal')!=state.get('terminal')
        or receipt.get('termination_reason')!=state.get('termination_reason')):
        raise ValueError('Checkpoint/receipt namespace or lifecycle differs')
    return row


def segment(config,args,run,target,*,expected,head):
    validate_run(run,expected,head,args.device)
    progress=checkpoints(run)
    if progress and progress[-1][0]>TOTAL:raise ValueError('Qualification already exceeded16 steps')
    complete=[r for r in progress if r[0]==target]
    if complete:return {'status':'REUSED_COMPLETE','target_step':target,'checkpoint':artifact(boundary(run,target)[2]),'new_updates':0}
    current=progress[-1][0] if progress else 0
    if current and not args.resume:raise ValueError('Partial qualification requires --resume')
    if current>=target:raise ValueError('Missing historical boundary; never rewind trained progress')
    cmd=[sys.executable,'-m','tools.action_effect.train_effect_flow','--config',str(config),
        '--context-manifest',str(Path(args.context_manifest).resolve()),'--repair-dataset',str(Path(args.repair_dataset).resolve()),
        '--arm','B0','--device',args.device,'--seed',str(args.seed),'--max-updates','1000',
        '--stop-after',str(target-current),'--output',str(run)]+(['--resume'] if current else [])
    budget_path=Path(args.output)/'DISPATCH_BUDGET.json'
    budget=read_json(budget_path) if budget_path.exists() else {'scheduled_updates':0,'reservations':[]}
    if (type(budget.get('scheduled_updates')) is not int or not 0<=budget['scheduled_updates']<=32
        or any(type(r.get('new_update_ceiling')) is not int or r['new_update_ceiling']<=0 for r in budget['reservations'])
        or sum(r['new_update_ceiling'] for r in budget['reservations'])!=budget['scheduled_updates']):
        raise ValueError('Malformed reservation ledger; cannot safely dispatch')
    if budget['scheduled_updates']+target-current>2*TOTAL:
        raise ValueError('32-update dispatch budget exhausted; do not silently repeat failed work')
    budget['scheduled_updates']+=target-current
    budget['reservations'].append({'run':str(run),'from_step':current,'target_step':target,
        'new_update_ceiling':target-current,'command':cmd})
    write_json(budget_path,budget,immutable=False)
    log=Path(args.output)/f'{run.name}_target{target:02d}_{time.time_ns()}'
    with log.with_suffix('.stdout.log').open('w') as stdout,log.with_suffix('.stderr.log').open('w') as stderr:
        done=subprocess.run(cmd,cwd=Path(__file__).resolve().parents[2],stdout=stdout,stderr=stderr,timeout=args.max_seconds_per_segment,
            env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    result={'command':cmd,'returncode':done.returncode,'stdout':artifact(log.with_suffix('.stdout.log')),
        'stderr':artifact(log.with_suffix('.stderr.log')),'starting_step':current,'target_step':target,'new_updates':target-current}
    write_json(log.with_suffix('.COMMAND.json'),result)
    if done.returncode:raise RuntimeError('Real B0 qualification child failed; preserve logs and reservation')
    validate_run(run,expected,head,args.device)
    result.update(status='COMPLETE',checkpoint=artifact(boundary(run,target)[2]));return result


def qualify(args):
    cfg=load_config(args.config);qcfg=validate_recipe(cfg)
    if args.seed!=42 or args.arm!='B0':raise ValueError('Only registered primary-seed42 B0 qualification is budgeted')
    if not args.device.startswith('cuda:') or not 1<=args.max_seconds_per_segment<=3600:
        raise ValueError('Actual admitted CUDA device and bounded per-child wall limit required')
    output=Path(args.output).resolve();args.output=str(output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError('Existing evidence requires --resume; never overwrite it')
    output.mkdir(parents=True,exist_ok=True)
    with lock(output/'QUALIFICATION.lock'):
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip()
        identity={'schema':'action_effect_flow_real_resume_qualification_v1','source_commit':source,
            'original_config':artifact(args.config),'context':artifact(args.context_manifest),'repairs':artifact(args.repair_dataset),
            'arm':'B0','seed':args.seed,'device':args.device,'scheduler_horizon':1000,'continuous_updates':16,
            'split_updates':[8,8],'total_new_update_ceiling':32,'qualification_save_validation_every':9,
            'production_recipe_changed':False,'Base_changed':False,'max_seconds_per_segment':args.max_seconds_per_segment}
        write_json(output/'IDENTITY.json',identity);config=output/'QUALIFICATION_CONFIG.json';write_json(config,qcfg)
        if args.dry_run:return {'status':'NOT_RUN','identity':identity,'new_updates':0,'new_simulation_calls':0}
        if torch.cuda.device_count()!=1:raise ValueError('Expose exactly one Lead-admitted GPU')
        from .context_data import ContextReader
        from .train_effect_flow import load_flow_examples,balance_examples,FlowLossRecipe
        from starVLA.model.modules.action_effect.effect_flow_adapter import EffectFlowAdapter,EffectFlowAdapterConfig
        reader=ContextReader(args.context_manifest,Path(cfg['artifact_root'])/'CAMPAIGN_BINDING.json')
        if len(reader.rows)!=64:raise ValueError('Use the original fixed64 context qualification population')
        examples,data_identity=load_flow_examples(args.repair_dataset,reader);pool,sampling=balance_examples(examples)
        t=qcfg['training'];head_receipt=read_json(verify_artifact(reader.manifest['head_receipt']))
        head={'head_receipt':reader.manifest['head_receipt'],'head_artifact':head_receipt['artifact'],
            'base_binding_sha256':reader.binding['base_checkpoint_binding_hash'],'strict_load':True,'Qwen_instantiated':False}
        adapter_config=EffectFlowAdapterConfig(condition_dim=2048,hidden_dim=t['hidden'],layers=t['layers'],
            heads=t['heads'],init_seed=42,effect_dim=len(data_identity['label_spec']['names']))
        small_adapter=EffectFlowAdapter(adapter_config)
        private_order=['adapter.'+name for name,_ in small_adapter.named_parameters()]
        del small_adapter
        expected={'source_commit':source,'arm':'B0','seed':42,'max_updates':1000,'recipe':t,'dataset_size':len(pool),
            'base_binding_hash':reader.binding['base_checkpoint_binding_hash'],'context_artifact':artifact(args.context_manifest),
            'repair_data':data_identity,'sampling':sampling,'effect_teacher':None,'noise_seed':343,'time_seed':344,
            'adapter_config':asdict(adapter_config),'private_names':sorted(private_order),
            'loss_recipe':asdict(FlowLossRecipe(microbatch_scenes=t['microbatch_scenes'])),
            'flow_semantics':'original_repeat8_Beta1.5_1_s.999','effect_protocol':'verified_pair_v1'}
        full=output/'continuous16';split=output/'split8plus8'
        for run in (full,split):validate_run(run,expected,head,args.device)
        if (output/'COMPLETE.json').exists():
            reused=reuse_complete(output)
            for name,run,steps in (('continuous',full,(FIRST,TOTAL)),('resumed',split,(BOUNDARY,FIRST,TOTAL))):
                for step in steps:
                    row=boundary(run,step)
                    if reused['checkpoints'][f'{name}{step}']!=artifact(row[2]):raise ValueError('Foreign completion checkpoint pin')
                    validate_state(row[3],read_json(run/'IDENTITY.json'),step,private_order)
            if reused['dispatch_budget']!=read_json(output/'DISPATCH_BUDGET.json'):
                raise ValueError('Qualification completion dispatch ledger changed')
            return reused
        commands=[]
        for run,target in ((full,TOTAL),(split,BOUNDARY),(split,TOTAL)):
            # The split8 -> split16 transition is explicitly the one authorized
            # resume, even on this first top-level invocation.
            resume_before=args.resume
            if commands and run==split and target==TOTAL:args.resume=True
            try:commands.append(segment(config,args,run,target,expected=expected,head=head))
            finally:args.resume=resume_before
        states={};pins={};state_audits={}
        for name,run,steps in (('continuous',full,(FIRST,TOTAL)),('resumed',split,(BOUNDARY,FIRST,TOTAL))):
            train_identity=read_json(run/'IDENTITY.json')
            for step in steps:
                _,_,path,state,_=boundary(run,step);key=f'{name}{step}'
                state_audits[key]=validate_state(state,train_identity,step,private_order);states[key]=state;pins[key]=artifact(path)
        comparisons={'first_resumed_update9':compare_states(states['continuous9'],states['resumed9'],9),
            'final_update16':compare_states(states['continuous16'],states['resumed16'],16)}
        next_ids=next_scene_proof(pool,states['resumed8'],states['continuous9'],states['resumed9'],seed=42,batch_size=32)
        exact=all(r['exact'] for r in comparisons.values()) and next_ids['exact']
        result={'schema':'action_effect_flow_real_resume_result_v1','status':'PASS' if exact else 'FAIL',
            'identity':artifact(output/'IDENTITY.json'),'qualification_config':artifact(config),'exact_resume_this_runtime':exact,
            'checkpoints':pins,'state_audits':state_audits,'comparisons':comparisons,'next_batch':next_ids,'commands':commands,
            'new_updates_this_invocation':sum(r['new_updates'] for r in commands),'total_completed_updates':32,
            'dispatch_budget':read_json(output/'DISPATCH_BUDGET.json'),'runtime':read_json(full/'IDENTITY.json')['runtime'],
            'actual_batch_scenes':32,'repeat':8,'unique_fit_scenes':len({r['scene_id'] for r in pool}),
            'new_VLM_calls':0,'new_simulation_calls':0,'new_teacher_calls':0,
            'terminal_zero_extra_updates':'NOT_RUN: original1000 horizon retained, checkpoint16 is paused',
            'scope':'B0 original head and private velocity adapter, single actual CUDA backend engineering continuation only',
            'scientific_gain':'UNTESTED; does not qualify every B1-effect-only auxiliary loss on CUDA'}
        write_json(output/'RESULT.json',result,immutable=False)
        if exact:write_json(output/'COMPLETE.json',{'status':'PASS','identity':artifact(output/'IDENTITY.json'),
            'result':artifact(output/'RESULT.json'),'checkpoint_comparisons':{k:v['exact'] for k,v in comparisons.items()},'next_ids_exact':next_ids['exact']})
        return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','context-manifest','repair-dataset','output','device'):p.add_argument('--'+name,required=True)
    p.add_argument('--arm',choices=('B0',),default='B0');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-seconds-per-segment',type=int,default=1200)
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    args=p.parse_args(argv)
    try:
        result=qualify(args);print(json.dumps(result));return 0 if result['status'] in ('PASS','NOT_RUN') else 1
    except Exception as exc:
        print(json.dumps({'status':'FAIL','error_type':type(exc).__name__,'error':str(exc),'exact_resume':False}));return 1


if __name__=='__main__':raise SystemExit(main())
