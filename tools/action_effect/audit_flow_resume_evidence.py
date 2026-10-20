"""Read-only independent B0 CUDA resume artifact audit; never executes a model.

Deliberately does not import the qualification author's comparison or hash code.
Tensor equality includes dtype, shape and raw bytes (including signed zero).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import struct
import subprocess
import time
from pathlib import Path
import numpy as np
import torch


def json_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def exact_fingerprint(value):
    """Independent typed, length-framed byte fingerprint, not pickle equality."""
    h=hashlib.sha256()
    def add(tag,data=b''):
        h.update(tag);h.update(struct.pack('>Q',len(data)));h.update(data)
    def visit(x):
        if isinstance(x,torch.Tensor):
            y=x.detach().cpu().contiguous()
            if y.is_floating_point() and not torch.isfinite(y).all():raise ValueError('Nonfinite state tensor')
            add(b'T',str(y.dtype).encode());visit(tuple(y.shape));add(b'B',y.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,np.ndarray):
            if np.issubdtype(x.dtype,np.floating) and not np.isfinite(x).all():raise ValueError('Nonfinite numpy state')
            add(b'N',x.dtype.str.encode());visit(tuple(x.shape));add(b'B',x.tobytes(order='C'))
        elif isinstance(x,dict):
            add(b'D',str(len(x)).encode())
            for k in sorted(x,key=lambda k:(type(k).__name__,repr(k))):visit(k);visit(x[k])
        elif isinstance(x,(list,tuple)):
            add(b'L' if isinstance(x,list) else b'Q',str(len(x)).encode())
            for y in x:visit(y)
        elif x is None:add(b'0')
        elif type(x) is bool:add(b'b',bytes([x]))
        elif type(x) is int:add(b'i',str(x).encode())
        elif type(x) is float:
            if not math.isfinite(x):raise ValueError('Nonfinite scalar state')
            add(b'f',struct.pack('>d',x))
        elif isinstance(x,str):add(b's',x.encode())
        elif isinstance(x,np.generic):visit(np.asarray(x))
        else:raise TypeError('Unsupported state type '+str(type(x)))
    visit(value)
    return h.hexdigest()


def file_pin(path):
    p=Path(path).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}


def checked(pin):
    if file_pin(pin['path'])!=pin:raise ValueError('Artifact checksum/size mismatch '+pin['path'])
    return Path(pin['path'])


def compare_values(left,right):
    a,b=exact_fingerprint(left),exact_fingerprint(right)
    if a!=b:raise ValueError('Independent exact state comparison differs')
    return {'exact':True,'typed_byte_sha256':a}


def audit(root):
    if not __debug__:raise RuntimeError('Independent assertion audit must not run with Python optimization')
    torch.set_num_threads(2)
    start=time.monotonic();root=Path(root).resolve();pins={}
    def read(path):
        path=Path(path);pins[str(path.resolve())]=file_pin(path)
        return json.loads(path.read_text())
    complete=read(root/'COMPLETE.json');result=read(checked(complete['result']))
    qid=read(checked(complete['identity']));cfg=read(checked(result['qualification_config']))
    assert qid['source_commit']=='1ae4d3dcf84cecd0313548d597456298d5633ab6'
    assert complete['status']==result['status']=='PASS' and qid['arm']=='B0'
    assert qid['scheduler_horizon']==1000 and qid['continuous_updates']==16 and qid['split_updates']==[8,8]
    assert qid['total_new_update_ceiling']==32 and qid['qualification_save_validation_every']==9
    original_path=checked(qid['original_config'])
    import yaml
    original=yaml.safe_load(original_path.read_text());pins[str(original_path)]=file_pin(original_path)
    amended=json.loads(json.dumps(original));amended['training']['validation_every']=9
    assert amended==cfg
    assert {k:cfg['training'][k] for k in ('max_updates','warmup_updates','batch_scenes','microbatch_scenes')}==dict(max_updates=1000,warmup_updates=50,batch_scenes=32,microbatch_scenes=8)
    identities={run:read(root/run/'IDENTITY.json') for run in ('continuous16','split8plus8')}
    ident=identities['continuous16'];compare_values(ident,identities['split8plus8'])
    assert ident['source_commit']==qid['source_commit'] and ident['arm']=='B0' and ident['seed']==42
    assert ident['max_updates']==1000 and ident['recipe']==cfg['training'] and ident['effect_teacher'] is None
    assert ident['noise_seed']==343 and ident['time_seed']==344 and ident['loss_recipe']['repeat']==8
    assert ident['context_artifact']==qid['context'] and ident['repair_data']['dataset']==qid['repairs']
    runtime=ident['runtime']
    assert runtime==result['runtime'] and runtime['device']=='cuda:0' and runtime['gpu_name']=='NVIDIA A800-SXM4-80GB'
    assert {k:runtime[k] for k in ('trainable_dtype','autocast','deterministic','matmul_tf32','cudnn_tf32','cudnn_benchmark')}==dict(trainable_dtype='float32',autocast='disabled',deterministic=True,matmul_tf32=False,cudnn_tf32=False,cudnn_benchmark=False)
    context=read(checked(qid['context']));data=read(checked(qid['repairs']))
    assert context['status']=='COMPLETE' and context['max_samples']==64 and data['status']=='COMPLETE'
    assert data['split']=='train' and data['deployment_inputs'] is False
    binding=read(Path(cfg['artifact_root'])/'CAMPAIGN_BINDING.json')
    registration=read(checked(data['registration']));population=read(checked(data['population_completion']))
    assert binding['identity']==context['binding_identity']==registration['binding_identity']
    assert binding['base_checkpoint_binding_hash']==ident['base_binding_hash']
    assert binding['base_checkpoint']==data['base_checkpoint']
    assert population['registration_identity']==registration['identity']==data['registration_identity']
    context_rows={r['scene_id']:r for r in context['entries']}
    assert len(context_rows)==len(data['entries'])==64
    for row in data['entries']:
        assert row['context_entry']==context_rows[row['scene_id']]
        assert row['role']==row['context_entry']['role'] and row['source_log']==row['context_entry']['source_log']
    head_receipt=read(checked(ident['head']['head_receipt']))
    assert head_receipt['artifact']==ident['head']['head_artifact'] and head_receipt['strict_original_master'] is True
    assert head_receipt['base_binding_hash']==ident['base_binding_hash'] and ident['head']['strict_load'] is True
    # Reconstruct only the small CPU adapter's parameter insertion order. No
    # original 3.3GB head, Qwen, forward, backward or optimizer is instantiated.
    from starVLA.model.modules.action_effect.effect_flow_adapter import EffectFlowAdapter,EffectFlowAdapterConfig
    adapter=EffectFlowAdapter(EffectFlowAdapterConfig(**ident['adapter_config']))
    names=['adapter.'+n for n,_ in adapter.named_parameters()]
    parameter_count=sum(p.numel() for p in adapter.parameters());del adapter
    assert sorted(names)==ident['private_names']
    states={};adam_audits={}
    for key,pin in result['checkpoints'].items():
        path=checked(pin);pins[str(path)]=pin
        receipt=read(path.with_suffix('.COMPLETE.json'))
        assert receipt['checkpoint']==pin
        state=torch.load(path,map_location='cpu',weights_only=False);states[key]=state
        step=int(key.removeprefix('continuous').removeprefix('resumed'))
        assert state['schema']=='action_effect_private_checkpoint_v1' and state['optimizer_step']==step
        assert state['run_identity']==json_hash(ident)==receipt['run_identity']
        assert state['terminal'] is False and receipt['terminal'] is False
        assert state['termination_reason']==receipt['termination_reason']
        assert state['termination_reason']==('RUNNING_CHECKPOINT' if step==9 else 'OPERATOR_PAUSED')
        assert state['consumed_scenes']==step*32 and list(state['model'])==names
        assert [x['optimizer_step'] for x in state['history']]==list(range(1,step+1))
        assert set(state['stateful'])=={'flow_noise','flow_time'}
        assert set(state['rng'])=={'python','numpy','cpu','cuda'} and len(state['rng']['cuda'])==1
        for rng in [state['sampler'],*state['stateful'].values(),state['rng']['cpu'],*state['rng']['cuda']]:
            assert rng.dtype==torch.uint8 and rng.ndim==1 and rng.numel()>0
        groups=state['optimizer']['param_groups'];assert len(groups)==1
        assert groups[0]['params']==list(range(len(names)))
        active=[];dormant=[]
        for i,name in enumerate(names):
            tensor=state['model'][name]
            assert tensor.dtype==torch.float32 and torch.isfinite(tensor).all()
            moment=state['optimizer']['state'].get(i)
            if moment is None:
                assert name.startswith(('adapter.effect_projection.','adapter.effect_head.'));dormant.append(name);continue
            assert set(moment)=={'step','exp_avg','exp_avg_sq'} and int(moment['step'])==step
            for k in ('exp_avg','exp_avg_sq'):
                assert moment[k].dtype==torch.float32 and moment[k].shape==tensor.shape and torch.isfinite(moment[k]).all()
            active.append(name)
        assert len(active)==49 and len(dormant)==4 and len(state['optimizer']['state'])==49
        scheduler=state['scheduler'];assert scheduler['last_epoch']==step and scheduler['_step_count']==step+1
        assert scheduler['base_lrs']==[.0001]
        for row in state['history']:
            t=row['optimizer_step'];expected=.0001*((t+1)/50)
            assert math.isclose(row['lr'],expected,rel_tol=0,abs_tol=1e-20)
            assert row['consumed_scenes']==32*t and row['gradient_norm']>0 and math.isfinite(row['gradient_norm'])
            assert row['losses']['scene_presentations']==32 and row['losses']['FM_noise_presentations']==256
            assert row['losses']['repair_scenes']+row['losses']['preserve_scenes']==32
            assert row['losses']['verified_pair_teacher_calls']==0 and row['losses']['effect_supervised_scenes']==0
        adam_audits[key]={'private_tensor_count':len(names),'private_parameter_count':parameter_count,
            'active_Adam_names':active,'dormant_B0_aux_without_state':dormant,
            'optimizer_name_order_hash':json_hash(names),'step':step,'consumed_scenes':state['consumed_scenes']}
    comparisons={}
    fields=('model','optimizer','scheduler','sampler','stateful','rng','consumed_scenes','optimizer_step','run_identity','termination_reason','terminal')
    for step in (9,16):
        a,b=states['continuous'+str(step)],states['resumed'+str(step)]
        comparisons[str(step)]={k:compare_values(a[k],b[k]) for k in fields}
        # Timing is observational; every numerical history field must match.
        ah=[{k:v for k,v in r.items() if k!='seconds'} for r in a['history']]
        bh=[{k:v for k,v in r.items() if k!='seconds'} for r in b['history']]
        comparisons[str(step)]['all_history_except_wall_seconds']=compare_values(ah,bh)
    prefix=[{k:v for k,v in r.items() if k!='seconds'} for r in states['continuous16']['history'][:8]]
    compare_values(prefix,[{k:v for k,v in r.items() if k!='seconds'} for r in states['resumed8']['history']])
    dormant=adam_audits['resumed8']['dormant_B0_aux_without_state']
    for name in dormant:
        for key in ('resumed9','resumed16','continuous9','continuous16'):
            compare_values(states['resumed8']['model'][name],states[key]['model'][name])
    updated8to9=[name for name in names if exact_fingerprint(states['resumed8']['model'][name])!=exact_fingerprint(states['resumed9']['model'][name])]
    assert updated8to9 and not set(updated8to9)&set(dormant)
    # Independently reconstruct the 1:1 virtual reference pool from native
    # compact rows, not the qualification writer's saved next-batch result.
    fit=[r for r in data['entries'] if r['role']=='fit' and r['trainable'] is True]
    repairs=[r for r in fit if r['target_kind']=='verified_local_repair'];keeps=[r for r in fit if r['target_kind']=='identity_preservation']
    n=math.lcm(len(repairs),len(keeps));pool=[repairs[i%len(repairs)] for i in range(n)]+[keeps[i%len(keeps)] for i in range(n)]
    assert len(repairs)==1 and len(keeps)==44 and len(pool)==88
    assert json_hash([x['scene_id'] for x in pool])==ident['sampling']['reference_ids_sha256']
    gen=torch.Generator().manual_seed(143);next9=None;all_ids=[]
    for step in range(1,17):
        indices=torch.randint(len(pool),(32,),generator=gen).tolist()
        rows=[{'reference_index':i,'scene_id':pool[i]['scene_id'],'target_kind':pool[i]['target_kind']} for i in indices]
        all_ids.append(rows)
        hist=states['continuous16']['history'][step-1]['losses']
        assert hist['repair_scenes']==sum(pool[i]['target_kind']=='verified_local_repair' for i in indices)
        assert hist['unique_scene_ids']==len({pool[i]['scene_id'] for i in indices})
        if step==8:compare_values(gen.get_state(),states['resumed8']['sampler'])
        if step==9:
            next9=rows
            compare_values(gen.get_state(),states['continuous9']['sampler'])
            compare_values(gen.get_state(),states['resumed9']['sampler'])
        if step==16:compare_values(gen.get_state(),states['continuous16']['sampler'])
    compare_values(next9,result['next_batch']['next9_scene_ids'])
    from8=torch.Generator();from8.set_state(states['resumed8']['sampler'])
    resumed_indices=torch.randint(88,(32,),generator=from8).tolist()
    assert resumed_indices==[x['reference_index'] for x in next9]
    budget=read(root/'DISPATCH_BUDGET.json');assert budget==result['dispatch_budget'] and budget['scheduled_updates']==32
    launches=[]
    for j,cmd in enumerate(result['commands']):
        assert cmd['returncode']==0 and cmd['status']=='COMPLETE'
        assert (cmd['starting_step'],cmd['target_step'],cmd['new_updates'])==[(0,16,16),(0,8,8),(8,16,8)][j]
        assert budget['reservations'][j]['command']==cmd['command']
        args=cmd['command'];assert args[:3]==['/root/miniconda3/envs/ddp/bin/python','-m','tools.action_effect.train_effect_flow']
        for flag,v in [('--config',str(root/'QUALIFICATION_CONFIG.json')),('--context-manifest',qid['context']['path']),
                ('--repair-dataset',qid['repairs']['path']),('--stop-after',str(cmd['new_updates'])),
                ('--arm','B0'),('--device','cuda:0'),('--seed','42'),('--max-updates','1000')]:assert args[args.index(flag)+1]==v
        assert ('--resume' in args)==(j==2)
        stdout=checked(cmd['stdout']);stderr=checked(cmd['stderr'])
        pins[str(stdout)]=cmd['stdout'];pins[str(stderr)]=cmd['stderr']
        printed=json.loads(stdout.read_text().splitlines()[-1])
        assert printed['status']=='PAUSED' and printed['segment_updates']==cmd['new_updates']
        assert printed['optimizer_step']==cmd['target_step'] and printed['checkpoint']['checkpoint']==cmd['checkpoint']
        assert printed['peak_cuda_allocated_bytes']>0 and printed['peak_cuda_reserved_bytes']>0
        launches.append({'command':args,'stdout':cmd['stdout'],'stderr':cmd['stderr'],
            'new_updates':cmd['new_updates'],'step':cmd['target_step'],
            'CUDA_allocated_bytes':printed['peak_cuda_allocated_bytes'],'CUDA_reserved_bytes':printed['peak_cuda_reserved_bytes']})
    resolution=read(root/'split8plus8/RESUME_RESOLUTION.json')
    assert resolution['selected']==result['checkpoints']['resumed8']['path'] and resolution['optimizer_step']==8
    source_root=Path(qid['original_config']['path']).parents[2]
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source_root,text=True).strip()==qid['source_commit']
    assert not subprocess.check_output(['git','status','--porcelain=v1'],cwd=source_root,text=True).strip()
    sources={}
    for rel in ('tools/action_effect/train_effect_flow.py','tools/action_effect/training.py',
                'starVLA/model/modules/action_effect/effect_flow_adapter.py',
                'starVLA/model/modules/action_effect/effect_distillation.py',
                'starVLA/model/framework/effect_internalized_flow.py','tools/action_effect/qualify_flow_resume.py'):
        actual=(source_root/rel).read_bytes()
        committed=subprocess.check_output(['git','show',qid['source_commit']+':'+rel],cwd=source_root)
        assert actual==committed;sources[rel]=hashlib.sha256(actual).hexdigest()
    for p,pin in pins.items():assert file_pin(p)==pin
    return {'schema':'action_effect_independent_B0_CUDA_resume_audit_v1','status':'PASS',
        'scope':'B0 single-CUDA private velocity adapter numerical continuation only',
        'source_commit':qid['source_commit'],'source_file_hashes':sources,'qualification_root':str(root),
        'runtime':runtime,'data':{'context':qid['context'],'repair_dataset':qid['repairs'],
            'unique_fit_scenes':45,'repair_scenes':1,'keep_scenes':44,'virtual_reference_count':88,
            'Base_binding_hash':ident['base_binding_hash'],'head_receipt':ident['head']['head_receipt'],
            'head_artifact_expected_pin':ident['head']['head_artifact'],
            'Base_head_rehashed_by_this_read_only_audit':False},
        'checkpoints':result['checkpoints'],'checkpoint_state_comparisons':comparisons,'Adam_name_mapping':adam_audits,
        'next9_scene_ids':next9,'next9_scene_ids_hash':json_hash(next9),'all16_sample_sequences_hash':json_hash(all_ids),
        'actually_updated_parameter_names8to9':updated8to9,'dormant_B0_aux_unchanged_across_all_boundaries':True,
        'continuous8_prefix_history_exact':True,'launches':launches,
        'original_artifact_pins':list(pins.values()),'original_artifacts_unchanged':True,
        'wall_seconds':time.monotonic()-start,'new_GPU_calls':0,'new_forwards':0,'new_backward_or_updates':0,'new_simulations':0,
        'limitations':['Full per-parameter gradients were not saved; exact recorded gradient norms and updates were checked.',
            'CUDA RNG bytes were checked, not advanced on a new GPU.',
            'Base 3.3GB frozen head hash is inherited from the checked original-head receipt; no new full-head scan.',
            'Checkpoint16 is OPERATOR_PAUSED under original1000 axis; terminal zero-update CUDA resume NOT_RUN.',
            'Does not qualify B1-effect auxiliary gradients, production DDP, cross-device continuation or scientific gain.']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--qualification-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();report=audit(a.qualification_root);path=Path(a.output)
    payload=(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    if path.exists() and path.read_bytes()!=payload:raise ValueError('Never overwrite independent audit evidence')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
    print(json.dumps({'status':report['status'],'report':file_pin(path),'new_updates':0,'new_simulations':0}))


if __name__=='__main__':main()
