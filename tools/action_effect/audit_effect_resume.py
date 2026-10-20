"""Independent CPU-only audit of an already executed G1 CUDA qualification.

No training, signal, process restart, GPU access or evidence rewrite occurs.
Expected source is the source actually executed, not the reviewer's later SHA.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
import torch
from .common import artifact,digest,read_json,verify_artifact,verify_identity,write_json
from .context_data import ContextReader
from starVLA.model.modules.action_effect.effect_model import EffectModelConfig,FeedbackLabelSpec

FIELDS=('model','optimizer','scheduler','sampler','stateful','rng','consumed_scenes','optimizer_step','run_identity','termination_reason','terminal')


def exact_hash(value):
    """Independent typed byte fingerprint; no tolerance or NaN equality."""
    def encode(x):
        if isinstance(x,torch.Tensor):
            y=x.detach().cpu().contiguous()
            if y.is_floating_point() and not torch.isfinite(y).all():raise ValueError('Nonfinite checkpoint tensor')
            return ['tensor',str(y.dtype),list(y.shape),hashlib.sha256(y.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()]
        if isinstance(x,np.ndarray):
            if np.issubdtype(x.dtype,np.floating) and not np.isfinite(x).all():raise ValueError('Nonfinite RNG array')
            return ['ndarray',str(x.dtype),list(x.shape),hashlib.sha256(x.tobytes()).hexdigest()]
        if isinstance(x,dict):return ['dict',[[encode(k),encode(v)] for k,v in sorted(x.items(),key=lambda kv:(type(kv[0]).__name__,str(kv[0])))]]
        if isinstance(x,(tuple,list)):return [type(x).__name__,[encode(v) for v in x]]
        if isinstance(x,float):
            if not np.isfinite(x):raise ValueError('Nonfinite checkpoint scalar')
            return ['float',struct.pack('!d',x).hex()]
        if x is None or type(x) in (str,int,bool):return [type(x).__name__,x]
        raise TypeError('Unreviewed checkpoint state type: '+type(x).__name__)
    return digest(encode(value))


def check_subset(actual,expected):
    expected=json.loads(json.dumps(expected,allow_nan=False))
    for key,value in expected.items():
        if key not in actual or digest(actual[key])!=digest(value):raise ValueError('Actual training identity differs: '+key)


def read_state(path,run_identity,step):
    path=Path(path);receipt=read_json(path.with_suffix('.COMPLETE.json'))
    if verify_artifact(receipt['checkpoint']).resolve()!=path.resolve() or receipt['run_identity']!=run_identity:
        raise ValueError('Checkpoint receipt namespace/identity mismatch')
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['run_identity']!=run_identity or state['optimizer_step']!=step or receipt['optimizer_step']!=step:
        raise ValueError('Wrong checkpoint optimizer boundary')
    if sorted(state['model'])!=sorted(read_json(path.parent/'IDENTITY.json')['private_names']):
        raise ValueError('Checkpoint private parameter names differ')
    if state['consumed_scenes']!=step*read_json(path.parent/'IDENTITY.json')['recipe']['batch_scenes']:
        raise ValueError('Consumed data cursor differs from actual optimizer boundary')
    if [r['optimizer_step'] for r in state['history']]!=list(range(1,step+1)):
        raise ValueError('Incomplete/duplicated optimizer history')
    for i,(name,tensor) in enumerate(state['model'].items()):
        if tensor.dtype!=torch.float32:raise ValueError('Private parameter precision changed')
        moment=state['optimizer']['state'].get(i)
        if moment is None:raise ValueError('Missing actual Adam state: '+name)
        for k in ('exp_avg','exp_avg_sq'):
            if moment[k].shape!=tensor.shape or moment[k].dtype!=torch.float32 or not torch.isfinite(moment[k]).all():
                raise ValueError('Adam moment/name/precision mismatch: '+name)
        if int(moment['step'])!=step:raise ValueError('Adam update count differs: '+name)
    params=[i for g in state['optimizer']['param_groups'] for i in g['params']]
    if params!=list(range(len(state['model']))):raise ValueError('Optimizer parameter order changed')
    return state


def compare_states(a,b,step):
    out={}
    for field in FIELDS:
        ah,bh=exact_hash(a[field]),exact_hash(b[field]);out[field]={'equal':ah==bh,'first_sha256':ah,'second_sha256':bh}
    for actual_step in (9,16):
        if actual_step>step:continue
        ra=a['history'][actual_step-1];rb=b['history'][actual_step-1]
        for field in ('losses','gradient_norm','lr','consumed_scenes'):
            ah,bh=exact_hash(ra[field]),exact_hash(rb[field]);out[f'update{actual_step}_{field}']={'equal':ah==bh,'first_sha256':ah,'second_sha256':bh}
    return {'exact':all(r['equal'] for r in out.values()),'components':out}


def audit(root,*,expected_source,config,context_manifest,pair_dataset,arm,seed,output):
    root=Path(root).resolve();meta=read_json(root/'IDENTITY.json');cfg=read_json(root/'QUALIFICATION_CONFIG.json')
    original=read_json(config) if str(config).endswith('.json') else __import__('yaml').safe_load(Path(config).read_text())
    expected_cfg=json.loads(json.dumps(original));expected_cfg['training']['validation_every']=9
    if digest(cfg)!=digest(expected_cfg):raise ValueError('Qualification altered more than declared save/validation cadence')
    if original['training']['max_updates']!=1000:raise ValueError('Unexpected original scheduler horizon')
    check_subset(meta,{'schema':'action_effect_real_resume_qualification_v1','source_commit':expected_source,
        'original_config':artifact(config),'context':artifact(context_manifest),'pairs':artifact(pair_dataset),
        'arm':arm,'seed':seed,'scheduler_horizon':1000,'continuous_updates':16,'split_updates':[8,8],'total_new_update_ceiling':32,
        'qualification_save_validation_every':9})
    done=read_json(root/'COMPLETE.json');result=read_json(root/'RESULT.json')
    if done['identity']!=artifact(root/'IDENTITY.json') or done['result']!=artifact(root/'RESULT.json'):
        raise ValueError('Foreign qualification completion namespace')
    if result['identity']!=artifact(root/'IDENTITY.json') or result['qualification_config']!=artifact(root/'QUALIFICATION_CONFIG.json'):
        raise ValueError('Foreign qualification result namespace')
    if done['status']!='PASS' or result['status']!='PASS':raise ValueError('Author qualification was not complete PASS')
    reader=ContextReader(context_manifest,Path(original['artifact_root'])/'CAMPAIGN_BINDING.json')
    data=verify_identity(read_json(pair_dataset));groups={'fit':set(),'validation':set()};by={};role_scenes={r:set() for r in groups}
    for row in data['entries']:
        verify_identity(row);role=row['role'];groups[role].add(row['source_log']);role_scenes[role].add(row['scene_id'])
        if role=='fit' and row['kind']!='joint':by.setdefault(row['scene_id'],[]).append(row)
    if groups['fit']&groups['validation'] or not by:raise ValueError('Invalid real fit/validation log separation')
    scenes=sorted(by);shuffle={}
    for role,values in role_scenes.items():
        ordered=sorted(values,key=lambda s:digest(['effect_shuffle',s]))
        if len(ordered)>1:shuffle.update({s:ordered[(i+1)%len(ordered)] for i,s in enumerate(ordered)})
    t=cfg['training'];spec=FeedbackLabelSpec.from_label_arrays(data['entries'][0]['base_labels'])
    expected={'schema':'action_effect_small_training_v1','source_commit':expected_source,'arm':arm,'seed':seed,
        'max_updates':1000,'recipe':t,'dataset_size':len(scenes),
        'base_binding_hash':reader.binding['base_checkpoint_binding_hash'],'data_artifact':artifact(pair_dataset),'context_artifact':artifact(context_manifest),
        'label_spec':asdict(spec),'model_config':asdict(EffectModelConfig(mode=arm,hidden_dim=t['hidden'],layers=t['layers'],heads=t['heads'],init_seed=seed)),
        'withheld_training_interventions':['joint'],'scene_shuffle_map_hash':digest(shuffle),
        'sampling':'uniform fit scene then uniform valid pair',
        'loss_weights':{'value':1.,'ranking':.2,'events':.2,'temporal':.2,'uncertainty':.05,'repair':.2}}
    identities={};states={};pins={}
    for name,boundaries in (('continuous16',(9,16)),('split8plus8',(8,9,16))):
        identity=read_json(root/name/'IDENTITY.json');check_subset(identity,expected);identities[name]=artifact(root/name/'IDENTITY.json')
        check_subset(identity['runtime'],{'device':meta['device'],'deterministic':True,'trainable_dtype':'float32','autocast':'disabled',
            'matmul_tf32':False,'cudnn_tf32':False,'cudnn_benchmark':False})
        if not identity['runtime']['device'].startswith('cuda'):raise ValueError('Actual qualification did not execute CUDA')
        for boundary in boundaries:
            path=root/name/f'step_{boundary:06d}.pt';key=f'{name}_{boundary}'
            states[key]=read_state(path,digest(identity),boundary);pins[key]=artifact(path)
    first=compare_states(states['continuous16_9'],states['split8plus8_9'],9)
    final=compare_states(states['continuous16_16'],states['split8plus8_16'],16)
    sampler=torch.Generator().manual_seed(seed+101);pair_rng=torch.Generator().manual_seed(seed+202)
    def draw():
        ids=torch.randint(len(scenes),(t['batch_scenes'],),generator=sampler).tolist();rows=[]
        for i in ids:
            options=by[scenes[i]];j=int(torch.randint(len(options),(1,),generator=pair_rng))
            rows.append({'scene_id':scenes[i],'pair_identity':options[j]['identity']})
        return rows
    for _ in range(8):draw()
    s8=states['split8plus8_8']
    boundary8=exact_hash(sampler.get_state())==exact_hash(s8['sampler']) and exact_hash(pair_rng.get_state())==exact_hash(s8['stateful']['pair_rng'])
    actual_next=draw();next_states={}
    for name in ('continuous16_9','split8plus8_9'):
        x=states[name];next_states[name]=exact_hash(sampler.get_state())==exact_hash(x['sampler']) and exact_hash(pair_rng.get_state())==exact_hash(x['stateful']['pair_rng'])
    if digest(actual_next)!=result['next_batch']['scene_pair_ids_sha256']:raise ValueError('Author next-scene proof differs from independent reconstruction')
    budget=read_json(root/'DISPATCH_BUDGET.json');reservations=budget['reservations']
    expected_dispatch=[('continuous16',0,16),('split8plus8',0,8),('split8plus8',8,16)]
    if budget['scheduled_updates']!=32 or len(reservations)!=3 or len(result['commands'])!=3:
        raise ValueError('Actual dispatch not the bounded fresh 16 versus8+8 experiment')
    commands=[]
    for reservation,entry,(name,start,target) in zip(reservations,result['commands'],expected_dispatch):
        cmd=entry['command'];get=lambda key:cmd[cmd.index(key)+1]
        if (Path(reservation['run']).resolve()!=root/name or reservation['from_step']!=start or reservation['target_step']!=target
                or reservation['new_update_ceiling']!=target-start or reservation['command']!=cmd
                or Path(get('--output')).resolve()!=root/name or get('--max-updates')!='1000' or int(get('--stop-after'))!=target-start
                or ('--resume' in cmd)!=(start>0) or entry['returncode']!=0):
            raise ValueError('Actual training dispatch lineage/budget differs')
        for key,want in (('--config',root/'QUALIFICATION_CONFIG.json'),('--context-manifest',Path(context_manifest)),('--pair-dataset',Path(pair_dataset))):
            if Path(get(key)).resolve()!=want.resolve():raise ValueError('Actual command used wrong input: '+key)
        if get('--arm')!=arm or int(get('--seed'))!=seed or get('--device')!=meta['device']:
            raise ValueError('Actual command arm/seed/device differs')
        commands.append({k:artifact(verify_artifact(entry[k])) for k in ('stdout','stderr','checkpoint')})
    exact=first['exact'] and final['exact'] and boundary8 and all(next_states.values())
    receipt={'schema':'action_effect_independent_actual_resume_audit_v1','status':'PASS' if exact else 'FAIL',
        'scope':'already executed real single-GPU G1 update9/16 resume; CPU offline state/provenance recheck',
        'qualification_root':str(root),'actual_executed_source_commit':expected_source,
        'review_tool_source_commit':__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip(),
        'review_tool_source_file':artifact(__file__),
        'input_pins':{'qualification_identity':artifact(root/'IDENTITY.json'),'author_result':artifact(root/'RESULT.json'),
            'author_complete':artifact(root/'COMPLETE.json'),'original_config':artifact(config),'qualification_config':artifact(root/'QUALIFICATION_CONFIG.json'),
            'context':artifact(context_manifest),'pairs':artifact(pair_dataset)},
        'training_identity_pins':identities,'checkpoint_pins':pins,'commands':commands,
        'first_resumed_update9':first,'final_update16':final,'boundary8_sampler_exact':boundary8,
        'next9_sampler_states_exact':next_states,'next9_scene_pairs_sha256':digest(actual_next),
        'training_runtime':read_json(root/'continuous16'/'IDENTITY.json')['runtime'],
        'actual_completed_optimizer_updates':32,'new_optimizer_updates_by_this_audit':0,'new_GPU_calls_by_this_audit':0,'new_simulator_calls':0,
        'qualification_only':'No convergence, learned-effect quality, planning improvement or cross-hardware exactness claimed',
        'scientific_gain':'UNTESTED'}
    receipt['identity']=digest(receipt);write_json(output,receipt);return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('qualification-root','expected-source','config','context-manifest','pair-dataset','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--arm',default='delta_feedback');p.add_argument('--seed',type=int,default=42);a=p.parse_args(argv)
    try:
        r=audit(a.qualification_root,expected_source=a.expected_source,config=a.config,context_manifest=a.context_manifest,
            pair_dataset=a.pair_dataset,arm=a.arm,seed=a.seed,output=a.output)
        print(json.dumps({'status':r['status'],'identity':r['identity'],'output':a.output}));return 0 if r['status']=='PASS' else 1
    except Exception as e:print(json.dumps({'status':'FAIL','error_type':type(e).__name__,'error':str(e)}));return 1


if __name__=='__main__':raise SystemExit(main())
