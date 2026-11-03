"""Opt-in execution adapter; shares the unchanged native scene/loss loop."""
import copy
import json
import os
from pathlib import Path
import time
import uuid
import torch
from starVLA.rl.qwen_world_compare.contracts import file_sha,fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.d44_native_rl.config import read
from starVLA.rl.d44_native_rl.data import DataStream
from starVLA.rl.d44_native_rl.recovery import Attempt,checkpoint
from .native import SCHEMA,STATE_SCHEMA,CONTROL_PROTOCOL,World16Boundary,source_inventory,validate_virtual,tensor_sha,validate_supported_recipe,FPO_PROTOCOL
from .planning import fixed_scene_rng_plan,assert_scene_rng_end


class VirtualSceneStream:
    def __init__(self,base):
        self.base=base
    def __getattr__(self,name):return getattr(self.base,name)
    def at(self,cursor,rank,local_scene,world):
        if world!=16 or not 0<=local_scene<4:raise ValueError('Fixed B64/world16 scene assignment')
        return self.base.at(cursor,rank%8,local_scene+4*(rank//8),8)


def receipt(root,identity):
    value=json.loads((Path(root)/'WORLD_TRANSITION.json').read_text());body=dict(value);sha=body.pop('sha256',None)
    transition=identity.get('native_world_transition',{})
    if (fingerprint(body)!=sha or value.get('schema')!=SCHEMA or value.get('target_identity_sha256')!=identity['sha256']
        or value.get('source_identity_sha256')!=transition.get('source_run_identity_sha256')
        or value.get('common_optimizer_sha256')!=transition.get('common_optimizer_sha256')
        or value.get('source_inventory')!=identity.get('source_files') or value.get('source_inventory')!=source_inventory()
        or value.get('control_plane')!=CONTROL_PROTOCOL or transition.get('control_plane')!=CONTROL_PROTOCOL):
        raise ValueError('World16 native transition receipt/source changed')
    if identity['recipe']['arm']=='D3_FPO'and(value.get('flow_energy_protocol')!=FPO_PROTOCOL or transition.get('flow_energy_protocol')!=FPO_PROTOCOL):
        raise ValueError('World16 FPO transition protocol changed')
    return value


class World16Attempt(Attempt):
    """New execution journal starts at verified origin, never forges old ranks."""
    def __init__(self,root,rank,state,identity,resume):
        receipt(root,identity)
        self.root=Path(root);self.rank=rank;self.started=time.monotonic();self.id=checkpoint.gather(uuid.uuid4().hex if rank==0 else None)[0]
        self.path=self.root/'invocations'/self.id/f'rank{rank}.json';self.path.parent.mkdir(parents=True,exist_ok=True)
        origin=identity['native_world_transition']['origin_updates']
        self.record=dict(schema='d46_world16_invocation_v1',id=self.id,rank=rank,run=identity['sha256'],status='RUNNING',started=time.time(),
            start_update=state['updates'],resume=str(resume),physical_candidates=0,seconds_save=0.,GPU_time_is_shared_worker_wall=True,
            verified_native_origin_updates=origin,original_world8_run=identity['native_world_transition']['source_run_identity_sha256'])
        journal=self.root/f'metrics_rank{rank}.jsonl'
        rows=[json.loads(x)for x in journal.read_text().splitlines()]if journal.exists()else[]
        prefix=[x for x in rows if x['updates']<=state['updates']]
        if ([x['updates']for x in prefix]!=list(range(origin+1,state['updates']+1))or any(x['run']!=identity['sha256']for x in rows)):
            raise ValueError('World16 effective history gap/foreign identity')
        suffix=[x for x in rows if x['updates']>state['updates']]
        if suffix:
            archive=self.path.parent/f'aborted_metrics_rank{rank}.jsonl';archive.write_text(''.join(json.dumps(x)+'\n'for x in suffix))
            tmp=journal.with_suffix('.reconciled');tmp.write_text(''.join(json.dumps(x)+'\n'for x in prefix));os.replace(tmp,journal)
        atomic_json(self.path,self.record)


class World16Extension:
    boundary=World16Boundary()
    @staticmethod
    def json_result(value):
        """Public JSON summary only; never mutate or stringify native RNG tensors."""
        output=dict(value)
        if 'world16_virtual_sampler'in value:
            bank=value['world16_virtual_sampler']
            if bank.get('schema')!=STATE_SCHEMA or len(bank.get('lanes',[]))!=8:
                raise ValueError('Invalid virtual sampler for public JSON summary')
            output['world16_virtual_sampler']=dict(schema=STATE_SCHEMA,
                summary_schema='d46_world16_virtual8_sampler_JSON_SHA_summary_v1',
                committed_collections=bank['committed_collections'],lane_count=8,
                lane_sha256=[tensor_sha(x)for x in bank['lanes']],
                full_tensor_storage='native checkpoint rank state only')
        json.dumps(output,allow_nan=False)
        return output
    @staticmethod
    def init_control_group():
        from datetime import timedelta
        from starVLA.rl.simwam_recipe import checkpoint as source
        if not torch.distributed.is_initialized()or torch.distributed.get_world_size()!=16:
            raise ValueError('World16 control group requires initialized16 ranks')
        if source.CONTROL_GROUP is not None:raise RuntimeError('Do not replace an existing control group')
        source.CONTROL_GROUP=torch.distributed.new_group(backend=CONTROL_PROTOCOL['backend'],
            timeout=timedelta(seconds=CONTROL_PROTOCOL['timeout_seconds']))
    make_attempt=staticmethod(World16Attempt)
    def read_recipe(self,args):
        self.args=args
        if not args.resume:raise ValueError('World16 starts only from explicit typed native transition/resume')
        cfg=read(args.config)
        validate_supported_recipe(cfg)
        self.saved_identity=json.loads((Path(args.resume)/'identity.json').read_text())
        if self.saved_identity['recipe']!=cfg or self.saved_identity.get('world')!=16 or self.saved_identity.get('qualification')is not args.qualification:
            raise ValueError('Saved world16 recipe/mode mismatch')
        receipt(args.output,self.saved_identity)
        return cfg
    def validate_world(self,world):
        if world!=16:raise ValueError('Explicit world16 execution only')
    def make_stream(self,cfg):return VirtualSceneStream(DataStream(cfg,cfg['global_scenes']))
    def identity(self,actual):
        saved=self.saved_identity
        exclude={'world','source_files','fresh_Adam','sha256','native_world_transition'}
        if {k:v for k,v in actual.items()if k not in exclude}!={k:v for k,v in saved.items()if k not in exclude}:
            raise ValueError('Actual actor/parent/data/closure differs from native ancestor')
        if saved['source_files']!=source_inventory():raise ValueError('Execution source changed')
        return copy.deepcopy(saved)
    def before_collection(self,state,generator,rank,world):
        self.validate_world(world);validate_virtual(state)
        cfg=validate_supported_recipe(self.saved_identity['recipe'])
        self.rank=rank;self.plan=fixed_scene_rng_plan(state['world16_virtual_sampler']['lanes'][rank%8],cfg['arm'],mc=cfg['mc_samples'])
        self.seen=[];self.cpu_rng=torch.get_rng_state().clone()
        self.cuda_rng=torch.cuda.get_rng_state().clone()if torch.cuda.is_available()else None
    def before_scene(self,generator,j):
        if j!=len(self.seen)or not 0<=j<4:raise ValueError('Scene ordinal changed')
        generator.set_state(self.plan['starts'][j+4*(self.rank//8)])
    def after_scene(self,generator,j):
        assert_scene_rng_end(generator,self.plan['ends'][j+4*(self.rank//8)]);self.seen.append(j)
    def after_collection(self,state,generator,rank,world):
        error=None
        if self.seen!=list(range(4))or not torch.equal(torch.get_rng_state(),self.cpu_rng):error='Unexpected scene count or global Torch RNG consumed'
        if self.cuda_rng is not None and not torch.equal(torch.cuda.get_rng_state(),self.cuda_rng):error='Unexpected global CUDA RNG consumed'
        checkpoint.check_errors(error)
        rows=checkpoint.gather(dict(rank=rank,end=self.plan['next_virtual_lane_state']))
        if len(rows)!=16 or sorted(x['rank']for x in rows)!=list(range(16)):raise ValueError('Incomplete virtual lane collective')
        rows=sorted(rows,key=lambda x:x['rank'])
        if any(not torch.equal(rows[r]['end'],rows[r+8]['end'])for r in range(8)):raise ValueError('Virtual lane preview disagrees')
        state['world16_virtual_sampler']=dict(schema=STATE_SCHEMA,lanes=[x['end'].clone()for x in rows[:8]],committed_collections=state['collections']+1)

    def after_restore(self,actor,conditions,stream,state,identity):
        from starVLA.rl.simwam_recipe.contracts import seed_for
        rank=torch.distributed.get_rank();token,_=stream.at(state['scene_cursor'],rank,0,16)
        observation=stream.data.observation(token);inputs=conditions.processor([observation])
        ego=torch.tensor([observation['proprio']],device=actor.device,dtype=torch.float32)
        noise=torch.randn((1,8,3),generator=torch.Generator().manual_seed(seed_for('native_qwen_sourceflow_ODE',2101,token))).to(actor.device)
        payload=torch.load(Path(self.args.resume)/'optimizer.pt',map_location='cpu',weights_only=False)
        errors=compare_restored_B1(actor,payload['adapters'],inputs,ego,noise)
        value=dict(schema='d46_world16_post_native_B1_restore_parity_v1',status='PASS',rank=rank,token=token,updates=state['updates'],next_inner=state['next_inner'],
            source_native=str(self.args.resume),source_optimizer_sha256=file_sha(Path(self.args.resume)/'optimizer.pt'),
            exact_same_NATIVE_parameters=True,permanent_reference='original IL46 unchanged',single_B1=True,steps=10,shift=5.,seed=2101,
            private_noise='seed_for(native_qwen_sourceflow_ODE,2101,token)',max_abs=errors,no_scoring=True)
        path=Path(self.args.output)/f"native_B1_restore_parity_rank{rank}_u{state['updates']:06}.json"
        if path.exists():
            if json.loads(path.read_text())!=value:raise ValueError('Earlier B1 parity changed')
        else:atomic_json(path,value)

def compare_restored_B1(actor,adapters,inputs,ego,noise):
    """Independent native payload reapply, sharing only immutable IL base bytes."""
    import torch.nn as nn
    from starVLA.rl.simwam_recipe.lora_adapter import restore_adapters
    memo={id(p):p for p in actor.action_expert.parameters()if not p.requires_grad}
    model=copy.deepcopy(actor.action_expert,memo)
    wrapper=nn.Module();wrapper.action_expert=model
    restored={n:p for n,p in actor.named_parameters()if p.requires_grad}
    if set(restored)!=set(adapters)or any(not torch.equal(p.detach().cpu(),adapters[n])for n,p in restored.items()):
        raise ValueError('Actual restored parameter bytes differ from native payload')
    if any(id(p)in {id(x)for x in model.parameters()}for p in restored.values()):raise ValueError('Parity clone aliases trainable storage')
    restore_adapters(wrapper,adapters);model.eval().requires_grad_(False)
    with torch.no_grad():
        a=actor.action_expert.predict(inputs,ego,noise);b=model.predict(inputs,ego,noise)
    errors={name:float((a[name]-b[name]).abs().max())for name in('normalized_action','requested_trajectory')}
    if any(value!=0 for value in errors.values()):raise FloatingPointError('Post-native B1 restore parity differs '+str(errors))
    return errors



