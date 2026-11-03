"""Fresh/own-resume AUGUSTUS extension of the shared atomic native trainer."""
import copy
import json
import math
from pathlib import Path
import torch
import torch.distributed as dist
from starVLA.rl.qwen_world_compare.contracts import fingerprint,file_sha
from starVLA.rl.qwen_world_compare.recovery_state import tensor_digest
from starVLA.rl.d44_native_rl.data import DataStream
from starVLA.rl.d44_native_rl.recovery import Boundary,Attempt,material_digest
from starVLA.rl.qwen_world_abcd_research import checkpoint
from .config import SCHEMA,validate,read,kernel_identity,aligned


def source_inventory():
    from tools.d44_native_rl.train import code_identity
    files=code_identity()
    for root in('starVLA/rl/d46_augustus','tools/d46_augustus'):
        files.update({str(p):file_sha(p)for p in sorted(Path(root).rglob('*.py'))})
    return files


class FixedKL:
    def __init__(self,cfg):self.beta=cfg['beta'];self.collections=0;self.ema=0.;self.target=cfg['target_kl']
    def update(self,joint_per_step):
        value=float(joint_per_step)/24  # Same legacy diagnostic units; never adapt beta.
        if not math.isfinite(value)or value<0:raise FloatingPointError('Invalid AUGUSTUS reference KL')
        self.ema=value if self.collections==0 else .9*self.ema+.1*value
        self.collections+=1;return self.beta
    def state_dict(self):return dict(schema='d46_augustus_fixed_KL_v1',beta=self.beta,collections=self.collections,ema=self.ema,target=self.target)
    def load_state_dict(self,value):
        if(value.get('schema')!='d46_augustus_fixed_KL_v1'or value.get('beta')!=self.beta or value.get('target')!=self.target
            or type(value.get('collections'))is not int or value['collections']<0 or not math.isfinite(value.get('ema',float('nan')))or value['ema']<0):
            raise ValueError('Changed fixed KL controller or collection clock')
        self.collections=value['collections'];self.ema=value['ema']


def advantages(raw,valid,G,world):
    """GRPO sample-group std only; global raw std is diagnostic, never a divisor."""
    if(raw.ndim!=2 or raw.shape[1]!=G or G!=8 or valid.shape!=raw.shape[:1]or valid.dtype!=torch.bool
        or raw.requires_grad or not torch.isfinite(raw).all()):raise ValueError('Invalid detached G8 credit/valid mask')
    if not torch.allclose(raw.mean(1),torch.zeros_like(raw.mean(1)),atol=1e-10,rtol=0):raise ValueError('Expected group-centered canonical credit')
    scale=raw.double().std(1,unbiased=True,keepdim=True)
    normalized=(raw.double()/(scale+1e-6)).float()*valid[:,None]
    moments=torch.stack(((raw*valid[:,None]).sum(),(raw.square()*valid[:,None]).sum(),valid.sum().double()*G))
    if world>1:
        if not dist.is_initialized()or dist.get_world_size()!=world:raise ValueError('Missing actual allrank diagnostic collective')
        dist.all_reduce(moments)
    if moments[2]<=0:raise ValueError('No valid scenes in global collection')
    std=(moments[1]/moments[2]-(moments[0]/moments[2]).square()).clamp_min(0).sqrt()
    return normalized.detach(),std.detach()


def credit(scenes,stream,epoch):
    del stream,epoch
    device=scenes[0]['chain'].device
    scores=torch.tensor([[q['metrics']['score']for q in r['records']]for r in scenes],device=device,dtype=torch.float64)
    if scores.shape[1]!=8 or not torch.isfinite(scores).all()or(scores<0).any()or(scores>1).any():raise ValueError('Invalid real canonical G8 reward')
    feasible=torch.tensor([[all(q['metrics'][k]==1 for k in('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound'))for q in r['records']]for r in scenes],device=device,dtype=torch.bool)
    return scores-scores.mean(1,keepdim=True),feasible


def recipe_credit(scenes,stream,epoch,cfg):
    if not aligned(cfg['experiment_id']):return credit(scenes,stream,epoch)
    from starVLA.rl.d44_native_rl.safety_credit import raw_credit
    return raw_credit(torch.stack([r['factors']for r in scenes]))


class AugustusBoundary(Boundary):
    @staticmethod
    def validate_boundary(state,batch,identity,optimizer=None,scheduler=None,**kw):
        cfg=validate(identity['recipe'],parent_checks=False)
        body=dict(identity);digest=body.pop('sha256',None)
        if(identity.get('algorithm_schema')!=SCHEMA or fingerprint(body)!=digest or identity.get('world')not in(8,16,32)
            or cfg['global_scenes']%identity['world']or identity.get('augustus_protocol_sha256')!=fingerprint(cfg['augustus'])
            or'any_virtual8'in state or'world16_virtual_sampler'in state):raise ValueError('Foreign AUGUSTUS native identity/world/virtual sampler')
        dual=FixedKL(cfg);dual.load_state_dict(state['adaptive_kl'])
        if state['scene_cursor']!=state['collections']*cfg['global_scenes']or state['raw_scene_exposures']!=state['scene_cursor']:
            raise ValueError('AUGUSTUS scene cursor/collection clock changed')
        if(state['padding_scene_exposures']!=state['raw_scene_exposures']-state['valid_scene_exposures']
            or state['new_candidates']!=state['raw_scene_exposures']*8 or state['valid_PG_candidates']!=state['valid_scene_exposures']*8
            or state['extra_B1_queries']!=0 or state['adaptive_G32_queries']!=0):raise ValueError('AUGUSTUS exposure/query accounting changed')
        if batch is not None:
            g=cfg['group'];shape=(g,3,8,3)
            for scene in batch['scenes']:
                for name,s in dict(chain=(g,4,8,3),old_mean=shape,ref_mean=shape,std=shape,old_clean_x=shape,ref_clean_x=shape,advantages=(g,)).items():
                    x=scene.get(name)
                    if(not torch.is_tensor(x)or tuple(x.shape)!=s or x.requires_grad or x.dtype!=torch.float32 or not torch.isfinite(x).all()):
                        raise ValueError('Invalid detached AUGUSTUS three-step material '+name)
                if(scene['std']<=0).any()or scene.get('augustus_kernel')!=kernel_identity(cfg):raise ValueError('Changed AUGUSTUS covariance/kernel profile')
                expected=torch.tensor(cfg['augustus']['training_sigma_grid'],dtype=torch.float32,device=scene['chain'].device)
                if(not torch.allclose(scene['timesteps'],expected,atol=1e-7,rtol=0)
                    or not torch.allclose(scene['deltas'],(expected[1:]-expected[:-1]),atol=1e-7,rtol=0)):
                    raise ValueError('Changed AUGUSTUS linear training grid')
                reward=torch.tensor([q['metrics']['score']for q in scene['records']],device=scene['chain'].device,dtype=torch.float64)
                if aligned(cfg['experiment_id']):
                    from starVLA.rl.d44_native_rl.safety_credit import raw_credit
                    raw,_=raw_credit(scene['factors'][None]);raw=raw[0]
                else:raw=reward-reward.mean()
                a=(raw/(raw.std(unbiased=True)+1e-6)).float()
                if scene['padding']:a.zero_()
                if not torch.allclose(scene['advantages'],a,atol=1e-6,rtol=1e-5)or scene['beta']!=cfg['beta']:
                    raise ValueError('Changed AUGUSTUS group normalization/fixed reference beta')
        result=Boundary.validate_boundary(state,batch,identity,optimizer,scheduler,**kw)
        result['augustus_pending_material_sha256']=None if batch is None else tensor_digest([{k:r[k]for k in('old_clean_x','ref_clean_x','augustus_kernel')}for r in batch['scenes']])
        return result


class AugustusExtension:
    boundary=AugustusBoundary();make_attempt=staticmethod(Attempt)
    actor_factory=staticmethod(lambda model,backend,cfg:__import__('starVLA.rl.d46_augustus.actor',fromlist=['AugustusActor']).AugustusActor(model,backend,cfg))
    collect_scene=staticmethod(lambda *args,**kw:__import__('starVLA.rl.d46_augustus.collector',fromlist=['collect_scene']).collect_scene(*args,**kw))
    advantages=staticmethod(advantages);make_dual=staticmethod(FixedKL)
    def credit(self,scenes,stream,epoch):return recipe_credit(scenes,stream,epoch,self.cfg)
    def initial_qualification(self,actor,conditions,token,noise,expected,root,rank):
        if not aligned(self.cfg['experiment_id']):return
        from starVLA.model.modules.qwen_world_postcot.flow import clean_x_step
        from starVLA.rl.flow_grpo.transactions import atomic_json
        spec=self.cfg['augustus'];grid=spec['deterministic_prefix_grid'][:-1]+spec['training_sigma_grid']
        with torch.no_grad(),actor._context_call(conditions.get(token))as material:
            prefix=actor._prepare(material,False).index_select(torch.zeros(1,device=actor.device,dtype=torch.long))
            z=noise.float()
            for t,nxt in zip(grid[:-1],grid[1:]):
                x0=actor.predict_clean(z,z.new_full((1,),t),prefix)
                z=clean_x_step(z,x0,t,nxt,validate=False)
        error=float((z-expected['normalized_action']).abs().max())
        if error!=0:raise FloatingPointError('Aligned cached-condition B1 differs from native deployment: '+str(error))
        atomic_json(Path(root)/f'aligned_B1_parity_rank{rank}.json',dict(status='PASS',normalized_action_max_abs=error,
            exact_native_K10_shift5=True,seed=2101,reference_parent=self.cfg['parent_checkpoint']))
    @staticmethod
    def json_result(value):json.dumps(value,allow_nan=False);return value
    @staticmethod
    def init_control_group():
        from datetime import timedelta
        from starVLA.rl.simwam_recipe import checkpoint as source
        if not dist.is_initialized()or dist.get_world_size()not in(8,16,32):raise ValueError('AUGUSTUS world8/16/32 required')
        if source.CONTROL_GROUP is not None:raise ValueError('Do not replace an existing control group')
        source.CONTROL_GROUP=dist.new_group(backend='gloo',timeout=timedelta(seconds=600))
    def read_recipe(self,args):
        self.args=args;self.cfg=read(args.config);self.saved_identity=None
        if args.resume:
            cp=Path(args.resume).resolve()
            if not cp.is_relative_to(Path(args.output).resolve()/'checkpoints'):raise ValueError('Only own AUGUSTUS native resume, never old RL/native world transition')
            self.saved_identity=json.loads((cp/'identity.json').read_text())
            if(self.saved_identity.get('algorithm_schema')!=SCHEMA or self.saved_identity.get('recipe')!=self.cfg
                or self.saved_identity.get('qualification')is not args.qualification):raise ValueError('AUGUSTUS native recipe/mode changed')
        return self.cfg
    def validate_world(self,world):
        if world not in(8,16,32)or self.cfg['global_scenes']%world:raise ValueError('AUGUSTUS requires divisible B64/128 on world8/16/32')
        if self.saved_identity is not None and self.saved_identity['world']!=world:raise ValueError('Own native resume cannot silently change world')
    def make_stream(self,cfg):return DataStream(cfg,cfg['global_scenes'])
    def identity(self,actual):
        from starVLA.rl.d44_native_rl.parent import D46_MODEL_SHA
        value=copy.deepcopy(actual);value.pop('sha256',None)
        if value.get('parent_model_sha256')!=D46_MODEL_SHA:raise ValueError('AUGUSTUS must start from original verified IL46, not another RL policy')
        value.update(algorithm_schema=SCHEMA,experiment_id=self.cfg['experiment_id'],augustus_protocol_sha256=fingerprint(self.cfg['augustus']),source_files=source_inventory())
        value['sha256']=fingerprint(value)
        if self.saved_identity is not None and value!=self.saved_identity:raise ValueError('Actual AUGUSTUS actor/parent/source/data/scope differs from own native')
        return value
    @staticmethod
    def optimizer_groups(actor,cfg):
        rows={}
        for name,p in actor.named_parameters():
            if not p.requires_grad:continue
            if not name.startswith('action_expert.')or'.embed_tokens.'in name or'.visual.'in name:raise ValueError('Reference/foreign trainable leaked into AUGUSTUS optimizer')
            if'.language_model.'in name:role='text_full'if cfg['augustus']['scope']=='full_text_action'else'text_lora'
            elif'.time_'in name:role='time_modulation'
            elif'.action_input.'in name or'.action_output.'in name:role='action_io'
            elif name.endswith('action_position'):role='action_position'
            elif name.endswith('p_embeddings'):role='predictive_tokens'
            elif'.role_embedding.'in name:role='role_embedding'
            elif'.ego_input.'in name:role='ego_input'
            else:raise ValueError('Unknown AUGUSTUS trainable role '+name)
            row=rows.setdefault(role,dict(name=role,lr=cfg['learning_rate'],initial_lr=cfg['learning_rate'],params=[]));row['params'].append(p)
        if not rows:raise ValueError('Empty AUGUSTUS trainable closure')
        return list(rows.values())
    def configure_actor(self,actor):
        profile=self.cfg['augustus']['execution'];actor.execution_fast_prefix=profile['fast_prefix'];actor.execution_fuse_fm_mc=False
        self.execution_caches=[]
        if profile['rotary_cache']:
            from starVLA.rl.d44_native_rl.execution_cache import install_invariant_cache
            for expert in(actor.action_expert,actor.ref_action_expert):
                self.execution_caches.append(install_invariant_cache(expert.decision,rotary=True,additive=True))
    def after_restore(self,actor,conditions,stream,state,identity):
        # Fresh model-vs-reference/K10 parity remains the shared trainer's strict
        # initial check; own native restore validates full Adam/RNG/material.
        pass
    def before_collection(self,*args):pass
    def before_scene(self,*args):pass
    def after_scene(self,*args):pass
    def after_collection(self,*args):pass
