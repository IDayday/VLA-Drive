"""Coherent packed S0 collection and DDP backward, using the existing trainer."""
import copy,time,traceback
from contextlib import nullcontext
from pathlib import Path
import torch
import torch.distributed as dist
from starVLA.rl.simwam_recipe import checkpoint
from starVLA.rl.qwen_world_postcot.scope_actor import ScopeRecipeActor
from starVLA.rl.qwen_world_performance.batch_runtime import PerformanceRuntime
from starVLA.rl.qwen_world_performance.contracts import SCHEMA
from starVLA.rl.qwen_world_postcot.contracts import file_sha
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss
from .actor import MultiSceneScopeActor
from .cache import ImmutableFeatureCache
from .conditions import PackedConditions
from .collection import collect_packed
from .objective import source_loss
from .recipe import validate_recipe


class PackedSourceLoss:
    def __init__(self,cfg):self.base=ResearchLoss(cfg);self.cfg=cfg
    def __getattr__(self,key):return getattr(self.base,key)
    def _policy_loss(self,actor,payload):
        if payload.get('packed_loss')is not True:raise ValueError('FAST_PACK cannot consume single-scene old behavior')
        value,rows,_=source_loss(actor,payload['samples'],payload['packed_condition'],loss=self.base,weights=payload['weights'])
        self.latest_rows=[{k:v.detach()for k,v in r.items()}for r in rows]
        keys=set().union(*(r.keys()for r in rows));zero=value.detach().new_zeros(())
        return value,{k:torch.stack([r.get(k,zero)for r in rows]).mean()for k in keys}


class FastPackedRuntime(PerformanceRuntime):
    recipe_validator=staticmethod(validate_recipe)
    def make_actor(self,cfg,args,device):
        actor,a,b=super().make_actor(cfg,args,device)
        if type(actor)is not ScopeRecipeActor:raise ValueError('Unregistered actor before packed construction')
        actor.__class__=MultiSceneScopeActor
        self.conditions=ImmutableFeatureCache(self.conditions.base,capacity=32)
        object.__setattr__(actor,'conditions',self.conditions);actor._loss=PackedSourceLoss(cfg)
        return actor,a,b

    @staticmethod
    def source_files():
        files=PerformanceRuntime.source_files()
        for root in ('starVLA/rl/qwen_world_multiscene','tools/qwen_world_multiscene'):
            files.update({str(p):file_sha(p)for p in Path(root).rglob('*.py')})
        return dict(sorted(files.items()))

    @torch.no_grad()
    def collect(self,actor,condition,token,log,service,generator,*,variant,version,progress=None):
        if variant!='R_CANONICAL':raise ValueError('Only unchanged canonical scorer/source W3')
        self._stage_boundary(actor,version)
        rank=dist.get_rank();world=dist.get_world_size();start=self.training_state['scene_cursor']
        size=self.spec['microbatch_scenes'];b=self.stage['B']//world;samples=[];positions=[];packs=[]
        for base in range(0,b,size):
            items=[self.stream.at(start+world*(base+j),rank)for j in range(size)]
            tokens=[x[0]for x in items];pos=[x[1]for x in items]
            descriptors=[self.conditions.get(t,actor.device)for t in tokens]
            leaves,material=collect_packed(actor,self.conditions,descriptors,tokens,[self.stream.logs[t]for t in tokens],
                service,generator,version=version,pack_index=len(packs),valid_masks=[bool(x.get('valid',True))for x in pos])
            for leaf,position in zip(leaves,pos):leaf.update(stream_position=position,valid_scene=bool(position.get('valid',True)))
            samples.extend(leaves);positions.extend(pos);packs.append(material)
            if progress:progress('scored packed independent groups',sum(x['seconds']['scoring_wait']for x in leaves),size*self.group_size)
        return dict(schema=SCHEMA,samples=samples,rank=rank,group_size=self.group_size,B=self.stage['B'],M=self.spec['inner_updates'],
            window_start_cursor=start,window_positions=positions,packed_groups=packs,packed_execution=copy.deepcopy(self.cfg['research']['packed_execution']),
            cond_bg=dict(schema='world_performance_window_conditions_v1',observations=[s['cond_bg']for s in samples]),
            chain=torch.stack([s['chain']for s in samples]),old_logp=torch.stack([s['old_logp']for s in samples]),
            token=samples[0]['token'],log=samples[0]['log'],tokens=[s['token']for s in samples],logs=[s['log']for s in samples],
            behavior_version=copy.deepcopy(version),records=[r for s in samples for r in s['records']],rewards=torch.cat([s['rewards']for s in samples]),
            protocol_sha256=service.protocol['sha256'],seconds={k:sum(s['seconds'][k]for s in samples)for k in samples[0]['seconds']},
            auxiliary_queries_local=[0,0,0],auxiliary_seconds_local=0.)

    def backward_update(self,model,actor,window,state):
        rows=[];losses=[];errors=[];size=self.spec['microbatch_scenes'];samples=window['samples']
        for offset in range(0,len(samples),size):
            leaves=samples[offset:offset+size]
            with model.no_sync()if offset+size<len(samples)and hasattr(model,'no_sync')else nullcontext():
                failure=None
                try:
                    raw=PackedConditions(self.conditions,selected_layers=actor.scene_provider.selected_layers).pack(
                        [s['cond_bg']for s in leaves],actor.device,valid_masks=[s['valid_scene']for s in leaves])
                    payload=dict(packed_loss=True,samples=leaves,packed_condition=raw,
                        weights=[8/window['valid_global_scenes']if s['valid_scene']else 0. for s in leaves])
                    total,_=model(payload);audit=actor._probability_audit['current']
                    ratio=(audit['source_clamped_logprob'].mean((-1,-2))-torch.cat([s['old_logp']for s in leaves])).exp()
                    error=float((ratio-1).abs().max())
                    if not torch.isfinite(total)or not torch.isfinite(ratio).all():raise FloatingPointError('Nonfinite coherent packed loss')
                    if state['next_inner']==0 and error>1e-4:raise ValueError('Coherent packed fresh ratio failed')
                except Exception:failure=traceback.format_exc()
                if failure and checkpoint.fatal_device_error(failure):raise RuntimeError(failure)
                checkpoint.check_errors(failure);total.backward()
            losses.append(total.detach());rows.extend(actor._loss.latest_rows);errors.append(error)
        zero=losses[0].new_zeros(());keys=set().union(*(r.keys()for r in rows))
        metrics={k:torch.stack([r.get(k,zero)for r in rows]).mean()for k in keys}
        metrics.update({k:zero+v for k,v in window['quality_counts_global'].items()})
        metrics.update(valid_scene_exposures=zero+state['valid_scene_exposures'],
            exposure_epoch=zero+state['valid_scene_exposures']/self.spec['population'],global_distinct_scenes=zero+window['B'],
            fresh_PDM_queries=zero+window['B']*self.group_size,packed_scenes=zero+size)
        return sum(losses,zero),metrics,dict(fresh_error=max(errors),source_clamp_hit_rate=float(audit['source_clamp_hit_rate']))

    def recipe_identity(self,cfg,args,condition_sha):
        value=super().recipe_identity(cfg,args,condition_sha)
        value['packed_execution']=copy.deepcopy(cfg['research']['packed_execution'])
        value['native_resume']='Only exact own packed numerical recipe; never migrate v1 single-scene sampling'
        return value

    def exploration_summary(self,window):
        value=super().exploration_summary(window)
        value.update(packed_execution=window['packed_execution'],packed_groups=len(window['packed_groups']),frozen_cache=self.conditions.statistics())
        return value

    def validate_boundary(self,*args,**kwargs):
        from .recovery import validate_boundary
        return validate_boundary(*args,**kwargs)

    def validate_ranks(self,summaries):
        from .recovery import validate_ranks
        return validate_ranks(summaries)
