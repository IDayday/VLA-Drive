"""Thin S6 extension; source sampler/Trainer/gradient calibration unchanged."""
import copy
from pathlib import Path
import torch
import torch.distributed as dist
from starVLA.rl.qwen_world_verified_quality.runtime import VerifiedQualityRuntime,VerifiedQualityLoss
from starVLA.rl.qwen_world_performance import nominal_improvement as N
from starVLA.rl.qwen_world_performance.ode_target import probe_nominal
from starVLA.rl.qwen_world_performance.adaptive_group import collect_extra_if_needed
from starVLA.rl.qwen_world_postcot.contracts import file_sha
from starVLA.rl.qwen_world_safety_exploration.retention_material import sealed
from .recipe import validate_recipe,PROTOCOL,PROTOCOL_SHA256
from .gt_reader import GTReader
from . import material as M

COUNTER_SCHEMA='world_structured_progress_query_ledger_v1'


def new_state(rank):
    return dict(schema=COUNTER_SCHEMA,protocol_sha256=PROTOCOL_SHA256,rank=rank,ledger=[])


class StructuredProgressLoss(VerifiedQualityLoss):
    def _policy_loss(self,model,rollout):
        value,metrics=super()._policy_loss(model,rollout)
        p=rollout['nominal_improvement'];pool=p['structured_progress'];z=value.detach().new_zeros(())
        chosen=p['enabled'] and str(p['teacher']['origin']).startswith('structured_progress_')
        metrics.update(structured_progress_teacher_enabled=z+int(chosen),structured_progress_triggered=z+int(pool['triggered']),
            structured_progress_scored_candidates=z+pool['queries'])
        return value,metrics


class StructuredProgressRuntime(VerifiedQualityRuntime):
    recipe_validator=staticmethod(validate_recipe)

    @staticmethod
    def source_files():
        result=VerifiedQualityRuntime.source_files()
        for folder in ('starVLA/rl/qwen_world_structured_progress','tools/qwen_world_structured_progress'):
            for path in Path(folder).rglob('*.py'):result[str(path)]=file_sha(path)
        return dict(sorted(result.items()))

    def make_actor(self,cfg,args,device):
        actor,a,b=super().make_actor(cfg,args,device)
        self.structured_reader=GTReader(self.conditions.data);M.bind_reader(self.structured_reader)
        actor._loss=StructuredProgressLoss(cfg)
        return actor,a,b

    def bind_training_state(self,state,rollout,identity):
        super().bind_training_state(state,rollout,identity)
        if 'structured_progress' not in state:
            if state['updates'] or state['collections']:raise ValueError('S7 cannot import an RL state or reconstruct its query ledger')
            state['structured_progress']=new_state(dist.get_rank())
        value=state['structured_progress']
        if value['schema']!=COUNTER_SCHEMA or value['protocol_sha256']!=PROTOCOL_SHA256 or value['rank']!=dist.get_rank():
            raise ValueError('Wrong S7 rank-private ledger/protocol')

    @torch.no_grad()
    def _attach_auxiliary(self,actor,leaf,service,rank):
        audit=actor._probability_audit
        try:
            nominal=extra=None
            if leaf.get('valid_scene',True):
                nominal=probe_nominal(actor,self.conditions,leaf['token'],leaf['log'],service,leaf['behavior_version'])
                extra=collect_extra_if_needed(actor,leaf,nominal,service,rank=rank)
            pool=M.collect_pool(leaf,nominal,extra,service,self.structured_reader,rank)
            N.attach_material(actor,leaf,self.conditions,service,
                exposure_epoch=self.training_state['valid_scene_exposures']/self.spec['population'],rank=rank,
                config=N.QUALITY_CONFIG,nominal=nominal,extra=extra,
                teacher_selector=lambda a,b,c,d:M.select_teacher(a,b,c,d,pool))
            p={k:v for k,v in leaf['nominal_improvement'].items() if k!='material_sha256'}
            p['structured_progress']=pool;leaf['nominal_improvement']=sealed(p,'material_sha256')
        finally:actor._probability_audit=audit

    @torch.no_grad()
    def finalize_advantages(self,window):
        super().finalize_advantages(window)
        local=M.local_counts(window);values=torch.tensor(list(local.values()),device=window['rewards'].device,dtype=torch.int64)
        dist.all_reduce(values)
        window['structured_progress_counts_local']=local
        window['structured_progress_counts_global']=dict(zip(local,values.cpu().tolist()))
        window['structured_progress_protocol_sha256']=PROTOCOL_SHA256
        return window

    def commit_collection_accounting(self,window,state,world):
        super().commit_collection_accounting(window,state,world)
        state['structured_progress']['ledger'].append(dict(collection=state['collections'],
            local=copy.deepcopy(window['structured_progress_counts_local']),global_counts=copy.deepcopy(window['structured_progress_counts_global'])))

    def exploration_summary(self,window):
        value=super().exploration_summary(window)
        value['structured_progress']=copy.deepcopy(window['structured_progress_counts_global'])
        value['structured_progress_protocol_sha256']=PROTOCOL_SHA256
        return value

    def backward_update(self,*args,**kwargs):
        value,metrics,diagnostics=super().backward_update(*args,**kwargs)
        window=args[2] if len(args)>2 else kwargs['window']
        metrics.update({f'structured_progress_global_{k}':value.new_tensor(v) for k,v in window['structured_progress_counts_global'].items()})
        return value,metrics,diagnostics

    def recipe_identity(self,cfg,args,condition_sha):
        value=super().recipe_identity(cfg,args,condition_sha)
        value.update(structured_progress_protocol=copy.deepcopy(PROTOCOL),structured_progress_protocol_sha256=PROTOCOL_SHA256,
            structured_GT_source_identity=copy.deepcopy(self.structured_reader.identity))
        return value

    def validate_boundary(self,*args,**kwargs):
        from .recovery import validate_boundary
        return validate_boundary(*args,**kwargs)

    def validate_ranks(self,summaries):
        from .recovery import validate_ranks
        return validate_ranks(summaries)

    def public_training_state(self,state):
        value=super().public_training_state(state);own=state.get('structured_progress',{})
        value['structured_progress']=dict(protocol_sha256=PROTOCOL_SHA256,committed_collections=len(own.get('ledger',[])),
            global_queries=sum(x['global_counts']['queries'] for x in own.get('ledger',[])),material_stored_in_native_checkpoint=True)
        return value
