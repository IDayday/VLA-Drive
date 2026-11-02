"""Explicit multi-scene Scope actor; original B1 path is inherited unchanged."""
import torch
from starVLA.rl.qwen_world_postcot.scope_actor import ScopeRecipeActor
from .conditions import SCHEMA


class MultiSceneScopeActor(ScopeRecipeActor):
    def materialize(self, condition):
        if condition.get('schema')!=SCHEMA:return super().materialize(condition)
        expected={'schema','scene_tokens','groups','scene_index','frozen_features','base_memory','base_valid','base_ego','context','valid','ego'}
        n=len(condition['scene_tokens']);g=condition['groups'];index=condition['scene_index']
        if (set(condition)!=expected or n not in (2,4) or len(set(condition['scene_tokens']))!=n
                or g not in (8,16,32) or condition['base_memory'].shape[0]!=n
                or condition['base_ego'].shape!=(n,self.action_expert.ego_dim)
                or condition['base_valid'].shape!=condition['base_memory'].shape[:2]
                or not torch.equal(index,torch.arange(n,device=self.device).repeat_interleave(g))
                or condition['context'].shape[0]!=n*g):
            raise ValueError('Invalid genuine multi-scene Scope carrier/index')
        return condition

    def _prepare(self, condition, use_ref):
        if condition.get('schema')!=SCHEMA:return super()._prepare(condition,use_ref)
        key='reference' if use_ref else 'current'
        if self._prepared_scope is not None and key in self._prepared_scope:return self._prepared_scope[key]
        head=self.ref_action_expert if use_ref else self.action_expert
        with torch.autocast(self.device.type,dtype=torch.bfloat16):
            if use_ref:
                memory,valid=condition['base_memory'].detach(),condition['base_valid'].detach()
            else:
                value=self.predictive_provider(self.scene_provider(condition['frozen_features'],condition['base_ego']))
                memory,valid=value.actor_memory,value.actor_valid
            prepared=head.prepare_context(memory,valid,condition['base_ego'])
        if any(x.dtype!=torch.bfloat16 for pair in prepared.layer_kv for x in pair):
            raise ValueError('Packed current/reference KV execution dtype changed')
        self.current_memory_builds+=int(not use_ref)
        self.prepared_context_builds[key]+=1
        self.prepared_context_source_batches.append(len(condition['scene_tokens']))
        del self.prepared_context_source_batches[:-64]
        prepared=prepared.index_select(condition['scene_index'])
        if self._prepared_scope is not None:self._prepared_scope[key]=prepared
        return prepared
