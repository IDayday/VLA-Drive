"""Fresh IL70 SafetyEP + independently sampled online success-memory CFM."""
import copy
import traceback

from starVLA.rl.simwam_recipe import checkpoint
from .safety_efficiency_entry import SafetyEfficiencyRuntimeMixin
from .safety_efficiency import validate_safety_efficiency_boundary,validate_safety_efficiency_ranks
from .success_memory import (
    ARM,CONFIG,SCHEMA,SuccessMemoryLoss,FrozenFeatureLRU,validate_recipe,
    new_memory,attach_history,memory_summary,rank_id,
)


class SuccessMemoryRuntimeMixin:
    def load_recipe(self,args):
        cfg=validate_recipe(super().load_recipe(args))
        if self.parent['seal']['updates']!=70000 or self.parent['model_sha256']!='7dda5b14b1ea67fc631b06c7fa336950fa07a0b84c969f4e714bdeb5b5436d40':
            raise ValueError('P1 starts fresh from the same pinned complete IL70,never old RL teacher weights')
        return cfg
    def make_actor(self,cfg,args,device):
        actor,oldcfg,sft=super().make_actor(cfg,args,device)
        actor._loss=SuccessMemoryLoss(cfg)
        self.conditions=FrozenFeatureLRU(self.conditions)
        object.__setattr__(actor,'conditions',self.conditions)
        self.allowed=dict(self.conditions.data.logs)
        return actor,oldcfg,sft
    def bind_training_state(self,state,rollout,identity):
        """Trainer calls after fresh/restore and before Invocation,all ranks."""
        error=None
        try:
            if 'success_memory' not in state:
                if state['updates']!=0 or state['collections']!=0 or rollout is not None:
                    raise ValueError('Cannot resume without complete success bank/RNG/pendingaux')
                state['success_memory']=new_memory(identity['sha256'],rank_id())
            memory_summary(state,rollout,identity['sha256'],rank_id(),self.allowed,full=True)
            self.success_state=state;self.success_memory=state['success_memory'];self.pending_success_rollout=rollout
        except Exception:error=traceback.format_exc()
        checkpoint.check_errors(error)
    def finalize_advantages(self,rollout):
        super().finalize_advantages(rollout)
        error=None
        try:attach_history(self.success_memory,rollout,self.conditions,self.actor.device,self.allowed)
        except Exception:error=traceback.format_exc()
        checkpoint.check_errors(error)
        self.pending_success_rollout=rollout
        return rollout
    def public_training_state(self,state):
        # Native checkpoint receives the FULL state;JSON progress/result gets
        # counters only,never1024teacher tensors or privateRNG bytes.
        public={k:v for k,v in state.items() if k!='success_memory'}
        pending=self.pending_success_rollout if state['phase'] in ('ROLLOUT_COMMITTED','INNER_COMMITTED') else None
        try:public['success_memory_summary']=memory_summary(state,pending,self.success_memory['run'],rank_id(),self.allowed)
        except Exception as exc:
            # Failure evidence must survive a partially finalized transaction;
            # native checkpoint validation still rejects the full bad state.
            public['success_memory_summary']=dict(status='UNCOMMITTED_DIAGNOSTIC_ONLY',error=repr(exc))
        return public
    def validate_boundary(self,state,rollout,identity,optimizer=None,scheduler=None,*,legacy=False):
        validate_recipe(identity['recipe'])
        summary=validate_safety_efficiency_boundary(state,rollout,identity,optimizer,scheduler,legacy=legacy)
        summary['success_memory']=memory_summary(state,rollout,identity['sha256'],rank_id(),self.allowed)
        return summary
    @staticmethod
    def validate_ranks(summaries):
        validate_safety_efficiency_ranks(summaries)
        for rank,s in enumerate(summaries):
            m=s.get('success_memory',{})
            if m.get('schema')!=SCHEMA or m.get('rank')!=rank or m.get('collections')!=s['collections']:
                raise ValueError('Eight rank-local online memories must preserve their own rank/cursor')
        return summaries
    def recipe_identity(self,cfg,args,condition_sha):
        value=super().recipe_identity(cfg,args,condition_sha)
        value['success_memory_protocol']=dict(config=copy.deepcopy(CONFIG),initial_bank='empty,freshsameIL70',teacher_source='this run,thisrank,officialTRAING8 scored endpoints only',
            admission='NC=DAC=TTC=Comfort1,EP>=.5;bestEP then rawprogress,one unique teacher/token',
            history='uniform independent privateRNG sampling BEFORE fresh winner insertion,FIFO1024;never same-current-token lookup',
            objective='one positive CFM velocityMSE,target clean endpoint,1MC sigmaU[.2,.8],coefficient.1',
            budget='one historical scene/rank every4collections,same saved material reused4inners;fresh saturation does not disable historical aux',
            state='full bank/privateRNG/exposure counters in native rank state,pending selected teacher/condition/noise in rollout',
            observation='current legal historical-scene images/ego/nav only;Qwen features/reference cache2;current Memory rebuilt each inner',
            base_unchanged='freshuniform/G8/W3/sourcePPO/SafetyEP/IL70anchor/constantLR/mainRNG',
            coverage='EP gates define bank coverage,not whether necessary stopping is a driving error',
            gradient_probe='optional live PG/anchor/aux derivatives;scalarloss magnitudes are not gradient attribution')
        return value


def main():
    from scripts.analysis.simwam_recipe.train import main as train
    from starVLA.rl.qwen_world_research.runtime import ResearchRuntime
    class SuccessMemoryRuntime(SuccessMemoryRuntimeMixin,SafetyEfficiencyRuntimeMixin,ResearchRuntime):pass
    train(SuccessMemoryRuntime())


if __name__=='__main__':main()
