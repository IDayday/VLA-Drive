"""Registered World PPO_LASTDET W2 collector/native recovery entry only."""
import copy

from starVLA.rl.qwen_sourceflow.collection import collect as source_collect
from starVLA.rl.simwam_recipe.contracts import LASTDET_SCHEMA,registered_stochastic_steps_for_recipe
from starVLA.rl.simwam_recipe.recovery_state import validate_boundary as source_boundary,validate_ranks as source_ranks

LASTDET_PROTOCOL=dict(schema=LASTDET_SCHEMA,stochastic_steps=[7,8],deterministic_final_step=9,
    group_size=8,probability_shape=[8,2],native_chain_shape=[8,11,8,3],eta=.1,
    policy_gradient='source canonical RAW-std PPO; Gaussian score and step mean over W2 only, never step9',
    anchor='source frozen-IL mean penalty averaged over W2[7,8], beta50 retained; step9 anchor is removed and remaining steps reweighted relative to W3',
    generator='source private generator initial draw +2 innovations per collection; seed42 preserved, later root stream differs from W3 consumption',
    scope='same LoRA/Memory closure and deployment; no extra W3 anchor forward')


def require_lastdet(cfg):
    if registered_stochastic_steps_for_recipe(cfg)!=[7,8]:
        raise ValueError('Use this entry only for the separately registered PPO_LASTDET')
    return cfg


class LastDetRuntimeMixin:
    validate_ranks=staticmethod(source_ranks)

    def load_recipe(self,args):
        return require_lastdet(super().load_recipe(args))

    def collect(self,actor,condition,token,log,service,generator,*,variant,version,progress=None):
        require_lastdet(self.cfg)
        return source_collect(actor,condition,token,log,service,generator,
            variant=variant,version=version,progress=progress,recipe=self.cfg)

    def validate_boundary(self,state,rollout,identity,optimizer=None,scheduler=None,*,legacy=False):
        require_lastdet(identity['recipe'])
        return source_boundary(state,rollout,identity,optimizer,scheduler,legacy=legacy)

    def recipe_identity(self,cfg,args,condition_sha):
        value=super().recipe_identity(cfg,args,condition_sha)
        value['sampler_window_protocol']=copy.deepcopy(LASTDET_PROTOCOL)
        return value


def main():
    from scripts.analysis.simwam_recipe.train import main as train
    from starVLA.rl.qwen_world_research.runtime import ResearchRuntime

    class LastDetRuntime(LastDetRuntimeMixin,ResearchRuntime):
        pass

    train(LastDetRuntime())


if __name__=='__main__':main()
