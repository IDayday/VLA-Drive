"""Independent PPO_SAFE_COMFORT candidate; canonical entry stays unchanged."""
import copy
from .safe_comfort import (
    COMFORT_SCHEMA, COMFORT_PROTOCOL, finalize_safe_comfort_advantages,
    validate_comfort_boundary, validate_comfort_ranks,
)


class SafeComfortRuntimeMixin:
    finalize_advantages=staticmethod(finalize_safe_comfort_advantages)
    validate_boundary=staticmethod(validate_comfort_boundary)
    validate_ranks=staticmethod(validate_comfort_ranks)

    def load_recipe(self,args):
        cfg=super().load_recipe(args)
        r=cfg['research']
        if (r.get('reward_transform') != COMFORT_SCHEMA or r.get('policy_objective') != 'ppo'
                or r.get('advantage_normalization') is not None
                or r.get('data_policy') != 'uniform_full' or cfg['variant'] != 'R_CANONICAL'):
            raise ValueError('Register a separate PPO/full/RAW-batchstd safe-comfort arm')
        return cfg

    def recipe_identity(self,cfg,args,condition_sha):
        value=super().recipe_identity(cfg,args,condition_sha)
        value['reward_transform_protocol']=copy.deepcopy(COMFORT_PROTOCOL)
        value['training_reward_is_canonical_PDMS']=False
        return value


def main():
    from scripts.analysis.simwam_recipe.train import main as train
    from starVLA.rl.qwen_world_research.runtime import ResearchRuntime

    class SafeComfortRuntime(SafeComfortRuntimeMixin,ResearchRuntime):
        pass

    train(SafeComfortRuntime())


if __name__=='__main__':main()
