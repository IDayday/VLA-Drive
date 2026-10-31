"""D own-best-IL action-only JaT-inspired adaptation, separate from ABC credit."""
import copy
from .d_objective import (
    PROTOCOL,DActionJaTLoss,validate_recipe,finalize_advantages,
    validate_boundary_summary,validate_ranks,
)


class DRuntimeMixin:
    validate_ranks=staticmethod(validate_ranks)

    def load_recipe(self,args):return validate_recipe(super().load_recipe(args))

    def make_actor(self,cfg,args,device):
        actor,oldcfg,sft=super().make_actor(cfg,args,device)
        actor._loss=DActionJaTLoss(cfg)
        return actor,oldcfg,sft

    def finalize_advantages(self,rollout):
        finalize_advantages(rollout)
        object.__setattr__(self.actor,'_d_expect_fresh_material',rollout['d_gaussian']['sha256'])
        return rollout

    def recipe_identity(self,cfg,args,condition_sha):
        identity=super().recipe_identity(cfg,args,condition_sha)
        identity.update(d_objective_protocol=copy.deepcopy(PROTOCOL),
            policy_ratio='D joint24-dimensional Gaussian transition ratio, unclipped innovation, JaT algebra/sourceW3 kernel adaptation',
            reference='own selectedD IL native GaussianKL.001; no beta50 physicalanchor; no CoT RL')
        return identity

    def validate_boundary(self,state,rollout,identity,optimizer=None,scheduler=None,*,legacy=False):
        summary=super().validate_boundary(state,rollout,identity,optimizer,scheduler,legacy=legacy)
        summary=validate_boundary_summary(summary,state,rollout,identity)
        if summary['pending']:
            expected=rollout['d_gaussian']['sha256'] if state['next_inner']==0 else None
            object.__setattr__(self.actor,'_d_expect_fresh_material',expected)
        return summary


def main():
    from .runtime import ABCDBestRuntime
    from tools.qwen_world_abcd_research.train import main as train
    class DRuntime(DRuntimeMixin,ABCDBestRuntime):pass
    train(DRuntime())


if __name__=='__main__':main()
