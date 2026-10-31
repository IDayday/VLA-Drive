"""Fresh-IL70 FPO per-MC ordinary PPO, canonical SafetyEP and source W3 anchor."""
import copy
import json
import os
import traceback
from pathlib import Path

from .fpo_training import (
    PROTOCOL,FPOTrainingLoss,validate_recipe,attach_material,validate_boundary,validate_ranks,scale_diagnostic,
)
from .safety_efficiency import finalize_safety_efficiency_advantages,STRUCTURED_PROTOCOL
from starVLA.rl.simwam_recipe import checkpoint


class FPORuntimeMixin:
    validate_ranks=staticmethod(validate_ranks)

    def load_recipe(self,args):
        return validate_recipe(super().load_recipe(args))

    def make_actor(self,cfg,args,device):
        actor,old_cfg,sft=super().make_actor(cfg,args,device)
        actor._loss=FPOTrainingLoss(cfg)
        return actor,old_cfg,sft

    def finalize_advantages(self,rollout):
        if 'advantages' in rollout or 'fpo' in rollout:raise ValueError('Finalize fresh SafetyA/FPO cache once')
        finalize_safety_efficiency_advantages(rollout)
        error=None
        try:
            rank=int(os.environ.get('RANK',0))
            attach_material(self.actor,rollout,rank=rank)
            if rollout['behavior_version']['update']==0:
                p=Path(self.args.output)/f'initial_FPO_scale_rank{rank}.json'
                if not p.exists():
                    tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
                    tmp.write_text(json.dumps(scale_diagnostic(rollout),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
                    os.replace(tmp,p)
        except Exception:error=traceback.format_exc()
        checkpoint.check_errors(error)
        return rollout

    def validate_boundary(self,state,rollout,identity,optimizer=None,scheduler=None,*,legacy=False):
        s=validate_boundary(state,rollout,identity,optimizer,scheduler,legacy=legacy)
        # First-inner freshness must also be enforced after native restore.
        # A resumed inner1..3 intentionally compares changedtheta to oldCFM.
        if s['pending']:
            expect=rollout['fpo']['material_sha256'] if state['next_inner']==0 else None
            object.__setattr__(self.actor,'_fpo_expect_fresh_material',expect)
        return s

    def recipe_identity(self,cfg,args,condition_sha):
        identity=super().recipe_identity(cfg,args,condition_sha)
        identity.update(fpo_protocol=copy.deepcopy(PROTOCOL),structured_advantage_protocol=copy.deepcopy(STRUCTURED_PROTOCOL))
        return identity


def main():
    from scripts.analysis.simwam_recipe.train import main as train
    from starVLA.rl.qwen_world_research.runtime import ResearchRuntime
    class FPORuntime(FPORuntimeMixin,ResearchRuntime):pass
    train(FPORuntime())


if __name__=='__main__':main()
