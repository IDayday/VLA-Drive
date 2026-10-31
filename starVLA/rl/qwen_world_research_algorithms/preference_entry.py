"""Independent canonical PPO plus offline TRAIN pair CFM auxiliary entry."""
import copy
import json
import traceback
from pathlib import Path

from .preference_auxiliary import (
    AUX_PROTOCOL, PreferenceBank, PreferenceAuxiliaryLoss, attach_preference_material,
    validate_preference_recipe, validate_preference_boundary, validate_preference_ranks,
)
from starVLA.rl.simwam_recipe.collection import finalize_advantages as source_finalize
from starVLA.rl.simwam_recipe import checkpoint


class PreferenceRuntimeMixin:
    validate_ranks = staticmethod(validate_preference_ranks)

    def load_recipe(self, args):
        cfg = validate_preference_recipe(super().load_recipe(args))
        r = cfg['research']
        # The scene sampler manifest is the eligible FULL103k universe, not a
        # Test failure list or an unscored trajectory proposal source.
        eligible = json.loads(Path(args.manifest).read_text())['tokens']
        self.preference_bank = PreferenceBank(r['preference_manifest'], r['preference_manifest_sha256'], eligible)
        return cfg

    def make_actor(self, cfg, args, device):
        actor, old_cfg, sft = super().make_actor(cfg, args, device)
        actor._loss = PreferenceAuxiliaryLoss(cfg)
        return actor, old_cfg, sft

    def finalize_advantages(self, rollout):
        if 'advantages' in rollout or 'preference_auxiliary' in rollout:
            raise ValueError('Finalize canonical PPO and auxiliary material once per collection')
        source_finalize(rollout)
        error = None
        try:
            attach_preference_material(self.actor, rollout, self.preference_bank)
        except Exception:
            error = traceback.format_exc()
        # Explicit CPU control plane: a failed extra reference forward on one
        # rank must not let other ranks commit a partially prepared collection.
        checkpoint.check_errors(error)
        return rollout

    def validate_boundary(self, state, rollout, identity, optimizer=None, scheduler=None, *, legacy=False):
        return validate_preference_boundary(state, rollout, identity, optimizer, scheduler,
            bank=self.preference_bank, legacy=legacy)

    def recipe_identity(self, cfg, args, condition_sha):
        value = super().recipe_identity(cfg, args, condition_sha)
        value['preference_auxiliary_protocol'] = copy.deepcopy(AUX_PROTOCOL)
        return value


def main():
    from scripts.analysis.simwam_recipe.train import main as train
    from starVLA.rl.qwen_world_research.runtime import ResearchRuntime

    class PreferenceRuntime(PreferenceRuntimeMixin, ResearchRuntime):
        pass

    train(PreferenceRuntime())


if __name__ == '__main__':
    main()
