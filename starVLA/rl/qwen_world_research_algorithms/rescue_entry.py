"""Matched W2/SafetyEP warmstarted replay vs verified-positive CFM pilot."""
import copy
import json
import traceback
from pathlib import Path

from starVLA.rl.qwen_sourceflow.collection import collect as source_collect
from starVLA.rl.simwam_recipe import checkpoint
from .lastdet_entry import LASTDET_PROTOCOL
from .policy import ResearchLoss
from .safety_efficiency import finalize_safety_efficiency_advantages, STRUCTURED_PROTOCOL
from .rescue_auxiliary import (
    AUX_PROTOCOL, RescueTeacherBank, RescueCFMLoss, attach_rescue_material,
    validate_rescue_recipe, validate_rescue_boundary, validate_rescue_ranks, file_sha,
)


class RescueRuntimeMixin:
    validate_ranks = staticmethod(validate_rescue_ranks)

    def load_recipe(self, args):
        cfg = validate_rescue_recipe(super().load_recipe(args))
        r = cfg['research']; manifest = json.loads(Path(r['rescue_manifest']).read_text())
        full = Path(manifest['full_population_manifest'])
        if file_sha(full) != manifest['full_population_manifest_sha256']:
            raise ValueError('Rescue FULL103k eligible population SHA changed')
        eligible = json.loads(full.read_text())['tokens']
        if len(eligible) != manifest['full_population_count'] or len(set(eligible)) != len(eligible):
            raise ValueError('Rescue FULL eligible population count changed')
        self.rescue_bank = RescueTeacherBank(r['rescue_manifest'], r['rescue_manifest_sha256'], eligible)
        return cfg

    def make_actor(self, cfg, args, device):
        actor, old_cfg, sft = super().make_actor(cfg, args, device)
        actor._loss = RescueCFMLoss(cfg) if cfg['research']['arm'] == 'PPO_RESCUE_CFM' else ResearchLoss(cfg)
        return actor, old_cfg, sft

    def collect(self, actor, condition, token, log, service, generator, *, variant, version, progress=None):
        validate_rescue_recipe(self.cfg)
        return source_collect(actor, condition, token, log, service, generator,
            variant=variant, version=version, progress=progress, recipe=self.cfg)

    def finalize_advantages(self, rollout):
        if 'advantages' in rollout or 'rescue_auxiliary' in rollout:
            raise ValueError('Finalize fresh SafetyEP and rescue material once per collection')
        finalize_safety_efficiency_advantages(rollout)
        error = None
        try:
            attach_rescue_material(self.actor, rollout, self.rescue_bank,
                enabled=self.cfg['research']['arm'] == 'PPO_RESCUE_CFM')
        except Exception:
            error = traceback.format_exc()
        checkpoint.check_errors(error)
        return rollout

    def validate_boundary(self, state, rollout, identity, optimizer=None, scheduler=None, *, legacy=False):
        return validate_rescue_boundary(state, rollout, identity, optimizer, scheduler,
            bank=self.rescue_bank, legacy=legacy)

    def recipe_identity(self, cfg, args, condition_sha):
        value = super().recipe_identity(cfg, args, condition_sha)
        window = copy.deepcopy(LASTDET_PROTOCOL)
        window['policy_gradient'] = 'fresh canonical SafetyEP metric-conditioned advantages; source PPO score over W2 only, never step9'
        value.update(sampler_window_protocol=window, structured_advantage_protocol=copy.deepcopy(STRUCTURED_PROTOCOL),
            rescue_auxiliary_protocol=copy.deepcopy(AUX_PROTOCOL) if cfg['research']['arm'] == 'PPO_RESCUE_CFM' else None,
            rescue_teacher_manifest_sha256=self.rescue_bank.sha256,
            rescue_comparison='matched Safety10k weights-only warmstart/phase update0, sampler, W2 and SafetyEP credit; CFM adds only positive teacher loss')
        return value


def main():
    from scripts.analysis.simwam_recipe.train import main as train
    from starVLA.rl.qwen_world_research.runtime import ResearchRuntime
    from .warmstart import WeightsOnlyWarmstartMixin

    class RescueRuntime(WeightsOnlyWarmstartMixin, RescueRuntimeMixin, ResearchRuntime):
        pass

    train(RescueRuntime())


if __name__ == '__main__':
    main()
