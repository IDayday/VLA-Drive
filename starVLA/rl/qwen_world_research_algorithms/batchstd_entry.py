"""Separately identified PPO arm with centered-global-batch advantage std.

Run through torchrun with the ordinary research train arguments, replacing
the module with `starVLA.rl.qwen_world_research_algorithms.batchstd_entry`.
No module globals are patched. The source engine accepts explicit runtime
hooks and passes this runtime's validators to native save and restore.
"""
from .batchstd import (
    NORMALIZATION_SCHEMA, finalize_centered_batch_advantages,
    validate_centered_batch_boundary, validate_centered_batch_ranks,
)


class CenteredBatchRuntimeMixin:
    finalize_advantages = staticmethod(finalize_centered_batch_advantages)
    validate_boundary = staticmethod(validate_centered_batch_boundary)
    validate_ranks = staticmethod(validate_centered_batch_ranks)

    def load_recipe(self, args):
        cfg = super().load_recipe(args)
        r = cfg['research']
        if (r.get('advantage_normalization') != NORMALIZATION_SCHEMA
                or r.get('policy_objective') != 'ppo' or r.get('data_policy') != 'uniform_full'):
            raise ValueError('Register a separate PPO/full-data centered-batchstd arm')
        return cfg

    def recipe_identity(self, cfg, args, condition_sha):
        value = super().recipe_identity(cfg, args, condition_sha)
        value['advantage_normalization_protocol'] = dict(
            schema=NORMALIZATION_SCHEMA,
            numerator='r_gi - group_mean(r_g)',
            denominator='population_std of all 64 group-centered rewards + 1e-8',
            population='8 fresh rank scenes x 8 candidates; includes tied groups',
            finalization='once after successful scoring, before four fixed inner updates',
            global_std_field='raw reward population std, preserved for legacy exploration readout',
            normalization_std_field='centered advantage population std actually used',
            paper='https://arxiv.org/html/2501.03262v8#S3.SS1',
            scope='normalization-only RF++-Baseline ablation; source PPO ratio and beta50 IL anchor retained')
        return value


def main():
    from scripts.analysis.simwam_recipe.train import main as train
    from starVLA.rl.qwen_world_research.runtime import ResearchRuntime

    class CenteredBatchRuntime(CenteredBatchRuntimeMixin, ResearchRuntime):
        pass

    train(CenteredBatchRuntime())


if __name__ == '__main__':
    main()
