"""Registered milestone evaluations; reuse the capacity-aware single-owner queue."""
from starVLA.rl.fm_transfer.source_process import MODES, MILESTONES
from scripts.analysis.fm_transfer.milestone_controller import main


class ProcessEvaluations:
    @staticmethod
    def jobs(root):
        # Step0 is the authenticated existing B100k SFT; never resample a new base.
        for update in MILESTONES[1:]:
            for mode in MODES:
                for cohort in ('DEV','TRAIN_DIAGNOSTIC'):
                    yield dict(variant='B_SOURCE_ACTION_DIT',process_mode=mode,stage='RL',update=update,cohort=cohort,
                        checkpoint=str(root/'runs'/f'{mode}_seed42_v1'/'checkpoints'/f'update{update:06d}_inner0'),
                        export=str(root/'exports'/mode/f'RL_update{update:06d}'),
                        output=str(root/'evaluations'/f'{mode}_RL_update{update:06d}_{cohort}'))

    @staticmethod
    def reference_for(root,job):
        return '/mnt/project/fm-transfer-20261016/evaluations/B_SOURCE_ACTION_DIT_SFT_update100000_'+job['cohort']


if __name__=='__main__':main(runtime=ProcessEvaluations())
