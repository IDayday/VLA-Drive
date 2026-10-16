"""Original raw-observation evaluation, with separate native and shift5 readouts."""
import json
from pathlib import Path
import torch
from starVLA.rl.fm_transfer.contracts import require_dispatch
from starVLA.rl.simwam_recipe.contracts import file_sha, fingerprint
from scripts.analysis.simwam_recipe.evaluate_policy import main, shifted_ode


class TransferEvaluation:
    require_dispatch=staticmethod(require_dispatch)

    @staticmethod
    def validate_export(export,args):
        value=json.loads((export/'export.json').read_text())
        payload=dict(value);sha=payload.pop('sha256')
        if value['schema']!='fm_transfer_original_entry_export_v1' or fingerprint(payload)!=sha:
            raise ValueError('Not an authenticated new-base export')
        if value['weight_sha256']!=args.export_sha256 or value['config_sha256']!=file_sha(export/'config.yaml'):
            raise ValueError('Export/config changed')
        if value['qualification'] and not args.max_scenes:
            raise ValueError('Qualification exports cannot enter full formal evaluation')
        if args.cohort=='NAVTEST':
            selection=json.loads(Path(args.final_lock).read_text())
            if selection.get('transfer_evaluation_sha256')!=file_sha(__file__):
                raise ValueError('New architecture evaluation must be locked before Navtest')

    @staticmethod
    def shifted_ode(head,condition,root):
        if not hasattr(head,'sample_shift5'):return shifted_ode(head,condition,root)
        with torch.autocast('cuda',enabled=False):
            actions=head.sample_shift5(root,condition)
        times,_=head.policy.scheduler.build_inference_schedule(10,root.device,torch.float32)
        # Compatibility field: these are continuous source timesteps, not
        # integer DDP buckets. Its meaning is explicit in the evaluation ID.
        return actions,times.tolist()

    @staticmethod
    def identity_fields(export,model):
        info=json.loads((export/'export.json').read_text())
        return dict(schema='fm_transfer_raw_observation_dual_grid_v1',new_SFT=info['new_SFT'],
            architecture=info['architecture'],stage=info['stage'],
            actor='original VLAAgent observation entry; new supervised velocity head',
            shift5_time_field=('source continuous sigma*1000' if info['architecture']=='source_action_dit' else 'DDP integer t buckets'),
            transfer_evaluation_sha256=file_sha(__file__),
            export_receipt_sha256=file_sha(export/'export.json'))


if __name__=='__main__':main(runtime=TransferEvaluation())
