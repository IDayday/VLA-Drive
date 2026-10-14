"""Capture the selected endpoint, then verify its exported bundle in a new process.

Four fixed stage_val observations in their original image batch layouts; no
training, endpoint choice, metric simulation or test-set access is performed.
"""
import argparse
import json
from pathlib import Path
import torch
from iqe.contracts import require
from iqe.io import atomic_json,atomic_torch,file_hash,read_json
from iqe.data.sources import load_observation
from iqe.robust_scorer.qualification import compare_outputs
from iqe.specialist_study.binding import scenes_by_role
from iqe.specialist_study.context_batch_layout import OriginalBatchReplay
from iqe.specialist_study.training import runtime
from .bundle import assemble,payload_for,load
from .calibration import CalibrationReader
from .evaluation import selected
from .full import load_config,resource_check


@torch.no_grad()
def run(config,mode,device):
    require(mode in ('capture','verify'),'explicit new-process bundle check mode')
    resource_check(config);device=runtime(device);source=config.source()
    bundle=config.root/'research_bundle/bundle.pt'
    require(file_hash(bundle)==read_json(bundle.parent/'BUNDLE.json')['checksum'],'bundle incomplete/corrupt')
    folder=config.root/'qualification/selected_bundle'
    identity=dict(bundle_hash=file_hash(bundle),choice_hash=file_hash(config.root/'MODEL_CHOICE.json'),
        calibration_manifest_hash=file_hash(Path(config.calibration_bank)/'MANIFEST.json'),
        source=file_hash(__file__),requested_scenes=4,role='stage_val')
    reference=folder/'reference.pt';receipt=folder/'REFERENCE.json';report=folder/'REPORT.json'
    if mode=='capture' and receipt.exists():
        old=read_json(receipt)
        require(old['identity']==identity and old['checksum']==file_hash(reference),'stale bundle capture')
        return dict(status='REUSED',identity=identity)
    if mode=='verify' and report.exists():
        old=read_json(report);require(old['identity']==identity,'stale bundle verification');return old
    if mode=='capture':
        model,_=selected(config)
        system=assemble(payload_for(config,model,read_json(config.root/'scorer/SCORER_INITIALIZATION.json')),device)
        expected=None
    else:
        old=read_json(receipt)
        require(old['identity']==identity and old['checksum']==file_hash(reference),'bundle reference changed')
        expected=torch.load(reference,map_location='cpu',weights_only=False)
        system=load(bundle,device)
    scenes=scenes_by_role(source)['stage_val'][:4];wanted={s.scene_id for s in scenes}
    replay=OriginalBatchReplay(source);groups=list(dict.fromkeys(replay.batch_for(s.scene_id) for s in scenes))
    cameras=CalibrationReader(32);outputs=[];checks=[]
    for group in groups:
        batch=[replay.scenes[sid] for sid in group]
        require(all(s.split_role!='final_test' for s in batch),'forbidden online check role')
        observations=[load_observation(s,system.adapter.contract) for s in batch]
        actual=system.shared_forward(observations,calibration=cameras.batch(batch,device))
        require(actual['calls']==dict(visual=1,language=1),'extra VLA forward in selected bundle')
        require(actual['branch_calls']==dict(Base_decoder=1,expert_decoders=[1]*len(system.experts),
            score_context_projections=1,spatial_reader=1),'extra decoder/readout execution')
        # Retain all members of the canonical batch to compare identical kernels.
        row=dict(scene_ids=list(group),outputs={k:v.cpu() for k,v in actual.items() if isinstance(v,torch.Tensor)},
            observation_hashes=[s.observation_hash for s in batch],requested_scene_ids=[s.scene_id for s in batch if s.scene_id in wanted])
        if expected is not None:
            prior=expected[len(outputs)]
            require(prior['scene_ids']==row['scene_ids'] and prior['observation_hashes']==row['observation_hashes'],
                'real input or canonical batch changed')
            checks.append(dict(scene_ids=list(group),max_abs=compare_outputs(actual,prior['outputs']),
                calls=actual['calls'],branch_calls=actual['branch_calls']))
        outputs.append(row)
    if mode=='capture':
        atomic_torch(reference,outputs,immutable=True)
        atomic_json(receipt,dict(identity=identity,checksum=file_hash(reference)),immutable=True)
        return dict(status='CAPTURED',identity=identity)
    require(len(outputs)==len(expected),'incomplete bundle output comparison')
    result=dict(status='PASS',identity=identity,checks=checks,new_training_steps=0,new_navtest=False,
        ACTIVE_updated=False,hardware=torch.cuda.get_device_name() if str(device).startswith('cuda') else 'CPU',
        scientific_gain_claimed=False)
    atomic_json(report,result,immutable=True);return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('capture','verify'))
    p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    print(json.dumps(run(load_config(a.config),a.mode,a.device),allow_nan=False))


if __name__=='__main__':main()
