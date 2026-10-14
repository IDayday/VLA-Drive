"""Real-image equivalence/call-count check of the new current-camera boundary.

Can use an already complete native checkpoint from this trial before the final
endpoint. Such an engineering check is explicitly not heldout quality evidence.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from iqe.contracts import require
from iqe.data.sources import load_observation
from iqe.io import atomic_json, read_json, file_hash
from iqe.specialist_study.context_batch_layout import OriginalBatchReplay
from iqe.specialist_study.binding import scenes_by_role
from iqe.specialist_study.data import ReusedContext
from iqe.specialist_study.context_reader import VerifiedSpatialReader
from iqe.specialist_study.scorer_training import fetch_context
from iqe.specialist_study.training import runtime
from .full import load_config, create, resource_check
from .calibration import CalibrationReader
from .calibration_bank import PreparedCalibrationReader
from .bundle import payload_for,assemble

TOLERANCES=dict(context=1e-5,raw=1e-5,physical=1e-4,score=1e-5)


@torch.no_grad()
def run(config,checkpoint,device):
    resource_check(config);device=runtime(device);source=config.source();checkpoint=Path(checkpoint).resolve()
    require(checkpoint.parent==(config.root/'scorer').resolve(),'same-trial checkpoint only')
    receipt=read_json(str(checkpoint)+'.COMPLETE.json')
    require(file_hash(checkpoint)==receipt['checksum'],'checkpoint receipt mismatch')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    identity=read_json(config.root/'scorer/RUN_IDENTITY.json')
    require(state['dependencies']==identity['dependencies'],'checkpoint experiment mismatch')
    model,initial=create(config);model.load_state_dict(state['module'],strict=True)
    system=assemble(payload_for(config,model,initial),device)
    scenes=scenes_by_role(source)['stage_val'][:4];wanted={s.scene_id for s in scenes}
    replay=OriginalBatchReplay(source);groups=list(dict.fromkeys(replay.batch_for(s.scene_id) for s in scenes))
    cache=ReusedContext(source,128);spatial=VerifiedSpatialReader(source,64)
    calibration=PreparedCalibrationReader(source,config.calibration_bank);live=CalibrationReader(32)
    rows=[]
    for group in groups:
        batch=[replay.scenes[sid] for sid in group]
        observations=[load_observation(s,system.adapter.contract) for s in batch]
        cameras=live.batch(batch,device);captured={}
        def context(_m,args):captured['context']={k:v.detach().cpu().clone() for k,v in args[0].items()}
        def raw(_m,_a,out):captured['raw']=out.raw.detach().cpu().clone()
        handles=[system.scorer.register_forward_pre_hook(context),
                 system.adapter.framework.action_model.expert.register_forward_hook(raw)]
        try:online=system.shared_forward(observations,calibration=cameras)
        finally:
            for h in handles:h.remove()
        require(online['calls']=={'visual':1,'language':1},'second VLA inference')
        require(online['branch_calls']==dict(Base_decoder=1,expert_decoders=[1]*len(system.experts),
            score_context_projections=1,spatial_reader=1),'extra branch/projection call')
        ix=[i for i,s in enumerate(batch) if s.scene_id in wanted];sub=[batch[i] for i in ix]
        c=fetch_context(source,sub,device,cache,spatial);c.update(calibration.batch(sub,device))
        differences={}
        for name,value in c.items():
            reference=captured['context'][name][ix];actual=value.cpu()
            require(actual.shape==reference.shape and actual.dtype==reference.dtype,'context schema changed')
            error=0. if value.dtype==torch.bool and torch.equal(actual,reference) else float((actual.float()-reference.float()).abs().max())
            require(error<=TOLERANCES['context'],'online/cache context changed: '+name);differences[name]=error
        x=cache.batch(sub,device)
        from iqe.contracts import FeatureBundle
        features=FeatureBundle(x['planner_scene'],x['planner_ego'],x['planner_mask'],system.adapter.contract_hash,tuple(s.scene_id for s in sub))
        base=system.adapter.predict_s0(features)
        trajectory=torch.stack([base.physical]+[m(x).physical for m in system.experts],1)
        prediction=system.scorer(c,trajectory)['values']
        differences.update(Base_raw=float((base.raw.cpu()-captured['raw'][ix]).abs().max()),
            physical=float((trajectory-online['candidates'][ix]).abs().max()),
            score=float((prediction-online['predicted_scores'][ix]).abs().max()))
        require(differences['Base_raw']<=TOLERANCES['raw'] and differences['physical']<=TOLERANCES['physical']
                and differences['score']<=TOLERANCES['score'],'online/cache final output differs')
        require(torch.equal(prediction.argmax(1),online['selected'][ix]),'cache selection differs')
        polluted=[o|{'GT':object(),'future_actors':object(),'true_score':float('nan')} for o in observations]
        again=system.shared_forward(polluted,calibration=cameras)
        for key in ('candidates','predicted_scores','selected'):
            require(torch.equal(again[key],online[key]),'training labels changed online output')
        rows.append(dict(scene_ids=[s.scene_id for s in sub],canonical_batch=list(group),differences=differences,
            calls=online['calls'],branch_calls=online['branch_calls'],label_independent=True))
    result=dict(status='PASS',checkpoint=str(checkpoint),checkpoint_hash=file_hash(checkpoint),
        optimizer_step=receipt['optimizer_step'],native_world_size=receipt['world_size'],real_scenes=4,
        tolerances=TOLERANCES,rows=rows,scientific_quality_result=False,
        hardware=torch.cuda.get_device_name(),new_optimizer_steps=0,
        temporary_host=False,external_scorer_weights_loaded=False)
    atomic_json(config.root/f'qualification/online_{receipt["optimizer_step"]:06d}.json',result,immutable=True)
    return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();print(json.dumps(run(load_config(a.config),a.checkpoint,a.device),allow_nan=False))


if __name__=='__main__':main()
