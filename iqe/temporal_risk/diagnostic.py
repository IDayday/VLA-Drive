"""One300-update matched diagnostic; official event targets never enter context."""
import argparse
from dataclasses import replace
import io
import json
from pathlib import Path
import numpy as np
import torch

from iqe.contracts import require, TrajectoryContract
from iqe.io import atomic_bytes, atomic_json, digest, file_hash, lock, read_json
from iqe.evaluation.retention import module_hash, _tensor_map_hash
from iqe.internal_baseline.scorer import initialization_rng
from iqe.specialist_study.bank import load_bank
from iqe.specialist_study.binding import scenes_by_role
from iqe.specialist_study.config import StudyConfig
from iqe.specialist_study.context_reader import VerifiedSpatialReader
from iqe.specialist_study.scorer_training import batch_fetch, fetch_context
from iqe.specialist_study.objectives import StudySampler
from iqe.specialist_study.training import preload, runtime
from iqe.training.trainer import train
from iqe.geometric_readout.calibration import CalibrationReader
from iqe.geometric_readout.model import GeometricReadoutScorer
from .data import resource
from .model import TemporalRiskScorer
from .objectives import terms, denominators


def run(args):
    root=Path(args.output).resolve(); resource(root)
    job=read_json(root/'JOB.json'); complete=read_json(root/'COMPLETE.json')
    require(complete['job_hash']==digest(job) and complete['array_hash']==file_hash(root/'events.npz'), 'event bank incomplete')
    require(complete['original_bank_score_max_abs_error']<=1e-10, 'official-score equivalence required')
    plan=read_json(root/'PLAN.json')
    require(plan['maximum_new_optimizer_updates']==300 and plan['training_runs']==1, 'diagnostic-only budget')
    source=StudyConfig(**read_json(Path(job['source_campaign'])/'resolved_config.json'))
    old=Path(args.geometric_control)
    reference=read_json(old/'DIAGNOSTIC.json'); oldinit=read_json(old/'SCORER_INITIALIZATION.json')
    require(reference['status']=='COMPLETE' and reference['optimizer_steps']==300, 'matched completed control')
    oldbinding=read_json(old/'EXPERIMENT_BINDING.json')
    require(oldbinding['identity']['scene_ids']==job['scene_ids'] and
            oldbinding['identity']['rich_bank']==job['rich_bank_hash'], 'control data changed')
    dm,deploy=load_bank(source,'deploy','scorer_fit');rm,rich=load_bank(source,'rich','scorer_fit')
    require(dm['hash']==job['deployment_bank_hash'] and rm['hash']==job['rich_bank_hash'], 'current bank identity')
    ids=job['scene_ids'];ix=[dm['scene_ids'].index(sid) for sid in ids]
    scenes_map={s.scene_id:s for s in scenes_by_role(source)['scorer_fit']};scenes=[scenes_map[s] for s in ids]
    require(all(s.split_role=='incremental_fit' for s in scenes), 'fit only')
    deploy={k:v[ix] for k,v in deploy.items()};rich={k:v[ix] for k,v in rich.items()}
    with np.load(root/'events.npz') as f:events={k:f[k].copy() for k in f.files}
    folder=root/'diagnostic'
    with lock(folder/'RUN.lock'):
        device=runtime(args.device)
        cache=preload(replace(source,feature_memory_gib=16),scenes)
        spatial=VerifiedSpatialReader(source,64);calibration=CalibrationReader(64)
        first=cache.read(scenes[0]);v=spatial.read(scenes[0])
        contract=TrajectoryContract(**read_json(source.root/'OFFLINE_BASE_CONTRACT.json')['trajectory'])
        with initialization_rng(source.seed):
            control=GeometricReadoutScorer(first['world'].shape[-1],first['action'].shape[-1],
                                          contract,visual_dim=v['visual'].shape[-1])
        require(module_hash(control)==oldinit['initial_state_hash'], 'random control initialization changed')
        with initialization_rng(source.seed):
            model=TemporalRiskScorer(first['world'].shape[-1],first['action'].shape[-1],
                contract,visual_dim=v['visual'].shape[-1],event_init_seed=source.seed+30)
        require(all(torch.equal(value,model.state_dict()[name]) for name,value in control.state_dict().items()),
                'all common random weights must match')
        del control
        model.to(device)
        initial=dict(initial_state_hash=module_hash(model),module_init_seed=source.seed,event_init_seed=source.seed+30,
            parameter_count=sum(p.numel() for p in model.parameters()),
            parameters={name:list(p.shape) for name,p in model.named_parameters()},
            matched_common_initial_state_hash=oldinit['initial_state_hash'],external_scorer_weights_loaded=False,
            loaded_scorer_weights=[],allowed_resume_lineage=str(folder),
            control_weight_loaded=False,extra_event_parameters=sum(p.numel() for p in model.event_head.parameters()))
        model._iqe_initialization_identity=initial
        atomic_json(folder/'SCORER_INITIALIZATION.json',initial,immutable=True)
        sampler=StudySampler(ids,deploy['scores'],deploy['comparison_valid'],seed=source.seed+10,
            bank_hash=digest([dm['hash'],rm['hash'],ids]),phase_steps=3000)
        ordinary=batch_fetch(source,scenes,cache,spatial,deploy,rich,True)
        index={sid:i for i,sid in enumerate(ids)}
        def fetch(entries,dev):
            require(all(e['phase']==1 and e['natural'] and not e['comparison'] for e in entries), 'rich natural phase only')
            batch=ordinary(entries,dev)
            batch['context'].update(calibration.batch([scenes_map[e['scene_id']] for e in entries],dev))
            rows=[index[e['scene_id']] for e in entries]
            batch['events']=torch.from_numpy(events['rich_events'][rows]).to(dev)
            batch['event_valid']=torch.from_numpy(events['rich_valid'][rows]).to(dev)
            return batch
        inherited=read_json(old/'RUN_IDENTITY.json')['training']
        require(inherited['max_optimizer_steps']==300 and inherited['global_batch_size']==128,
                'matched control recipe mismatch')
        dependencies=dict(job_hash=digest(job),event_bank=file_hash(root/'events.npz'),sampler=sampler.identity,
            control_result=file_hash(old/'DIAGNOSTIC.json'),control_registration=file_hash(old/'EXPERIMENT_BINDING.json'),
            initial_state_hash=initial['initial_state_hash'],training_rng_seed=source.seed+20,
            sources={n:file_hash(Path(__file__).with_name(n)) for n in ('diagnostic.py','model.py','objectives.py')})
        atomic_json(folder/'RUN_IDENTITY.json',dict(dependencies=dependencies,training=inherited,
            event_loss_weight=1.,same_trajectory_value_head=True,diagnostic_generalization_claim=False),immutable=True)
        result=train(model,lambda m,b:terms(m(b['context'],b['trajectories']),b),fetch,sampler,
            inherited,dependencies,folder,device=device,seed=source.seed+20,resume=args.resume,
            stop_after=args.stop_after,denominator_function=denominators,
            weights={'value':1.,'components':.2,'ranking':.2,'events':1.})
        result=read_json(folder/'result.json')
        if result['status']!='COMPLETE':return result
        model.eval()
        @torch.no_grad()
        def predict(kind,wrong=False):
            active=rich if kind=='rich' else deploy; values=[];probabilities=[]
            for at in range(0,len(scenes),16):
                sub=scenes[at:at+16];context=fetch_context(source,sub,device,cache,spatial)
                if wrong:
                    shifted=[scenes[(i+1)%len(scenes)] for i in range(at,at+len(sub))]
                    context.update(spatial.batch(shifted,device))
                context.update(calibration.batch(sub,device))
                p=model(context,torch.from_numpy(active['trajectories'][at:at+len(sub)]).to(device))
                values.append(p['values'].cpu().numpy());probabilities.append(p['event_logits'].sigmoid().cpu().numpy())
            value=np.concatenate(values);probability=np.concatenate(probabilities)
            require(np.isfinite(value).all() and np.isfinite(probability).all(),'nonfinite diagnostic prediction')
            valid=events[kind+'_valid'];target=events[kind+'_events'];chosen=value.argmax(1)
            result=dict(selected_01=float(active['scores'][np.arange(len(scenes)),chosen].mean()),
                base_01=float(active['scores'][:,0].mean()),oracle_01=float(np.where(active['valid'],active['scores'],-np.inf).max(1).mean()),
                mse=float(((value-active['scores'])**2)[active['valid']].mean()),
                event_brier=float(((probability-target)**2)[valid].mean()),
                event_recall_at_0_5=float((probability[target&valid]>=.5).mean()) if (target&valid).any() else None,
                false_positive_at_0_5=float((probability[~target&valid]>=.5).mean()) if (~target&valid).any() else None)
            return value,probability,result
        correct,cp,cr=predict('deploy');wrong,wp,wr=predict('deploy',True);rv,rp,rr=predict('rich')
        stream=io.BytesIO();np.savez(stream,correct=correct,wrong_visual=wrong,deploy_events=cp,rich_values=rv,rich_events=rp)
        atomic_bytes(folder/'PREDICTIONS.npz',stream.getvalue(),immutable=True)
        report=dict(status='COMPLETE',optimizer_steps=300,scenes=64,logs=complete['logs'],
            correct=cr,wrong_visual=wr,rich_fit=rr,matched_control=reference['correct'],
            selected_gain_over_control_01=cr['selected_01']-reference['correct']['selected_01'],
            changed_selections_with_wrong_visual=int((correct.argmax(1)!=wrong.argmax(1)).sum()),
            source_event_hash=complete['array_hash'],predictions_hash=file_hash(folder/'PREDICTIONS.npz'),
            parameter_hash=module_hash(model),parameters_changed=result['parameters_changed'],
            external_scorer_weights_loaded=False,VLM_calls=0,Base_updates=0,expert_updates=0,
            scientific_generalization_result=False,weights_eligible_for_deployment=False,
            full_training_started=False)
        atomic_json(folder/'REPORT.json',report,immutable=True)
        return report


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',required=True);p.add_argument('--geometric-control',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--resume',nargs='?',const='auto')
    p.add_argument('--stop-after',type=int);a=p.parse_args()
    require(a.stop_after is None or 0<a.stop_after<=300,'bounded diagnostic updates')
    print(json.dumps(run(a),allow_nan=False))


if __name__=='__main__':main()
