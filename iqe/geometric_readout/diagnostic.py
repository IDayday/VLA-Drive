"""One registered300-update geometry diagnostic; no automatic formal training."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import io
import json
import socket
import subprocess
from types import SimpleNamespace
import numpy as np
import torch
from iqe.contracts import require, TrajectoryContract
from iqe.io import atomic_json, atomic_bytes, read_json, digest, file_hash, lock
from iqe.evaluation.retention import module_hash
from iqe.internal_baseline.scorer import initialization_rng
from iqe.specialist_study.config import StudyConfig
from iqe.specialist_study.binding import scenes_by_role
from iqe.specialist_study.bank import load_bank, pool
from iqe.specialist_study.context_reader import VerifiedSpatialReader
from iqe.specialist_study.data import ReusedContext
from iqe.specialist_study.objectives import StudySampler, terms, denominators
from iqe.specialist_study.scorer_training import batch_fetch, fetch_context
from iqe.specialist_study.training import preload, runtime
from iqe.training.trainer import train
from .calibration import CalibrationReader
from .model import GeometricReadoutScorer


@torch.no_grad()
def predictions(model,source,scenes,bank,device,cache,spatial,calibration,wrong=False):
    model.eval(); values=[]
    for start in range(0,len(scenes),16):
        sub=scenes[start:start+16]
        context=fetch_context(source,sub,device,cache,spatial)
        if wrong:
            shifted=[scenes[(i+1)%len(scenes)] for i in range(start,start+len(sub))]
            context.update(spatial.batch(shifted,device))
        context.update(calibration.batch(sub,device))
        trajectory=torch.from_numpy(bank['trajectories'][start:start+len(sub)]).to(device)
        prediction=model(context,trajectory)['values']
        require(torch.isfinite(prediction).all().item(), 'nonfinite diagnostic prediction')
        values.append(prediction.cpu().numpy())
    return np.concatenate(values)


def run(args):
    root=Path(args.output);source=StudyConfig(**read_json(Path(args.source_campaign)/'resolved_config.json'))
    require(root.resolve()!=source.root.resolve(), 'isolated diagnostic output')
    from iqe.specialist_study.resource_guard import check
    policy=read_json('/mnt/project/server_dispatch_policy.json')
    permanent=set(policy['allowed_hosts'])
    for item in policy['allowed_hosts'].values():permanent.update(item.get('aliases',[]))
    require(socket.gethostname() in permanent,'new follow-up uses permanent hosts only')
    check(SimpleNamespace(root=root),'train-scorer-cell' if args.command=='train' else 'research-preflight')
    cfg=dict(source_campaign=str(source.root),control_root=str(Path(args.control_root).resolve()),
        output_root=str(root.resolve()),geometry_audit=str(Path(args.geometry_audit).resolve()),
        max_optimizer_steps=300,global_batch_size=128,micro_batch_size=16,
        phase1_only=True,formal_training_authorized_by_this_entry=False)
    atomic_json(root/'resolved_config.json',cfg,immutable=True)
    control=Path(args.control_root)
    data=read_json(control/'DIAGNOSTIC_DATA.json')
    dm,deploy=load_bank(source,'deploy','scorer_fit');rm,rich=load_bank(source,'rich','scorer_fit')
    scenes=scenes_by_role(source)['scorer_fit'];lookup={s.scene_id:i for i,s in enumerate(scenes)}
    require(data['rich_bank_hash']==rm['hash'] and len(data['scene_ids'])==64
            and len(set(data['scene_ids']))==64, 'matched64-scene control required')
    ix=[lookup[s] for s in data['scene_ids']];scenes=[scenes[i] for i in ix]
    require(all(s.split_role=='incremental_fit' for s in scenes),'fit scenes only')
    deploy={k:v[ix] for k,v in deploy.items()};rich={k:v[ix] for k,v in rich.items()}
    audit=read_json(Path(args.geometry_audit)/'REPORT.json')
    require(audit['all_calibrations_match'] and audit['all_lidar2ego_identity']
            and audit['official_projection_max_abs']<1e-8,'geometry audit required')
    audit_registration=read_json(Path(args.geometry_audit)/'REGISTRATION.json')
    require(audit_registration['base_binding_hash']==file_hash(source.root/'BASE_BINDING.json')
            and audit_registration['bank_hash']==dm['hash'],'geometry audit source changed')
    original_recipe=read_json(source.root/'resolved_recipes/scorer_2x2.json')
    require(file_hash(original_recipe['source'])==original_recipe['checksum'],'reference recipe changed')
    control_initial=read_json(control/'diagnostics/global/SCORER_INITIALIZATION.json')
    reference_result=read_json(control/'diagnostics/global/DIAGNOSTIC.json')
    require(reference_result['optimizer_steps']==300 and reference_result['status']=='COMPLETE','actual completed control required')
    registration=dict(schema='geometric_readout_diagnostic_v1',source_binding=file_hash(source.root/'BASE_BINDING.json'),
        pool_hash=pool(source)['hash'],deploy_bank=dm['hash'],rich_bank=rm['hash'],scene_ids=data['scene_ids'],
        audit_hash=file_hash(Path(args.geometry_audit)/'REPORT.json'),configuration=cfg,
        control_result_hash=file_hash(control/'diagnostics/global/DIAGNOSTIC.json'),
        recipe_hash=file_hash(source.root/'resolved_recipes/scorer_2x2.json'),
        changed='candidate-local3x3 image-token reading using current camera geometry and two explicit height hypotheses',
        camera_model='locked_NAVSIM_pinhole', distortion_reapplied=False, height_hypotheses_metres=[0.,1.],
        matched=['Base','experts','64 real D_S scenes','rich labels','initial weights','optimizer','sample sequence','loss'],
        new_training_runs=1,optimizer_update_cap=300,full_training=False,
        prior_dev_results_seen=True,dev_or_test_used_in_this_diagnostic=False,
        old_matrix_expanded=False,external_scorer_weights_loaded=False,
        stopping_rule='finish300 unless numerical/data/resource failure; no automatic formal expansion',
        files={name:file_hash(Path(__file__).with_name(name)) for name in ('diagnostic.py','model.py','geometry.py','calibration.py')})
    with lock(root/'RUN.lock'):
        registered=root/'EXPERIMENT_BINDING.json'
        if registered.exists():
            old=read_json(registered);require(old['identity']==registration,'registered diagnostic mutated')
        else:
            atomic_json(registered,dict(identity=registration,registered_utc=datetime.now(timezone.utc).isoformat(),
                code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),immutable=True)
        if args.command=='bind':return dict(status='REGISTERED',max_updates=300)
        device=runtime(args.device)
        cache=preload(replace(source,feature_memory_gib=16),scenes)
        spatial=VerifiedSpatialReader(source,64);calibration=CalibrationReader(64)
        first=cache.read(scenes[0]);v=spatial.read(scenes[0])
        contract=read_json(source.root/'OFFLINE_BASE_CONTRACT.json')
        with initialization_rng(source.seed):
            model=GeometricReadoutScorer(first['world'].shape[-1],first['action'].shape[-1],
                TrajectoryContract(**contract['trajectory']),visual_dim=v['visual'].shape[-1])
        require(module_hash(model)==control_initial['initial_state_hash'],'random initialization must match control')
        initial=dict(initial_state_hash=module_hash(model),module_init_seed=source.seed,
            external_scorer_weights_loaded=False,loaded_scorer_weights=[],
            allowed_resume_lineage=str(root),parameter_count=sum(p.numel() for p in model.parameters()),
            parameters={name:list(p.shape) for name,p in model.named_parameters()})
        model._iqe_initialization_identity=initial
        atomic_json(root/'SCORER_INITIALIZATION.json',initial,immutable=True)
        ids=[s.scene_id for s in scenes]
        sampler=StudySampler(ids,deploy['scores'],deploy['comparison_valid'],seed=source.seed+10,
            bank_hash=digest([dm['hash'],rm['hash'],ids]),phase_steps=3000)
        ordinary=batch_fetch(source,scenes,cache,spatial,deploy,rich,True)
        by_id={s.scene_id:s for s in scenes}
        def fetch(entries,dev):
            batch=ordinary(entries,dev)
            batch['context'].update(calibration.batch([by_id[e['scene_id']] for e in entries],dev))
            return batch
        recipe=original_recipe['resolved']
        keys=('lr','weight_decay','global_batch_size','max_optimizer_steps','warmup_steps',
              'lr_min_ratio','grad_clip_norm','schedule','validation_steps','checkpoint_every_steps')
        training={k:recipe[k] for k in keys}|dict(micro_batch_size=16,validation_every_steps=1000,
            early_stop_patience=10**9,max_optimizer_steps=300,validation_steps=[],checkpoint_every_steps=100)
        dependencies=dict(registration=digest(registration),sampler=sampler.identity,
            initial_state_hash=initial['initial_state_hash'],training_rng_seed=source.seed+20)
        atomic_json(root/'RUN_IDENTITY.json',dict(dependencies=dependencies,training=training),immutable=True)
        def loss(m,b):return terms(m(b['context'],b['trajectories']),b)
        train(model,loss,fetch,sampler,training,dependencies,root,device=device,
            weights={'value':1.,'components':.2,'ranking':.2},seed=source.seed+20,
            resume=args.resume,stop_after=args.stop_after,denominator_function=denominators)
        result=read_json(root/'result.json')
        if result['status']!='COMPLETE':return result
        correct=predictions(model,source,scenes,deploy,device,cache,spatial,calibration)
        wrong=predictions(model,source,scenes,deploy,device,cache,spatial,calibration,True)
        def summary(p):
            selected=p.argmax(1)
            return dict(selected_01=float(deploy['scores'][np.arange(len(scenes)),selected].mean()),
                base_01=float(deploy['scores'][:,0].mean()),oracle_01=float(deploy['scores'].max(1).mean()),
                mse=float(((p-deploy['scores'])**2)[deploy['valid']].mean()))
        data=io.BytesIO();np.savez(data,correct=correct,wrong_visual=wrong)
        atomic_bytes(root/'DIAGNOSTIC_PREDICTIONS.npz',data.getvalue(),immutable=True)
        report=dict(status='COMPLETE',optimizer_steps=300,scenes=64,correct=summary(correct),
            wrong_visual=summary(wrong),global_control=reference_result['correct'],
            maximum_prediction_change=float(np.abs(correct-wrong).max()),
            changed_selections=int((correct.argmax(1)!=wrong.argmax(1)).sum()),
            final_state_hash=module_hash(model),parameters_changed=result['parameters_changed'],
            experimental_risk='geometry assumptions and frozen V may still be insufficient',
            weights_eligible_for_deployment=False,scientific_generalization_result=False)
        atomic_json(root/'DIAGNOSTIC.json',report,immutable=True)
        return report


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('command',choices=('bind','train'))
    p.add_argument('--source-campaign',required=True);p.add_argument('--control-root',required=True)
    p.add_argument('--geometry-audit',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--resume',nargs='?',const='auto')
    p.add_argument('--stop-after',type=int)
    a=p.parse_args()
    require(a.stop_after is None or 0<a.stop_after<=300,'diagnostic update cap300')
    print(json.dumps(run(a),allow_nan=False))


if __name__=='__main__':main()
