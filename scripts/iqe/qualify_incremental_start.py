"""Bounded engineering qualification of an actual qualified, trained Query Base.

No full training, navtest, activation or synthetic labels. Reuses only verified
metric protocol evidence; output is a separate diagnostic experiment namespace.
"""
import argparse
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from iqe.config import load_config
from iqe.contracts import require, strict_record, SceneRecord
from iqe.io import atomic_json, file_hash, read_json, digest
from iqe.pipeline import Pipeline, stack_features
from iqe.resources import qualify
from iqe.data.sources import load_observation, load_target
from iqe.data.sampler import ConsumedSampler
from iqe.evaluation.retention import module_hash
from iqe.training.trainer import train
from iqe.training.checkpoint import capture_rng
from iqe.rounds import run_round


def verify_bundle(pipeline):
    """Load the actual deployment files and compare fixed cached-input outputs."""
    from iqe.export import load_bundle
    model = pipeline.model(1)
    model.scorer, rule, _ = pipeline.calibrated_selector(1)
    model.set_trainable_stage('inference')
    model.eval()
    scene = pipeline.scenes(['incremental_fit'])[0]
    features = pipeline.cached(scene).to(pipeline.device)
    path = pipeline.round_root(1) / 'bundle'
    restored, restored_rule, metadata = load_bundle(path, pipeline.device)
    with torch.no_grad():
        before = model.forward_candidates(None, features=features)
        after = restored.forward_candidates(None, features=features)
        before_scores = model.forward_scores(features, before)
        after_scores = restored.forward_scores(features, after)
        selected_before = rule(before, before_scores)
        selected_after = restored_rule(after, after_scores)
    require(torch.equal(before.raw, after.raw) and torch.equal(before.trajectories, after.trajectories),
            'bundle trajectory roundtrip differs')
    require(torch.equal(before_scores.values, after_scores.values) and all(
        torch.equal(value, after_scores.components[name]) for name, value in before_scores.components.items()),
        'bundle Scorer roundtrip differs')
    require(torch.equal(selected_before[0], selected_after[0]) and selected_before[1:] == selected_after[1:],
            'bundle selected output differs')
    result = {'status': 'PASS', 'bundle': str(path), 'bundle_hash': metadata['bundle_hash'],
              'device': str(pipeline.device), 'input': 'actual_cached_features',
              'raw_physical_scores_selection_equal': True, 'strict_deployment_weight_load': True,
              'GT_or_metric_cache_used_for_prediction': False, 'formal_active_updated': False}
    atomic_json(pipeline.root / 'REAL_BUNDLE_RESTORE.json', result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--max-samples',type=int,default=32)
    parser.add_argument('--max-steps',type=int,default=2)
    parser.add_argument('--reference-evidence',required=True)
    parser.add_argument('--verify-bundle-only', action='store_true', help='read-only restore check of the completed diagnostic bundle')
    a=parser.parse_args()
    require(1<a.max_steps<=32 and 32<=a.max_samples<=128,'explicit qualification budget')
    resource=qualify(a.device)
    config=load_config(a.config);torch.set_num_threads(config['execution']['num_threads'])
    require(not config['execution']['scientific_expansion'],'qualification cannot be a scientific expansion run')
    # Deterministic head backward is needed for the exact-resume control. Model,
    # loss, precision, data and optimization defaults remain in resolved config.
    torch.use_deterministic_algorithms(True)
    p=Pipeline(config,mode='full',max_samples=a.max_samples,device=a.device)
    if a.verify_bundle_only:
        p.use_locked_base()
        print(json.dumps(verify_bundle(p)), flush=True)
        return
    p.freeze_base(config['s0']['query_checkpoint']);p.use_locked_base()
    # Full trained Base, diagnostic incremental experts: never relabel Base weights.
    p.mode='smoke'
    root=p.root
    require(not (root/'ACTIVE.json').exists(),'qualification output must not be serving')
    atomic_json(root/'QUALIFICATION_LAUNCH.json',{'resource':resource,'checkpoint':p.contract['query_checkpoint'],
        'checkpoint_hash':p.contract['query_checkpoint_hash'],'max_steps':a.max_steps,'max_samples':a.max_samples,
        'scientific_gain':'UNTESTED','formal_active_updated':False,'config_hash':p.config['config_hash']},immutable=True)
    def stage(name, fn):
        start=time.perf_counter();result=fn()
        atomic_json(root/(name+'.evidence.json'),{'result':result,'wall_seconds':time.perf_counter()-start})
        print(name, result.get('status','PASS'),flush=True)
        return result
    evidence=read_json(a.reference_evidence)
    require(evidence['passed'] and evidence['original_entry_equivalence'] and evidence['protocol_hash']==p.backend().protocol_hash,
            'stale reference evidence')
    require(any(s.scene_id==evidence['scene_id'] and s.metric_context_ref==evidence['real_metric_context'] for s in p.scenes(['stage_val'])),
            'reference context not in this qualification manifest')
    atomic_json(root/'protocol_audit/SCORE_CONTEXT_EVIDENCE.json',evidence,immutable=True)
    atomic_json(root/'protocol_audit/REUSED_PROTOCOL_SOURCE.json',{'path':a.reference_evidence,'checksum':file_hash(a.reference_evidence)},immutable=True)
    roles=['incremental_fit','stage_val','selector_cal','dev_report']
    model=p.model()
    scene=p.scenes(['incremental_fit'])[0];observation=load_observation(scene,p.contract)
    with torch.no_grad():
        features=model.adapter.encode_scene([observation])
        original=model.adapter.predict_s0(features)
        wrapped=model.forward_candidates([observation],['expert_0'])
        clone=model.adapter.clone_s0_expert('expert_probe').eval()
        cloned=model.adapter.expert_forward(clone,features)
    require(torch.equal(original.raw,cloned.raw) and torch.equal(original.physical,cloned.physical),'clone output differs')
    require(torch.equal(original.physical,wrapped.trajectories[:,0]),'K1 wrapper differs')
    atomic_json(root/'REAL_IMAGE_K1.json',{'status':'PASS','checkpoint_hash':p.contract['query_checkpoint_hash'],
        'scene_id':scene.scene_id,'clone_raw_equal':True,'K1_physical_equal':True,'input_kind':'actual_images_and_ego'})
    stage('cache_features',lambda:p.cache_features(roles))
    cached=p.cached(scene).to(p.device)
    require(torch.equal(cached.scene,features.scene) and torch.equal(cached.ego,features.ego),'online/cache features differ')
    stage('round0_candidates',lambda:p.export_candidates(0,roles))
    stage('round0_scores',lambda:p.score_candidates(0))
    pool=read_json(p.round_root(0)/'scored.json')
    keys=pool['scenes'][scene.scene_id]['expert_0'];candidate=p.bank.candidate(keys['candidate_key'])
    bank_score=p.bank.score(keys['score_key'],candidate,p.backend().protocol_hash,scene.metric_context_hash)
    direct=p.backend().score(candidate,scene,original_entry_source=p.contract['source_root'])
    require(asdict(bank_score)==asdict(direct),'official entry and bank differ')
    atomic_json(root/'REAL_BANK_REFERENCE.json',{'status':'PASS','score':asdict(direct)})
    stage('round1_manifest',lambda:p.build_round(1,a.max_steps))
    stage('round1_paused',lambda:p.train_expert(1,steps=a.max_steps,stop_after=1))
    stage('round1_resumed',lambda:p.train_expert(1,steps=a.max_steps,resume='auto'))
    stage('round1_frozen',lambda:p.frozen_audit(1))
    # A continuous head-only control uses identical cached real inputs/targets.
    checkpoint=p.round_root(1)/'expert'/f'step_{a.max_steps:06d}.pt'
    saved=torch.load(checkpoint,weights_only=False,map_location='cpu')
    manifest=read_json(p.round_root(1)/'expert_manifest.json')
    lookup={r['scene_id']:strict_record(SceneRecord,r) for r in manifest['records']}
    def fetch(entries,device):
        rows=[lookup[e['scene_id']] for e in entries]
        return {'features':stack_features([p.cached(s) for s in rows]).to(device),
            'target':torch.stack([load_target(s,p.trajectory_contract) for s in rows]).to(device),
            'masks':torch.tensor([not e.get('_padding',False) for e in entries],device=device)[:,None].expand(-1,p.trajectory_contract.horizon)}
    def loss(expert,batch):return model.adapter.compute_expert_il_loss(expert(batch['features']),batch['target'],batch['masks'])
    denominator=lambda data:{'il':data['masks'].any(-1).sum().float()}
    control=model.adapter.clone_s0_expert('expert_control').requires_grad_(True)
    history=train(control,loss,fetch,ConsumedSampler(manifest['sampling_plan']),saved['resolved_config'],saved['dependencies'],
        root/'resume_continuous',device=p.device,seed=p.config['randomness']['training_rng_seed']+1,denominator_function=denominator)
    require(all(torch.equal(v.cpu(),saved['module'][k]) for k,v in control.state_dict().items()),'real resumed parameters differ from continuous control')
    require(torch.equal(capture_rng()['cpu'],saved['rng_by_rank'][0]['cpu']),'real resume CPU RNG mismatch')
    resumed_rows=[json.loads(line) for line in (p.round_root(1)/'expert/steps.jsonl').read_text().splitlines()]
    for continuous,resumed in zip(history,resumed_rows):
        require(all(continuous[k]==resumed[k] for k in ('sample_ids','loss','gradient_norm')), 'real resume batch/loss/gradient differs')
    control_saved=torch.load(root/'resume_continuous'/f'step_{a.max_steps:06d}.pt',weights_only=False,map_location='cpu')
    require(control_saved['scheduler']==saved['scheduler'],'real resume scheduler differs')
    for key,values in saved['optimizer']['state'].items():
        for name,value in values.items():
            other=control_saved['optimizer']['state'][key][name]
            require(torch.equal(value,other) if torch.is_tensor(value) else value==other,'real resume Adam state differs')
    recovered=model.adapter.clone_s0_expert('expert_recovered').requires_grad_(True)
    sampler=ConsumedSampler(manifest['sampling_plan'])
    history_terminal=train(recovered,loss,lambda *a: (_ for _ in ()).throw(AssertionError('terminal fetched')),
        sampler,saved['resolved_config'],saved['dependencies'],root/'terminal_recovery',device=p.device,
        resume=checkpoint,seed=p.config['randomness']['training_rng_seed']+1,denominator_function=denominator)
    require(history_terminal==[] and sampler.state_dict()==saved['sampler'],'real terminal resume consumed samples')
    atomic_json(root/'REAL_RESUME.json',{'status':'PASS','continuous_vs_resumed_parameters_equal':True,'next_ids_losses_gradient_norms_adam_scheduler_equal':True,
        'terminal_new_updates':0,'checkpoint':str(checkpoint),'checkpoint_hash':file_hash(checkpoint)})
    stage('round1_candidates',lambda:p.export_candidates(1,roles))
    stage('round1_scores',lambda:p.score_candidates(1))
    stage('round1_oracle',lambda:p.evaluate_oracle(1,'stage_val'))
    stage('round1_scorer_paused',lambda:p.train_selector(1,steps=a.max_steps,stop_after=1))
    stage('round1_scorer_resumed',lambda:p.train_selector(1,steps=a.max_steps,resume='auto'))
    stage('round1_calibration',lambda:p.calibrate(1,'selector_cal'))
    stage('round1_dev',lambda:p.evaluate(1,'dev_report'))
    stage('round1_fit',lambda:p.evaluate(1,'incremental_fit'))
    # Diagnostic bundle only; smoke mode cannot publish formal ACTIVE.
    stage('bundle',lambda:p.export_bundle(1,require_gates=False))
    stage('bundle_restore',lambda:verify_bundle(p))
    stage('round2_selector_only',lambda:run_round(p,2,steps=a.max_steps,selector_only=True,resume=True))
    require(not (root/'ACTIVE.json').exists(),'qualification changed ACTIVE')
    atomic_json(root/'REAL_START_QUALIFICATION.json',{'status':'PASS','checkpoint_hash':p.contract['query_checkpoint_hash'],
        'incremental_fit_scenes':len(p.scenes(['incremental_fit'])),'steps_per_trainer':a.max_steps,
        'second_workflow_round':'selector_only','real_images':True,'real_metric_cache':True,
        'scientific_gain':'UNTESTED','formal_active_updated':False,'new_full_training_started':False})


if __name__=='__main__':main()
