"""Real bounded checkpoint evaluation and diagnostic freeze/clone/deploy roundtrip.

Uses the fixed legal stage_val manifest. It never changes formal22k/ACTIVE and
never upgrades a performance checkpoint to formal Base qualification.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import gc
import torch
from iqe.config import load_config
from iqe.contracts import CandidateRecord, require
from iqe.io import atomic_json, digest, file_hash, read_json
from iqe.pipeline import Pipeline
from iqe.resources import qualify
from iqe.base_qualification import evaluate_base, qualify_base, fixed_manifest
from iqe.data.sources import load_observation
from iqe.data.candidate_bank import save_trajectory
from iqe.evaluation.retention import module_hash
from iqe.export import export_bundle, load_bundle
from iqe.selector import SelectionRule
from iqe.training.runtime_identity import configure_runtime
from iqe.vision_cache.builder import FrozenFront
from iqe.vision_cache.identity import visual_identity, payload_key
from iqe.vision_cache.inputs import cache_observation, current_identity, current_numeric_fields, CachedVisionAhead
from iqe.vision_cache.storage import ShardWriter, commit_cache, CacheReader


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--max-samples',type=int,default=32);p.add_argument('--device',default='cuda')
    a=p.parse_args();require(1<=a.max_samples<=32,'bounded fixed validation list')
    resource=qualify(a.device);torch.set_num_threads(2)
    configure_runtime({'deterministic_backward':True})
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    config=load_config(a.config); pipeline=Pipeline(config,mode='smoke',max_samples=a.max_samples,device=a.device)
    require(config['execution']['scientific_expansion'] is False,'diagnostic namespace only')
    require(not (pipeline.root/'ACTIVE.json').exists(),'diagnostic output already serving')
    evaluation=evaluate_base(pipeline,a.checkpoint)
    require(evaluation['status']=='COMPLETE' and evaluation['summary']['valid_official_scores']==a.max_samples,
            'incomplete actual official evaluation')
    qualification=qualify_base(pipeline,a.checkpoint,evaluation['result_path'])
    require(qualification['status']!='QUALIFIED','bounded checkpoint must not become formal qualification')
    pipeline.freeze_base(a.checkpoint,diagnostic_checkpoint=True);pipeline.use_locked_base()
    scenes,_=fixed_manifest(pipeline);scenes=scenes[:8]
    model=pipeline.model();before=module_hash(model)
    visual=visual_identity(model.adapter.framework,pipeline.contract)
    root=pipeline.root/'holdout_visual_cache';access={};shards=[]
    expected={};observations={};source_records={}
    # Establish the deployment reference before installing the builder's guard.
    # FrozenFront deliberately forbids LLM execution while producing payloads.
    for scene in scenes:
        obs=load_observation(scene,pipeline.contract); observations[scene.scene_id]=obs
        current=read_json(scene.observation_ref);source_records[scene.scene_id]=current
        with torch.no_grad():
            features=model.adapter.encode_scene([obs]);pred=model.adapter.predict_s0(features)
        expected[scene.scene_id]=(features,pred)
    front=FrozenFront(model.adapter.framework)
    try:
        for i,scene in enumerate(scenes):
            obs=observations[scene.scene_id];current=source_records[scene.scene_id]
            state,lang=current_numeric_fields(current)
            key=payload_key([file_hash(x) for x in current['image_paths']],lang,visual)
            access[scene.scene_id]={'key':key,'role':'stage_val','observation_hash':current_identity(current),
                                  'state_hash':digest(state.tolist()),'lang_hash':digest(lang)}
            sid=f'holdout_{i:03d}';shards.append(sid)
            if not (root/f'{sid}.bin.COMPLETE.json').exists():
                writer=ShardWriter(root,sid,digest(visual));writer.add(key,front([obs])[0]);writer.commit()
    finally:front.close()
    commit_cache(root,visual,access,shards,plan_hash=digest([s.scene_id for s in scenes]))
    reader=CacheReader(root,identity=visual);vision=CachedVisionAhead(model.adapter.framework,reader)
    rows=[];eval_rows={r['scene_id']:r for r in evaluation['rows']}
    try:
        for scene in scenes:
            cached=cache_observation(source_records[scene.scene_id],reader)
            prepared=vision.prepare([cached],1)
            with torch.no_grad(),vision.activate(prepared):
                f=model.adapter.encode_scene([cached]);pred=model.adapter.predict_s0(f)
            original,previous=expected[scene.scene_id]
            require(torch.equal(original.scene,f.scene) and torch.equal(original.ego,f.ego)
                and torch.equal(original.valid_tokens,f.valid_tokens),'heldout frozen features differ')
            require(torch.equal(previous.raw,pred.raw) and torch.equal(previous.physical,pred.physical),
                    'heldout raw/physical trajectory differs')
            pipeline.features.put(pipeline.feature_record(scene),f);stored=pipeline.cached(scene).to(pipeline.device)
            clone=model.adapter.clone_s0_expert('expert_probe').eval()
            with torch.no_grad():cloned=model.adapter.expert_forward(clone,stored)
            require(torch.equal(pred.raw,cloned.raw) and torch.equal(pred.physical,cloned.physical),'clone/cache output differs')
            ref=pipeline.root/'cached_trajectories'/f'{scene.scene_id}.npz'
            th=save_trajectory(ref,pred.physical[0].cpu().numpy());t=pipeline.trajectory_contract
            candidate=CandidateRecord(1,scene.scene_id,'expert_0',pipeline.contract['query_checkpoint_hash'],
                f.contract_hash,th,str(ref),t.raw_representation,t.frame,t.horizon,t.dt,True,
                file_hash(__file__),digest(config),'fixed_diagnostic')
            score=pipeline.backend().score(candidate,scene);reference=eval_rows[scene.scene_id]['score']
            require(score.score_valid and score.total_score_01==reference['total_score_01'] and
                score.named_components==reference['named_components'] and
                score.component_valid_masks==reference['component_valid_masks'] and
                score.official_reference_hash==reference['official_reference_hash'],'official online/cache label differs')
            rows.append({'scene_id':scene.scene_id,'raw_bitwise':True,'physical_bitwise':True,
                         'features_bitwise':True,'clone_bitwise':True,'official_labels_equal':True})
    finally:vision.close()
    require(module_hash(model)==before,'read-only lifecycle changed Base weights/buffers')
    scene=scenes[0];features=expected[scene.scene_id][0]
    pool=digest({'expert_0':pipeline.contract['query_checkpoint_hash']})
    rule=SelectionRule(0.,{}, {},always_base=True,dependency_hash=pool)
    bundle=pipeline.root/'diagnostic_bundle'
    exported=export_bundle(model,rule,pipeline.registry,bundle,mode='smoke',gate_results={'passed':False},
                           config_hash=digest(config),pool_hash=pool)
    restored,restored_rule,metadata=load_bundle(bundle,pipeline.device)
    with torch.no_grad():
        result,ids,_=restored.predict([observations[scene.scene_id]],restored_rule)
    require(ids==['expert_0'] and torch.equal(result,expected[scene.scene_id][1].physical),'online bundle restore differs')
    require(not (pipeline.root/'ACTIVE.json').exists(),'formal ACTIVE modified')
    report={'status':'PASS','checkpoint':a.checkpoint,'checkpoint_hash':file_hash(a.checkpoint),
        'formal_qualification_status':qualification['status'],'diagnostic_binding_only':True,
        'actual_stage_val_count':len(evaluation['rows']),'online_cache_holdout':rows,
        'evaluation':evaluation['result_path'],'evaluation_hash':file_hash(evaluation['result_path']),
        'bundle':str(bundle),'bundle_hash':exported['bundle_hash'],'strict_model_loading':True,
        'online_inference_restored':True,'all_three_deepstack_preserved':True,
        'formal_ACTIVE_updated':False,'locked22k_modified':False,'resource':resource}
    atomic_json(pipeline.root/'CHECKPOINT_LIFECYCLE.json',report,immutable=True)
    print({'status':'PASS','report':str(pipeline.root/'CHECKPOINT_LIFECYCLE.json')})

if __name__=='__main__':main()
