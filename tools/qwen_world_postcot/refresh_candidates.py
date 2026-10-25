"""Asynchronous own-policy candidate refresh: frozen formal IL, two samples/scene.

No optimizer/teacher/post-head runs. Four current scenes share one Qwen pass;
the prepared actor context is repeated for two independent K10 noise samples.
Fourteen existing physical candidates keep their already verified real labels.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import fcntl
import io
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import torch

from starVLA.rl.qwen_world_postcot.contracts import ROOT, require_dispatch, boundary_stop_due
from starVLA.rl.qwen_world_postcot.data import WorldSceneData
from starVLA.rl.qwen_world_postcot.deployment import inspect_export, load_export
from starVLA.model.modules.qwen_world_postcot.candidate_bank import (
    CandidateBank, CandidateBankWriter, CandidateRecord, atomic_json, file_sha,
    fingerprint, trajectory_hash, observation_fingerprint, merge_bank_parts)
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
from starVLA.rl.simwam_recipe.contracts import seed_for
from starVLA.rl.simwam_recipe.reward_readouts import RewardService
from starVLA.rl.flow_grpo.reproducibility import configure_numerics

REFRESH_STEPS=tuple(range(10000,90001,10000))
REGISTRY_SCHEMA='world_postcot_bank_registry_v1'
RECIPE=dict(new_candidates_per_scene=2,historical_candidates_per_scene=14,
    maximum_candidates_per_scene=16,batch_current_scenes=4,inference_steps=10,shift=5,
    initial_noise='independent fixed CPU FP32 generator per checkpoint/scene/candidate',
    retention='loggedGT then original ownIL/RL then coherent history then prior refreshed policy; stable record_id; no score-based selection')


def snapshot_binding(checkpoint,metadata,*,qualification=False):
    root=Path(checkpoint).resolve();seal=json.loads((root/'COMPLETE').read_text())
    identity=json.loads((root/'identity.json').read_text());files=json.loads((root/'files.json').read_text())
    if (seal.get('status')!='COMPLETE' or seal['files_sha256']!=file_sha(root/'files.json')
            or seal['identity_sha256']!=fingerprint(identity)
            or identity.get('schema')!='world_postcot_full_IL_v1'
            or identity.get('uses_rl')is not False
            or(not qualification and(identity.get('qualification')is not False or seal['updates']not in REFRESH_STEPS))
            or(qualification and not 2<=seal['updates']<=250)):
        raise ValueError('Refresh only the nine real formal IL10k..90k milestones; never a random/qualification/RL policy')
    if ((metadata['qualification']and not qualification)or metadata['RL_parent']is not None
            or Path(metadata['IL_parent']['path']).resolve()!=root
            or metadata['IL_parent']['model_sha256']!=files[seal['model_file']]
            or metadata['IL_parent']['identity']!=identity):
        raise ValueError('Refresh export does not represent the exact pinned formal IL snapshot')
    client_name='rank00_state.pt'
    if file_sha(root/client_name)!=files[client_name]:raise ValueError('Source checkpoint consumed-bank state changed')
    client=torch.load(root/client_name,map_location='cpu',weights_only=False)
    builder=client['builder']
    if builder.get('schema')!='world_postcot_batch_builder_v2' or not builder.get('bank_root')or not builder.get('bank_hash'):
        if not qualification:raise ValueError('Source checkpoint must record its actually consumed bank root/hash, not latest registry')
        # Earlier engineering checkpoints predate live bank versioning. Their
        # sealed run identity fixes one immutable bank and permits no refresh.
        builder=dict(bank_root=identity['config']['candidate_bank'],bank_hash=identity['candidate_bank_hash'])
    return dict(source_updates=seal['updates'],source_checkpoint=str(root),source_model_sha256=files[seal['model_file']],
        source_export_model_sha256=metadata['model_sha256'],parent_bank_root=builder['bank_root'],parent_bank_hash=builder['bank_hash'],
        qualification=qualification)


def retain_history(records):
    priority={'logged_gt':0,'native_qwen_il':1,'native_qwen_rl':1,'coherent_variant':2,'current_student_snapshot':3}
    selected=[];seen=set()
    for record in sorted(records,key=lambda value:(priority[value.candidate_source],value.record_id)):
        if record.trajectory_sha256 in seen:continue
        selected.append(record);seen.add(record.trajectory_sha256)
        if len(selected)==14:break
    if len(selected)<2:raise ValueError('Refresh cannot fabricate missing parent candidates')
    if any(value.candidate_source=='logged_gt'for value in records)and not any(value.candidate_source=='logged_gt'for value in selected):
        raise ValueError('Logged-pose historical candidate was accidentally removed')
    return selected


def current_only_input(row,manifest_sha256,expected_observation):
    images={};hashes={}
    for name,path in row['images'].items():
        if path is None:raise FileNotFoundError('A registered three-view current observation is missing')
        payload=Path(path).read_bytes();hashes[name]=__import__('hashlib').sha256(payload).hexdigest()
        with Image.open(io.BytesIO(payload))as image:images[name]=image.convert('RGB')
    actual=observation_fingerprint(row,manifest_sha256,hashes)
    if actual!=expected_observation:raise ValueError('Current image/state/instruction identity changed since parent bank')
    # No GT, teacher-frame, object-target or score field enters the processor.
    return dict(images=images,proprio=row['proprio'],navigation=row['navigation'])


def _compact_feedback(value):
    return dict(token=value['token'],log=value['log'],status=value['status'],protocol=value['protocol'],
        protocol_sha256=value['protocol_sha256'],cache=value['cache'],metrics={k:float(v)for k,v in value['metrics'].items()},
        requested_trajectory=np.asarray(value['requested_trajectory'],dtype=np.float32).tolist())


def refresh(args):
    require_dispatch(admit_long_job=True)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('FLASH_ATTENTION_DETERMINISTIC','1')
    configure_numerics();torch.set_num_threads(4);torch.cuda.set_device(0)
    model,metadata=load_export(args.export,'cuda');binding=snapshot_binding(args.source_checkpoint,metadata,qualification=args.qualification)
    data=WorldSceneData(args.data,role='TRAIN');parent=CandidateBank(binding['parent_bank_root'],allowed_splits=('TRAIN',),
        expected_identity=dict(source_data_manifest_sha256=data.sha,source_records_sha256=data.manifest['records_sha256']))
    if parent.bank_hash!=binding['parent_bank_hash']or(not args.qualification and set(parent.manifest['scene_ids'])!=set(data.tokens)):
        raise ValueError('The checkpoint parent bank is incomplete or changed')
    tokens=data.tokens
    if args.qualification:
        fixed=json.loads((ROOT/'data/PERFORMANCE_TOKENS.json').read_text())
        fixed=fixed['tokens']if isinstance(fixed,dict)else fixed
        tokens=[token for token in fixed if token in set(parent.manifest['scene_ids'])][:args.max_scenes]
        if len(tokens)!=args.max_scenes:raise ValueError('Qualification cannot invent additional scenes outside its actual saved parent bank')
    assigned=tokens[args.rank::args.world]
    if not assigned or args.max_new_candidates<2*len(assigned):raise ValueError('Explicit unique-candidate budget must cover the entire assigned fixed scene shard')
    identity=dict(parent.manifest['identity'],refresh_schema='world_postcot_current_student_refresh_DIAGNOSTIC_v1'if args.qualification else'world_postcot_current_student_refresh_v1',
        **binding,refresh_recipe=RECIPE,scene_manifest_sha256=fingerprint(tokens),source_data_manifest_sha256=data.sha,
        source_records_sha256=data.manifest['records_sha256'])
    if args.qualification:identity['diagnostic_scene_ids']=tokens
    output=Path(args.output);part=output/'parts'/f'rank{args.rank:05d}'
    if(part/'COMPLETE.json').exists():
        old=CandidateBank(part,allowed_splits=('TRAIN',),expected_identity=identity)
        if old.manifest['scene_ids']!=assigned:raise ValueError('Completed refresh producer changed fixed scene ownership')
        old.close();parent.close();return 0
    part.mkdir(parents=True,exist_ok=True)
    atomic_json(part/'RESOLVED_CONFIG.json',dict(arguments=vars(args),identity=identity,assigned_scene_count=len(assigned),GPU=True))
    processor=QwenObservationProcessor(str(Path(args.export)/'processor'))
    service=RewardService(Path.cwd(),workers=1)
    if service.protocol['sha256']!=identity['score_protocol_sha256']:raise ValueError('Refresh canonical evaluator changed')
    revision=dict(name='NAVSIM_v1.1',git_commit=None,source_tree_sha256=fingerprint(service.protocol['sources']),
        adapter_source_sha256=file_sha('scripts/analysis/constrained_flow_noise/batched_canonical_score.py'),dependencies=service.protocol['dependencies'])
    progress=dict(status='RUNNING',scenes_completed=0,new_candidates_generated=0,new_official_scores=0,
        parent_exact_scores_reused=0,duplicate_generated_requests=0,VLM_batch_calls=0,rank=args.rank,world=args.world,
        qualification=args.qualification)
    counts=dict(visual_batch_calls=0,language_batch_calls=0)
    def hook(name):
        def count(*_):counts[name]+=1
        return count
    handles=[model.qwen.model.visual.register_forward_hook(hook('visual_batch_calls')),
             model.qwen.model.language_model.register_forward_hook(hook('language_batch_calls'))]
    started=time.monotonic();score_attempts=0
    for file in(part/'score_attempts').glob('*.json'):
        score_attempts+=json.loads(file.read_text())['requested_candidates']
    try:
        with CandidateBankWriter(part,identity,assigned,minimum_candidates=2,maximum_candidates=16,
                resume=args.resume)as writer:
            pending=[]
            for token in assigned:
                if(part/'committed'/f'{token}.json').exists():
                    value=json.loads((part/'committed'/f'{token}.json').read_text())
                    if value['source_model_sha256']!=binding['source_model_sha256']or writer.scene_count(token)!=value['record_count']:
                        raise ValueError('Refresh committed scene receipt changed')
                    progress['scenes_completed']+=1;continue
                pending.append(token)
            for begin in range(0,len(pending),4):
                if boundary_stop_due():
                    atomic_json(part/'PAUSED.json',dict(status='STOPPED_AT_SCENE_BOUNDARY',progress=progress));return 75
                batch=pending[begin:begin+4];old={token:parent.records(token)for token in batch}
                absent=[token for token in batch if not(part/'proposals'/f'{token}.json').exists()]
                if absent:
                    observations=[current_only_input(data.records[token],data.sha,old[token][0].observation_hash)for token in absent]
                    ego=torch.tensor([value['proprio']for value in observations],device='cuda',dtype=torch.float32)
                    noise=torch.stack([torch.randn(8,3,generator=torch.Generator().manual_seed(
                        seed_for('world_postcot_current_student_refresh',42,binding['source_updates'],token,candidate)))
                        for token in absent for candidate in range(2)]).cuda()
                    before=model.qwen.forward_calls
                    visual_before,language_before=counts['visual_batch_calls'],counts['language_batch_calls']
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        memory=model.encode_condition(processor(observations),ego)
                        context=model.actor.prepare_context(memory)
                        context=context.index_select(torch.arange(len(absent),device='cuda').repeat_interleave(2))
                        normalized=model.policy.sample_ode(context,noise.float())
                        physical=ActionCodec.denorm_odo(normalized.float()).reshape(len(absent),2,8,3).cpu().numpy()
                    if model.qwen.forward_calls-before!=1:raise ValueError('Current Qwen must run exactly once per <=4-scene refresh batch')
                    if counts['visual_batch_calls']-visual_before!=1 or counts['language_batch_calls']-language_before!=1:
                        raise ValueError('Multiple candidates caused an extra visual/language forward')
                    progress['VLM_batch_calls']+=1
                    for token,requests in zip(absent,physical):
                        for request in requests:trajectory_hash(request)
                        atomic_json(part/'proposals'/f'{token}.json',dict(scene_id=token,**binding,
                            observation_hash=old[token][0].observation_hash,trajectory_fp32=requests.tolist(),
                            sampling=dict(seed=42,noise_seeds=[seed_for('world_postcot_current_student_refresh',42,binding['source_updates'],token,c)for c in range(2)],
                                inference_steps=10,shift=5,batch_scene_ids=absent,batch_current_scenes=len(absent),
                                current_Qwen_forward_calls=1,independent_candidates_per_scene=2)))
                        progress['new_candidates_generated']+=2
                for token in batch:
                    if boundary_stop_due():atomic_json(part/'PAUSED.json',dict(status='STOPPED_AT_SCENE_BOUNDARY',progress=progress));return 75
                    row=data.records[token];proposals=json.loads((part/'proposals'/f'{token}.json').read_text())
                    if any(proposals[name]!=binding[name]for name in binding)or proposals['observation_hash']!=old[token][0].observation_hash:
                        raise ValueError('Stored refresh requests belong to another model/parent observation')
                    requests=np.asarray(proposals['trajectory_fp32'],dtype=np.float32)
                    if requests.shape!=(2,8,3):raise ValueError('Refresh must generate exactly two complete independent-noise trajectories')
                    plan_hash=fingerprint(proposals);saved_path=part/'scores'/f'{token}.json'
                    if saved_path.exists():
                        saved=json.loads(saved_path.read_text())
                        if saved['proposal_sha256']!=plan_hash:raise ValueError('A score receipt names a different refresh request')
                        feedback=saved['feedback']
                    else:
                        feedback=[];lookup={value.trajectory_sha256:value for value in old[token]};fresh=[];positions=[]
                        for index,request in enumerate(requests):
                            existing=lookup.get(trajectory_hash(request))
                            if existing is not None:
                                from starVLA.rl.simwam_recipe.contracts import CACHE_ROOTS
                                context_path=next((Path(root)/row['log']/'unknown'/token/'metric_cache.pkl'for root in CACHE_ROOTS
                                    if(Path(root)/row['log']/'unknown'/token/'metric_cache.pkl').is_file()),None)
                                if context_path is None or file_sha(context_path)!=existing.context_sha256:raise ValueError('Parent metric context changed')
                                feedback.append(dict(token=token,log=row['log'],status='OK',protocol=existing.metric_version,
                                    protocol_sha256=existing.score_protocol_sha256,cache={'path':str(context_path),'sha256':existing.context_sha256},
                                    metrics=dict(existing.raw_factor_labels,score=existing.raw_total_score),requested_trajectory=request.tolist()))
                                progress['parent_exact_scores_reused']+=1
                            elif any(np.array_equal(request,requests[j])for j in positions):
                                feedback.append(None);progress['duplicate_generated_requests']+=1
                            else:feedback.append(None);fresh.append(request);positions.append(index)
                        if fresh:
                            for attempt in range(args.scoring_retries+1):
                                if score_attempts+len(fresh)>args.max_simulation_attempts:raise RuntimeError('Explicit simulation-attempt budget exhausted; exact requests retained')
                                score_attempts+=len(fresh);attempt_path=part/'score_attempts'/f'{token}_{time.time_ns()}.json'
                                try:
                                    scored=service.score((token,row['log'],np.asarray(fresh,dtype=np.float32)))
                                    if len(scored)!=len(fresh):raise ValueError('Incomplete real official refresh labels')
                                    atomic_json(attempt_path,dict(status='PASS',requested_candidates=len(fresh),proposal_sha256=plan_hash))
                                    progress['new_official_scores']+=len(fresh);break
                                except(OSError,TimeoutError)as error:
                                    atomic_json(attempt_path,dict(status='RETRYABLE_FAILURE',requested_candidates=len(fresh),proposal_sha256=plan_hash,error=repr(error)))
                                    if attempt==args.scoring_retries:raise
                            for index,value in zip(positions,scored):feedback[index]=_compact_feedback(value)
                        for index,value in enumerate(feedback):
                            if value is None:
                                duplicate=next(j for j in range(index)if np.array_equal(requests[j],requests[index]))
                                feedback[index]=feedback[duplicate]
                        atomic_json(saved_path,dict(proposal_sha256=plan_hash,feedback=feedback))
                    retained=retain_history(old[token]);seen={value.trajectory_sha256 for value in retained};new=[]
                    for index,value in enumerate(feedback):
                        if(value['status']!='OK'or value['token']!=token or value['log']!=row['log']
                                or value['protocol_sha256']!=identity['score_protocol_sha256']
                                or value['cache']['sha256']!=old[token][0].context_sha256
                                or trajectory_hash(value['requested_trajectory'])!=trajectory_hash(requests[index])):
                            raise ValueError('New trajectory/physical feedback scene/context join changed')
                        if trajectory_hash(requests[index])in seen:continue
                        seen.add(trajectory_hash(requests[index]));new.append(CandidateRecord.from_feedback(
                            row=row,observation_hash=old[token][0].observation_hash,feedback=value,evaluator_revision=revision,
                            source='current_student_snapshot',actor_checkpoint_sha256=binding['source_model_sha256'],
                            sampling=dict(proposals['sampling'],candidate_index=index),source_parameters=binding))
                    writer.append_many(retained+new)
                    atomic_json(part/'committed'/f'{token}.json',dict(status='COMPLETE',record_count=writer.scene_count(token),
                        source_model_sha256=binding['source_model_sha256'],proposal_sha256=plan_hash))
                    progress['scenes_completed']+=1;progress['seconds']=time.monotonic()-started
                    atomic_json(part/'PROGRESS.json',progress)
            result=writer.finalize()
        atomic_json(part/'COST.json',dict(progress,**counts,status='COMPLETE',records=result['record_count'],simulation_attempts=score_attempts))
        return 0
    finally:
        for handle in handles:handle.remove()
        service.close();parent.close()


def publish_registry(bank_root,registry):
    bank=CandidateBank(bank_root,allowed_splits=('TRAIN',));manifest=bank.manifest;bank.close();identity=manifest['identity']
    if identity.get('qualification')or identity.get('refresh_schema')!='world_postcot_current_student_refresh_v1'or identity.get('refresh_recipe')!=RECIPE:
        raise ValueError('Not a complete registered own-policy refresh bank')
    if identity['source_updates']not in REFRESH_STEPS:raise ValueError('Refresh milestone exceeds fixed nine-source budget')
    data=Path(ROOT/'data/TRAIN/manifest.json');raw=json.loads(data.read_text())
    if(identity['source_data_manifest_sha256']!=file_sha(data)or set(manifest['scene_ids'])!=set(raw['tokens'])
            or max(manifest['candidate_counts'].values())>16):raise ValueError('Registry cannot publish partial/foreign/oversized training bank')
    entry={name:identity[name]for name in('source_updates','source_checkpoint','source_model_sha256','parent_bank_hash')}
    entry.update(bank_root=str(Path(bank_root).resolve()),bank_hash=manifest['sha256'],
        data_manifest_sha256=identity['source_data_manifest_sha256'],score_protocol_sha256=identity['score_protocol_sha256'])
    registry=Path(registry);registry.parent.mkdir(parents=True,exist_ok=True)
    with(registry.with_suffix('.lock')).open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        value=json.loads(registry.read_text())if registry.exists()else dict(schema=REGISTRY_SCHEMA,entries=[])
        body=dict(value);seal=body.pop('sha256',None)
        if body['schema']!=REGISTRY_SCHEMA or(seal is not None and fingerprint(body)!=seal):raise ValueError('Registry seal/schema changed')
        initial=json.loads((ROOT/'candidates/full_train_initial_v1/MANIFEST.json').read_text())['sha256']
        known={initial}|{row['bank_hash']for row in body['entries']}
        if entry['parent_bank_hash']not in known:raise ValueError('Refresh parent lineage is not rooted in the fixed initial bank')
        previous=[row for row in body['entries']if row['source_updates']==entry['source_updates']]
        if previous:
            if previous!=[entry]:raise ValueError('A frozen refresh milestone cannot be replaced')
            return entry
        body['entries'].append(entry);body['entries'].sort(key=lambda row:row['source_updates'])
        if len(body['entries'])>9:raise ValueError('Refresh budget exceeded')
        body['sha256']=fingerprint(body);atomic_json(registry,body)
    return entry


def finalize(args):
    root=Path(args.output);parts=[root/'parts'/f'rank{rank:05d}'for rank in range(args.world)]
    bindings=[json.loads((part/'BUILD_IDENTITY.json').read_text())for part in parts]
    identity=bindings[0]['identity']
    if any(value['identity']!=identity for value in bindings):raise ValueError('Refresh producers changed source identities')
    tokens=identity.get('diagnostic_scene_ids',json.loads((Path(args.data)/'manifest.json').read_text())['tokens'])
    expected=[tokens[rank::args.world]for rank in range(args.world)]
    if any(value['scene_ids']!=selected for value,selected in zip(bindings,expected)):raise ValueError('Refresh producer ownership changed')
    merge_bank_parts(root,parts,identity=identity,scene_ids=tokens)
    if identity.get('qualification'):
        print(json.dumps(dict(status='DIAGNOSTIC_COMPLETE',registry_published=False,scenes=len(tokens))));return 0
    entry=publish_registry(root,args.registry);print(json.dumps(dict(status='COMPLETE',registry_entry=entry)));return 0


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--phase',choices=['build','finalize'],default='build');parser.add_argument('--source-checkpoint')
    parser.add_argument('--export');parser.add_argument('--data',default=str(ROOT/'data/TRAIN'));parser.add_argument('--output',required=True)
    parser.add_argument('--registry',default=str(ROOT/'candidate_refresh/REGISTRY.json'))
    parser.add_argument('--rank',type=int,default=0);parser.add_argument('--world',type=int,default=1)
    parser.add_argument('--max-new-candidates',type=int,default=181780);parser.add_argument('--max-simulation-attempts','--max-simulation-budget',dest='max_simulation_attempts',type=int,default=545340)
    parser.add_argument('--scoring-retries',type=int,default=2);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--qualification',action='store_true');parser.add_argument('--max-scenes',type=int)
    args=parser.parse_args()
    if not 0<=args.rank<args.world or args.world>8 or args.scoring_retries<0 or min(args.max_new_candidates,args.max_simulation_attempts)<1:
        parser.error('Refresh is bounded to <=8 producers,2 unique trajectories/scene and finite retry attempts')
    if args.phase=='build'and(not args.source_checkpoint or not args.export):parser.error('A pinned formal source checkpoint and its strict export are required')
    if args.qualification:
        if args.max_scenes is None or not 1<=args.max_scenes<=32:parser.error('Diagnostic refresh must have explicit1..32scene budget')
    elif args.max_scenes is not None:parser.error('Formal refresh must cover the full unchanged TRAIN role')
    try:return refresh(args)if args.phase=='build'else finalize(args)
    except BaseException:
        path=Path(args.output)/'failures'/f'rank{args.rank:05d}_{time.time_ns()}.json'
        atomic_json(path,dict(status='FAILED',error=traceback.format_exc(),arguments=vars(args)));raise


if __name__=='__main__':raise SystemExit(main())
