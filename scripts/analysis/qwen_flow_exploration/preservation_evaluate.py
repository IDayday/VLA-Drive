"""Plain merged source ActionDiT single-output deployment, no archive/dual/ref."""
import argparse
import fcntl
import gc
import json
import math
from pathlib import Path
import time
import uuid
import torch
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.rl.qwen_flow_exploration.native import NativeEnvironment
from starVLA.rl.qwen_flow_exploration.protocol import read_protocol, require_scene
from starVLA.rl.qwen_flow_exploration.preservation.deployment import export_adapters, read_adapter_package, merge_native_adapters_, DEPLOYMENT_ARMS
from starVLA.rl.simwam_recipe.contracts import seed_for
from starVLA.rl.qwen_sourceflow.contracts import fingerprint, file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


METRICS = ('score','no_at_fault_collisions','drivable_area_compliance',
           'time_to_collision_within_bound','ego_progress','comfort','driving_direction_compliance')


def verified_row(path, identity, token, seed):
    row=json.loads(path.read_text());body=dict(row)
    if (body.pop('sha256')!=fingerprint(body) or row['evaluation_identity']!=identity['sha256'] or
        row['scene_token']!=token or row['seed']!=seed or row['split']!=identity['split'] or row['status']!='OK' or
        row['scorer_config_hash']!=identity['scorer_config_hash'] or
        row['sampler_config_hash']!=identity['sampler_config_hash']):
        raise ValueError('Corrupt or mismatched single-output scene record')
    if any(not math.isfinite(float(row['native_subscores'][k])) or not 0<=float(row['native_subscores'][k])<=1 for k in METRICS):
        raise ValueError('Invalid scored scene cannot disappear from evaluation denominator')
    feedback=row['feedback']
    if (feedback['status']!='OK' or feedback['token']!=token or feedback['log']!=row['log_id'] or
        feedback['protocol_sha256']!=identity['scorer_config_hash'] or feedback['metrics']!=row['native_subscores'] or
        row['raw_reward']!=row['native_subscores']['score']):
        raise ValueError('Feedback belongs to a different scene/scorer or reward readout')
    planned=torch.tensor(row['planned_trajectory'],dtype=torch.float64)
    normalized=torch.tensor(row['normalized_action'],dtype=torch.float32)
    if (planned.shape!=(8,3) or normalized.shape!=(8,3) or not planned.isfinite().all() or not normalized.isfinite().all() or
        not torch.allclose(planned,ActionCodec.denorm_odo(normalized).double(),atol=1e-5,rtol=0) or
        not torch.equal(planned,torch.tensor(feedback['requested_trajectory'],dtype=torch.float64)) or
        row['simulated_trajectory']!=feedback['simulated_states']):
        raise ValueError('Requested encoding/trajectory differs from the actual scored request or simulated consequence')
    return row


def summarize(rows, tokens, seeds):
    grouped={token:[] for token in tokens}
    for row in rows:grouped[row['scene_token']].append(row)
    if any(len(group)!=len(seeds) or {r['seed'] for r in group}!=set(seeds) for group in grouped.values()):
        raise ValueError('Missing/duplicate scene-seed records')
    return {k:sum(sum(float(row['native_subscores'][k]) for row in group)/len(group) for group in grouped.values())/len(grouped) for k in METRICS}


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--arm',required=True,choices=['BASE',*DEPLOYMENT_ARMS])
    parser.add_argument('--seed',type=int,default=42,help='Training identity; inference seeds are fixed in common protocol')
    parser.add_argument('--checkpoint');parser.add_argument('--adapter-package')
    parser.add_argument('--split',choices=['TRAIN','DEV_SELECT','DEV_AUDIT','NAVTEST'],default='DEV_SELECT')
    parser.add_argument('--full-split',action='store_true');parser.add_argument('--qualification',action='store_true')
    parser.add_argument('--limit',type=int);parser.add_argument('--scorer-workers',type=int,default=1)
    args=parser.parse_args();job_started=time.monotonic();job_start_unix=time.time();protocol=read_protocol(args.protocol)
    if args.limit and (not args.qualification or args.limit<1):raise ValueError('Subsets are qualification only, never post-result selection')
    if args.arm=='BASE' and (args.checkpoint or args.adapter_package):raise ValueError('BASE cannot receive trained adapters')
    if args.arm!='BASE' and bool(args.checkpoint)==bool(args.adapter_package):raise ValueError('Provide exactly one complete checkpoint or pure adapter deployment package')
    tokens=(protocol['splits'][args.split] if args.full_split or args.split not in protocol['diagnostic'] else protocol['diagnostic'][args.split])
    if args.limit:tokens=tokens[:args.limit]
    for token in tokens:require_scene(protocol,token,args.split)
    seeds=list(protocol['evaluation_seeds'])
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    metadata=None;adapters=None
    if args.arm!='BASE':
        package=(Path(args.adapter_package) if args.adapter_package else
                 export_adapters(args.checkpoint,root/'deployment',protocol,arm=args.arm,seed=args.seed,qualification=args.qualification))
        adapters,metadata=read_adapter_package(package,protocol,arm=args.arm,seed=args.seed,qualification=args.qualification)
    identity=dict(schema='native_plain_merged_preservation_evaluation_v1',protocol_sha256=protocol['sha256'],
        base_model_sha256=protocol['base_model_sha256'],arm=args.arm,training_seed=args.seed,
        adapter_sha256=metadata['adapter_sha256'] if metadata else None,
        trained_updates=metadata['checkpoint_updates'] if metadata else 0,
        checkpoint_state=dict(arm=metadata['checkpoint_arm'],updates=metadata['checkpoint_updates'],
                              rounds=metadata['checkpoint_rounds']) if metadata else None,
        split=args.split,tokens_sha256=fingerprint(tokens),scenes=len(tokens),evaluation_seeds=seeds,
        scorer_config_hash=protocol['scorer']['sha256'],sampler_config_hash=fingerprint(protocol['samplers']),
        readout='single_candidate_source_shift5_ODE_K10',
        inference='single standard-Gaussian-root source shift5 ODE10; no output selection',
        root_innovations='native_qwen_sourceflow_ODE scene-bound CPU FP32 namespace',
        actor_dependencies='raw current images/navigation/legal ego; frozen complete base plus merged plain action weights',
        actor_uses_archive_dual_reference_teacher_or_training_cache=False,
        deployment_source_sha256=file_sha(Path(__file__).resolve().parents[3]/'starVLA/rl/qwen_flow_exploration/preservation/deployment.py'),
        qualification=args.qualification)
    identity['sha256']=fingerprint(identity)
    if (root/'identity.json').exists():
        if json.loads((root/'identity.json').read_text())!=identity:raise ValueError('Evaluation scientific identity changed')
    else:atomic_json(root/'identity.json',identity)
    expected={f'{token}_seed{seed}.json' for token in tokens for seed in seeds}
    if (root/'COMPLETE.json').exists():
        if {p.name for p in (root/'scenes').glob('*.json')}!=expected:raise ValueError('Complete marker has missing/extra population')
        rows=[verified_row(root/'scenes'/f'{token}_seed{seed}.json',identity,token,seed) for token in tokens for seed in seeds]
        means=summarize(rows,tokens,seeds)
        complete=json.loads((root/'COMPLETE.json').read_text())
        if complete['identity_sha256']!=identity['sha256'] or abs(complete['PDMS']-100*means['score'])>1e-10:
            raise ValueError('Complete score does not match sealed scene records')
        print(json.dumps(dict(status='REUSED_COMPLETE',PDMS=complete['PDMS'],scenes=len(tokens))));return
    attempt=uuid.uuid4().hex;invocation=root/'invocations'/attempt;invocation.mkdir(parents=True)
    started=job_started;env=None;record=dict(status='LOADING',invocation=attempt,start_unix=job_start_unix,
        identity_sha256=identity['sha256'],new=0,reused=0,physical_inference_attempts=0,completed_inferences=0,
        adapter_extraction_and_validation_seconds=time.monotonic()-job_started)
    atomic_json(invocation/'invocation.json',record)
    try:
        env=NativeEnvironment(protocol,trainable='frozen',scorer_workers=args.scorer_workers,
            ledger_path=invocation/'ledger.json',experiment_id=args.arm+'_single_merged_ODE')
        if env.actor is not None:raise AssertionError('Deployment must never construct a reference actor')
        merge=(merge_native_adapters_(env.model.policy.action_expert,adapters,expected_layers=30) if adapters is not None else
               dict(plain_ActionDiT=True,reference_actor_constructed=False,base_only=True))
        del adapters;gc.collect()
        if any('lora_A' in name or 'lora_B' in name or 'ref_action_expert' in name for name,_ in env.model.named_parameters()):
            raise AssertionError('Training/reference parameters leaked into final deployment model')
        atomic_json(root/'merge_validation.json',merge)
        env.set_behavior_identity(dict(arm=args.arm,training_seed=args.seed,adapter_sha256=identity['adapter_sha256'],
            base_model_sha256=identity['base_model_sha256'],merge_source_sha256=identity['deployment_source_sha256'],plain_deployment=True))
        record.update(status='RUNNING',model_loading_and_merge_seconds=time.monotonic()-started)
        rows=[]
        for token in tokens:
            needed=[(seed,root/'scenes'/f'{token}_seed{seed}.json') for seed in seeds]
            missing=any(not path.exists() for _,path in needed)
            condition=env.condition(token,args.split) if missing else None
            for seed,path in needed:
                if path.exists():
                    rows.append(verified_row(path,identity,token,seed));record['reused']+=1;continue
                rng=torch.Generator(device='cpu').manual_seed(seed_for('native_qwen_sourceflow_ODE',seed,token))
                noise=torch.randn((1,8,3),generator=rng,dtype=torch.float32)
                record['physical_inference_attempts']+=1;atomic_json(invocation/'invocation.json',record)
                action=env.sample_ode(condition,noise)
                record['completed_inferences']+=1;atomic_json(invocation/'invocation.json',record)
                row=env.score_actions(token,action,args.split,proposal_source=args.arm+'_plain_merged_single_ODE',seed=seed,
                    latent_or_proposal_parameters=[dict(root=noise[0].tolist(),single_candidate=True,selection='none',
                                                       adapter_sha256=identity['adapter_sha256'])])[0]
                row['evaluation_identity']=identity['sha256'];row['sha256']=fingerprint(row)
                path.parent.mkdir(exist_ok=True);atomic_json(path,row);rows.append(row);record['new']+=1
            del condition
            atomic_json(root/'progress.json',dict(completed_records=len(rows),expected_records=len(expected),new_this_invocation=record['new'],reused_this_invocation=record['reused']))
            atomic_json(invocation/'invocation.json',record)
        if {p.name for p in (root/'scenes').glob('*.json')}!=expected:raise ValueError('Wrong full scene/seed membership')
        means=summarize(rows,tokens,seeds)
        record.update(status='COMPLETE',end_unix=time.time(),wall_seconds=time.monotonic()-started)
        atomic_json(invocation/'invocation.json',record);env.ledger.save()
        invocations=[json.loads(path.read_text()) for path in (root/'invocations').glob('*/invocation.json')]
        if any(value['status'] in ('LOADING','RUNNING') for value in invocations):raise ValueError('Other invocation still owns work')
        result=dict(status='COMPLETE',identity_sha256=identity['sha256'],arm=args.arm,training_seed=args.seed,
            scenes=len(tokens),logs=len({row['log_id'] for row in rows}),seed_count=len(seeds),evaluation_seeds=seeds,
            logical_single_output_evaluations=len(rows),physical_inferences_including_retries=sum(v['physical_inference_attempts'] for v in invocations),
            completed_physical_inferences=sum(v['completed_inferences'] for v in invocations),
            new_inferences_this_invocation=record['new'],reused_records_this_invocation=record['reused'],
            PDMS=100*means['score'],means=means,qualification=args.qualification,
            model_has_no_LoRA_wrapper_or_reference=True,actor_uses_archive_dual_teacher_or_training_cache=False,
            single_output_native_ODE=True,job_elapsed_span_seconds=max(v.get('end_unix',v['start_unix']) for v in invocations)-min(v['start_unix'] for v in invocations),
            cumulative_worker_wall_seconds=sum(v.get('wall_seconds',0) for v in invocations),
            GPU_semantics='worker occupancy on possibly shared GPU, not exclusive compute seconds',
            current_invocation_ledger=env.ledger.snapshot())
        atomic_json(root/'COMPLETE.json',result)
        print(json.dumps({k:result[k] for k in ['status','arm','PDMS','scenes','seed_count']}))
    except BaseException:
        import traceback
        record.update(status='FAILED',end_unix=time.time(),wall_seconds=time.monotonic()-started,error=traceback.format_exc())
        atomic_json(invocation/'invocation.json',record);raise
    finally:
        if env is not None:env.close()


if __name__=='__main__':main()
