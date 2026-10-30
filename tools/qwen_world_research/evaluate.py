"""Native clean-x ODE evaluation for research TRAIN feedback and full Navtest.

The actor, scene seed, deployment export and canonical simulator/scorer are
reused without mathematical changes. TRAIN_DIAGNOSTIC explicitly authorizes
adaptive data feedback. NAVTEST never produces a training-selection bank.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import time
import traceback
import uuid

from starVLA.rl.qwen_world_postcot.contracts import file_sha, fingerprint

SCHEMA = 'world_rl_research_evaluation_v1'
POPULATION = Path('/mnt/project/flow-safe-improvement-20261013/population/manifest.json')
INFERENCE = dict(sampler='ODE', steps=10, shift=5.0, seed=2101,
                 scene_seed_namespace='native_qwen_sourceflow_ODE', candidates=1)
METRICS = ('score', 'no_at_fault_collisions', 'drivable_area_compliance',
           'time_to_collision_within_bound', 'ego_progress', 'comfort',
           'driving_direction_compliance')


def sealed(value):
    value = dict(value)
    if 'sha256' in value:
        raise ValueError('Seal only an unsealed body')
    return dict(value, sha256=fingerprint(value))


def read_sealed(path):
    value = json.loads(Path(path).read_text())
    body = dict(value)
    if body.pop('sha256', None) != fingerprint(body):
        raise ValueError('Changed sealed evaluation evidence: '+str(path))
    return value


def physical_log(log):
    return re.sub(r'_\d+_\d+$', '', log)


def validate_manifest(manifest, *, cohort, data_sha256, data_tokens, token_logs, population):
    """Normalize an explicit sealed diagnostic census or research stream list."""
    if cohort not in ('TRAIN_DIAGNOSTIC', 'NAVTEST'):
        raise ValueError('Explicit TRAIN_DIAGNOSTIC or NAVTEST cohort required')
    body = dict(manifest)
    digest = body.pop('sha256', None)
    if digest is not None and digest != fingerprint(body):
        raise ValueError('Evaluation manifest seal changed')
    tokens = body.get('tokens')
    if (not isinstance(tokens, list) or not tokens or any(not isinstance(t,str) for t in tokens)
            or len(set(tokens)) != len(tokens) or body.get('count',len(tokens)) != len(tokens)):
        raise ValueError('Evaluation requires an explicit unique nonempty scene census')
    if not set(tokens) <= set(data_tokens):
        raise ValueError('Evaluation census lacks current observations')
    if body.get('data_manifest_sha256',data_sha256) != data_sha256:
        raise ValueError('Evaluation data manifest binding changed')
    expected_logs = population['token_logs']
    if any(token_logs.get(t) != expected_logs.get(t) for t in tokens):
        raise ValueError('Scene/log join differs from the real population')
    if body.get('token_logs') is not None and any(body['token_logs'].get(t) != expected_logs[t] for t in tokens):
        raise ValueError('Manifest has changed scene/log identities')
    feedback_allowed = cohort == 'TRAIN_DIAGNOSTIC'
    if ('train_feedback_allowed' in body
            and body['train_feedback_allowed'] is not feedback_allowed):
        raise ValueError('Train-feedback permission contradicts evaluation cohort')
    if cohort == 'NAVTEST':
        if (tokens != population['navtest_tokens'] or len(tokens) != 12146
                or body.get('used_for_actor_update',False) is not False):
            raise ValueError('NAVTEST must retain the exact full 12146-scene census and forbid training feedback')
    else:
        legal = set(population['train_tokens']) | set(population['dev_tokens'])
        forbidden = set(population['navtest_tokens'])
        test_logs = {physical_log(expected_logs[t]) for t in forbidden}
        if (not set(tokens) <= legal or set(tokens) & forbidden
                or any(physical_log(expected_logs[t]) in test_logs for t in tokens)):
            raise ValueError('NAVTEST tokens or physical logs cannot become TRAIN diagnostic feedback')
        if body.get('used_for_actor_update') is False:
            raise ValueError('An evaluation-only snapshot cannot silently become adaptive TRAIN feedback')
    return dict(tokens=list(tokens), count=len(tokens), token_sha256=fingerprint(tokens),
                token_logs={t:expected_logs[t] for t in tokens},
                train_feedback_allowed=feedback_allowed,
                old_DEV_is_train_diagnostic=bool(set(tokens)&set(population['dev_tokens'])),
                selection_rule=body.get('selection_rule',body.get('mode','explicit fixed census')))


def verify_export_binding(meta, *, parent_checkpoint, rl_checkpoint=None):
    if meta.get('qualification') is not False or meta['IL_parent']['identity'].get('qualification') is not False:
        raise ValueError('Qualification weights cannot produce formal periodic performance evidence')
    if Path(meta['IL_parent']['path']).resolve() != Path(parent_checkpoint).resolve():
        raise ValueError('Export has a different pinned IL parent')
    if rl_checkpoint is None:
        if meta.get('RL_parent') is not None:
            raise ValueError('RL export requires its explicit native checkpoint binding')
    elif (meta.get('RL_parent') is None
          or Path(meta['RL_checkpoint']).resolve() != Path(rl_checkpoint).resolve()
          or meta['RL_parent'].get('schema') != 'world_postcot_cleanx_scope_RL_v1'
          or meta['RL_parent'].get('purpose') != 'formal'):
        raise ValueError('Research evaluation requires the exact complete formal native scope checkpoint')


def ensure_export(parent_checkpoint, rl_checkpoint, export):
    from starVLA.rl.qwen_world_postcot.deployment import export_checkpoint, inspect_export
    export = Path(export)
    export.parent.mkdir(parents=True,exist_ok=True)
    with export.with_name(export.name+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not export.exists():
            export_checkpoint(parent_checkpoint,export,rl_checkpoint=rl_checkpoint)
        meta = inspect_export(export)
        verify_export_binding(meta,parent_checkpoint=parent_checkpoint,rl_checkpoint=rl_checkpoint)
    return meta


def load_evaluation(output, *, row_validator=None):
    output = Path(output)
    identity = read_sealed(output/'identity.json')
    if identity.get('schema') != SCHEMA or identity.get('qualification') is not False:
        raise ValueError('Unknown/qualification research evaluation identity')
    cohort = identity['cohort']
    if (cohort not in ('TRAIN_DIAGNOSTIC','NAVTEST')
            or identity['train_feedback_allowed'] is not (cohort=='TRAIN_DIAGNOSTIC')
            or identity['inference'] != INFERENCE or identity['seed'] != 2101):
        raise ValueError('Evaluation role or deployment sampler contract changed')
    tokens = identity['tokens']
    if (not tokens or len(set(tokens))!=len(tokens) or len(tokens)!=identity['count']
            or fingerprint(tokens)!=identity['token_sha256']):
        raise ValueError('Changed evaluation scene census')
    if cohort=='NAVTEST' and len(tokens)!=12146:
        raise ValueError('Partial Navtest is not a complete evaluation')
    if identity.get('population_sha256'):
        if file_sha(POPULATION)!=identity['population_sha256']:
            raise ValueError('Canonical population changed after evaluation')
        population=json.loads(POPULATION.read_text())
        validate_manifest(dict(tokens=tokens,count=len(tokens),train_feedback_allowed=identity['train_feedback_allowed']),
            cohort=cohort,data_sha256=identity['data_sha256'],data_tokens=tokens,
            token_logs=identity['token_logs'],population=population)
    manifest = identity.get('manifest')
    if manifest and file_sha(manifest['path']) != manifest['file_sha256']:
        raise ValueError('Evaluation input manifest changed after collection')
    actual = {p.stem for p in (output/'scenes').glob('*.json')}
    if actual != set(tokens):
        raise ValueError('Missing or foreign scenes: no partial/filtered-population mean is published')
    layout = json.loads((output/'layout.json').read_text())
    for rank in range(layout['world']):
        state = json.loads((output/f'rank{rank}.json').read_text())
        if (state.get('status')!='COMPLETE' or state.get('rank')!=rank
                or state.get('world')!=layout['world'] or state.get('identity')!=identity['sha256']):
            raise ValueError('All evaluation shards must complete before aggregation')
    if row_validator is None:
        from tools.qwen_world_postcot.evaluate import validate_scene_row
        row_validator = validate_scene_row
    rows = {}
    for token in tokens:
        raw = json.loads((output/'scenes'/f'{token}.json').read_text())
        rows[token] = row_validator(raw,token=token,identity_sha256=identity['sha256'],
            protocol_sha256=identity['protocol']['sha256'],expected_log=identity['token_logs'][token])
    return identity, rows


def diagnostic_bank(identity, rows):
    if identity['cohort']!='TRAIN_DIAGNOSTIC' or identity['train_feedback_allowed'] is not True:
        raise ValueError('NAVTEST/evaluation-only results cannot enter adaptive TRAIN selection')
    if set(rows)!=set(identity['tokens']) or not rows:
        raise ValueError('Only the complete declared diagnostic census can create a feedback bank')
    return sealed(dict(schema='world_rl_research_diagnostic_bank_v1',cohort='RL_TRAIN_DIAGNOSTIC',
        train_feedback_allowed=True,data_manifest_sha256=identity['data_sha256'],
        score_units='0..1',policy_sha256=identity['model_sha256'],inference=dict(INFERENCE,
            protocol_sha256=identity['protocol']['sha256']),
        evaluation_identity_sha256=identity['sha256'],tokens_sha256=identity['token_sha256'],
        count=len(rows),records={token:dict(score=row['feedback']['metrics']['score'],
            log=row['log'],metrics=row['feedback']['metrics'],scene_receipt_sha256=row.get('sha256') or fingerprint(row))
            for token,row in rows.items()}))


def summarize(output, baseline=None, *, row_validator=None):
    identity, rows = load_evaluation(output,row_validator=row_validator)
    result = dict(schema='world_rl_research_evaluation_summary_v1',status='COMPLETE',
        cohort=identity['cohort'],scenes=len(rows),expected_scenes=identity['count'],invalid_scenes=0,
        seed=identity['seed'],train_feedback_allowed=identity['train_feedback_allowed'],
        old_DEV_is_train_diagnostic=identity['old_DEV_is_train_diagnostic'],
        model_sha256=identity['model_sha256'],parent_model_sha256=identity['parent_model_sha256'],
        IL_updates=identity.get('IL_updates'),RL_updates=identity.get('RL_updates'),
        updates=identity.get('RL_updates') if identity.get('RL_updates') is not None else identity.get('IL_updates'),
        protocol_sha256=identity['protocol']['sha256'],data_manifest_sha256=identity['data_sha256'],
        evaluation_identity_sha256=identity['sha256'],token_sha256=identity['token_sha256'],
        physical_logs=len({physical_log(r['log']) for r in rows.values()}),
        means={key:sum(r['feedback']['metrics'][key] for r in rows.values())/len(rows) for key in METRICS},
        evaluation=str(Path(output).resolve()),inference=identity['inference'])
    result['PDMS'] = 100*result['means']['score']
    if baseline:
        base_identity, base_rows = load_evaluation(baseline,row_validator=row_validator)
        for key in ('cohort','token_sha256','data_sha256','seed','inference','numerics'):
            if identity.get(key)!=base_identity.get(key):
                raise ValueError('Paired evidence differs: '+key)
        if identity['protocol']['sha256']!=base_identity['protocol']['sha256']:
            raise ValueError('Paired evidence has different canonical scorer protocol')
        from tools.qwen_world_postcot.summarize import paired_statistics
        result['paired'] = paired_statistics(rows,base_rows)
        deltas = [r['feedback']['metrics']['score']-base_rows[t]['feedback']['metrics']['score'] for t,r in rows.items()]
        result['paired'].update(G_plus_PDMS=100*sum(max(x,0) for x in deltas)/len(rows),
                                G_minus_PDMS=100*sum(min(x,0) for x in deltas)/len(rows))
    return sealed(result), identity, rows


def evaluate(args):
    import numpy as np
    import torch
    from starVLA.rl.qwen_world_postcot.contracts import require_dispatch,boundary_stop_due
    from starVLA.rl.qwen_world_postcot.checkpoint import atomic_json
    from starVLA.rl.qwen_world_postcot.deployment import load_export
    from starVLA.rl.qwen_world_postcot.data import WorldSceneData
    from starVLA.rl.simwam_recipe.contracts import seed_for
    from starVLA.rl.simwam_recipe.reward_readouts import RewardService
    from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
    from starVLA.model.modules.qwen_world_postcot.candidate_bank import trajectory_hash
    from scripts.analysis.qwen_sourceflow.evaluate_policy import evaluation_caches,preflight_caches,json_feedback
    from tools.qwen_world_postcot.evaluate import validate_feedback,validate_scene_row
    host = require_dispatch()
    if args.seed!=2101:
        raise ValueError('Periodic research comparisons keep scene sampling seed2101')
    data_manifest = json.loads((Path(args.data)/'manifest.json').read_text())
    if data_manifest.get('schema')=='world_rl_research_current_observations_v1':
        from starVLA.rl.qwen_world_research_data import ResearchSceneData
        data = ResearchSceneData(args.data,role='NAVTRAIN')
    else:
        data = WorldSceneData(args.data,role=data_manifest['cohort'])
    population = json.loads(POPULATION.read_text())
    manifest = json.loads(Path(args.manifest).read_text())
    contract = validate_manifest(manifest,cohort=args.cohort,data_sha256=data.sha,
        data_tokens=data.tokens,token_logs={t:r['log'] for t,r in data.records.items()},population=population)
    tokens = contract['tokens']
    if not 0<=args.rank<args.world<=len(tokens):
        raise ValueError('Nonempty rank/world shards must be inside the fixed census')
    roots = args.cache_root or evaluation_caches('NAVTEST' if args.cohort=='NAVTEST' else 'TRAIN')
    preflight_caches(data,tokens[args.rank::args.world],roots)
    output = Path(args.output);output.mkdir(parents=True,exist_ok=True)
    for name in ('scenes','invocations'):(output/name).mkdir(exist_ok=True)
    lock=(output/f'rank{args.rank}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    export = Path(args.export) if args.export else output.with_name(output.name+'_export')
    load_started=time.monotonic()
    ensure_export(args.parent_checkpoint,args.rl_checkpoint,export)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ.setdefault('FLASH_ATTENTION_DETERMINISTIC','1')
    from starVLA.rl.flow_grpo.reproducibility import configure_numerics
    configure_numerics();torch.set_num_threads(args.cpu_threads);torch.cuda.set_device(0)
    model,info=load_export(export,'cuda');parent=info['IL_parent']
    verify_export_binding(info,parent_checkpoint=args.parent_checkpoint,rl_checkpoint=args.rl_checkpoint)
    processor=QwenObservationProcessor(str(export/'processor'))
    service=RewardService(Path.cwd(),workers=1,cache_roots=roots)
    source_paths=[Path(__file__),Path('starVLA/rl/qwen_world_postcot/deployment.py'),
                  Path('starVLA/rl/qwen_world_postcot/snapshot_navtest.py'),
                  Path('starVLA/rl/safe_flow_improvement/v1_reward.py'),
                  Path('scripts/analysis/constrained_flow_noise/batched_canonical_score.py')]
    source_paths+=list(Path('starVLA/model/modules/qwen_world_postcot').glob('*.py'))
    source_paths+=list(Path('starVLA/rl/qwen_world_research_data').glob('*.py'))
    identity=dict(schema=SCHEMA,qualification=False,cohort=args.cohort,**contract,
        model_sha256=info['model_sha256'],parent_model_sha256=parent['model_sha256'],
        IL_updates=info.get('IL_updates',parent['seal']['updates']),RL_updates=info.get('RL_updates'),
        native_parent_checkpoint=str(Path(args.parent_checkpoint).resolve()),
        native_RL_checkpoint=str(Path(args.rl_checkpoint).resolve()) if args.rl_checkpoint else None,
        data_sha256=data.sha,population_sha256=file_sha(POPULATION),
        manifest=dict(path=str(Path(args.manifest).resolve()),file_sha256=file_sha(args.manifest)),
        seed=args.seed,inference=INFERENCE,protocol=service.protocol,
        inference_description='same native clean-x model.predict, one candidate, K10 shift5 ODE',
        research_protocol=(info.get('RL_parent') or {}).get('research_protocol'),
        numerics=dict(deterministic=torch.are_deterministic_algorithms_enabled(),
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
            bf16_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'],flash_deterministic=os.environ['FLASH_ATTENTION_DETERMINISTIC']),
        source={str(p):file_sha(p) for p in source_paths})
    if args.cohort=='NAVTEST':identity['used_for_actor_update']=False
    identity=sealed(identity)
    with (output/'identity.lock').open('a') as identity_lock:
        fcntl.flock(identity_lock,fcntl.LOCK_EX)
        if (output/'identity.json').exists() and json.loads((output/'identity.json').read_text())!=identity:
            raise ValueError('Evaluation identity changed; use a new output directory')
        atomic_json(output/'identity.json',identity)
        layout=dict(world=args.world,partition='tokens[rank::world]',scene_seed_independent_of_shard=True)
        if (output/'layout.json').exists() and json.loads((output/'layout.json').read_text())!=layout:
            raise ValueError('Active shard layout changed; requires explicit drained migration')
        atomic_json(output/'layout.json',layout)
    invocation=uuid.uuid4().hex
    record=dict(invocation=invocation,rank=args.rank,world=args.world,host=host,identity=identity['sha256'],
        start_unix=time.time(),model_loading_seconds=time.monotonic()-load_started,
        new=0,reused=0,physical_inference=0,forward_seconds=0.,scoring_seconds=0.,status='RUNNING')
    invfile=output/'invocations'/f'{invocation}.json';started=time.monotonic()
    atomic_json(invfile,record);atomic_json(output/f'rank{args.rank}.json',record)
    try:
        for token in tokens[args.rank::args.world]:
            if boundary_stop_due(host):
                record['status']='STOPPED_AT_SCENE_BOUNDARY';break
            destination=output/'scenes'/f'{token}.json';log=contract['token_logs'][token]
            if destination.exists():
                validate_scene_row(json.loads(destination.read_text()),token=token,
                    identity_sha256=identity['sha256'],protocol_sha256=service.protocol['sha256'],expected_log=log)
                record['reused']+=1;continue
            observation=data.observation(token);inputs=processor([observation])
            ego=torch.tensor([observation['proprio']],device='cuda',dtype=torch.float32)
            noise=torch.randn((1,8,3),generator=torch.Generator().manual_seed(
                seed_for('native_qwen_sourceflow_ODE',args.seed,token)),dtype=torch.float32).cuda()
            tick=time.monotonic();prediction=model.predict(inputs,ego,noise);torch.cuda.synchronize()
            record['forward_seconds']+=time.monotonic()-tick;record['physical_inference']+=1
            if args.native_parity_output and record['new']==0:
                import gc
                from starVLA.rl.qwen_world_postcot.snapshot_navtest import load_native_snapshot
                native,native_info=load_native_snapshot(parent,info)
                reference=native.predict(inputs,ego,noise);torch.cuda.synchronize()
                for key in ('normalized_action','requested_trajectory'):
                    if not torch.equal(prediction[key],reference[key]):
                        raise ValueError('Native/export prediction differs: '+key)
                atomic_json(args.native_parity_output,dict(status='PASS',token=token,
                    native_parent_checkpoint=parent['path'],native_RL_checkpoint=info.get('RL_checkpoint'),
                    model_sha256=info['model_sha256'],normalized_max_abs=0.,requested_max_abs=0.,
                    extra_validation_predictions=1,inference=INFERENCE))
                del reference,native;gc.collect();torch.cuda.empty_cache()
            requested=prediction['requested_trajectory'].cpu().numpy()
            if requested.shape!=(1,8,3) or not np.isfinite(requested).all():
                raise ValueError('Invalid native single trajectory; no replacement is scored')
            tick=time.monotonic();feedback=service.score((token,log,requested))
            record['scoring_seconds']+=time.monotonic()-tick
            if len(feedback)!=1:raise ValueError('Incomplete official feedback; no zero replacement')
            validate_feedback(feedback[0],token=token,log=log,requested=requested[0],protocol_sha256=service.protocol['sha256'])
            row=sealed(dict(token=token,log=log,identity=identity['sha256'],status='OK',invocation=invocation,
                requested_trajectory=requested[0].tolist(),normalized_action=prediction['normalized_action'][0].cpu().tolist(),
                requested_trajectory_sha256=trajectory_hash(requested[0]),feedback=json_feedback(feedback[0])))
            atomic_json(destination,row);record['new']+=1
            atomic_json(invfile,record);atomic_json(output/f'rank{args.rank}.json',record)
        else:record['status']='COMPLETE'
    except BaseException:
        record.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        record.update(wall_seconds=time.monotonic()-started,end_unix=time.time(),
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        atomic_json(invfile,record);atomic_json(output/f'rank{args.rank}.json',record);service.close()
    print(json.dumps(record),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--summarize',action='store_true')
    mode.add_argument('--export-only',action='store_true')
    parser.add_argument('--parent-checkpoint');parser.add_argument('--rl-checkpoint');parser.add_argument('--export')
    parser.add_argument('--data');parser.add_argument('--manifest')
    parser.add_argument('--cohort',choices=['TRAIN_DIAGNOSTIC','NAVTEST'])
    parser.add_argument('--output',required=True);parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--world',type=int,default=1);parser.add_argument('--seed',type=int,default=2101)
    parser.add_argument('--cache-root',action='append');parser.add_argument('--cpu-threads',type=int,default=4)
    parser.add_argument('--native-parity-output');parser.add_argument('--baseline')
    parser.add_argument('--training-bank');parser.add_argument('--summary-output')
    args=parser.parse_args()
    if args.summarize:
        from starVLA.rl.qwen_world_postcot.checkpoint import atomic_json
        result,identity,rows=summarize(args.output,args.baseline)
        destination=Path(args.summary_output) if args.summary_output else Path(args.output)/'SUMMARY.json'
        if destination.exists() and json.loads(destination.read_text())!=result:
            raise FileExistsError('Keep previous summary immutable; choose a new output')
        atomic_json(destination,result)
        if args.training_bank:
            bank=diagnostic_bank(identity,rows)
            path=Path(args.training_bank);path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists() and json.loads(path.read_text())!=bank:
                raise FileExistsError('Keep previous adaptive bank immutable')
            atomic_json(path,bank)
        print(json.dumps(result),flush=True);return
    if not args.parent_checkpoint:
        parser.error('--parent-checkpoint is required for native model binding')
    if args.export_only:
        from starVLA.rl.qwen_world_postcot.contracts import require_dispatch
        require_dispatch()
        meta=ensure_export(args.parent_checkpoint,args.rl_checkpoint,args.export or args.output)
        print(json.dumps(dict(status='COMPLETE',model_sha256=meta['model_sha256'])));return
    if not all((args.data,args.manifest,args.cohort)):
        parser.error('--data, --manifest and --cohort are required for evaluation')
    if args.native_parity_output and args.rank!=0:
        parser.error('Native/export parity is a single rank0 proof')
    evaluate(args)


if __name__=='__main__':main()
