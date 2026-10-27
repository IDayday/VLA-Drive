"""Fixed-population raw-image single-candidate clean-x K10 canonical evaluation."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from starVLA.rl.qwen_world_postcot.contracts import (
    require_dispatch, boundary_stop_due, fingerprint, file_sha)
from starVLA.rl.qwen_world_postcot.data import WorldSceneData, POPULATION
from starVLA.rl.qwen_world_postcot.loading import load_deployment_checkpoint, validate_rl_parent_selection
from starVLA.rl.qwen_world_postcot.deployment import load_export
from starVLA.rl.qwen_world_postcot.checkpoint import atomic_json
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
from starVLA.model.modules.qwen_world_postcot.candidate_bank import trajectory_hash
from starVLA.model.modules.qwen_world_postcot.post_action import OFFICIAL_METRIC_NAMES
from starVLA.rl.simwam_recipe.contracts import seed_for
from starVLA.rl.simwam_recipe.reward_readouts import RewardService
from scripts.analysis.qwen_sourceflow.evaluate_policy import evaluation_caches, preflight_caches, json_feedback


def validate_population(tokens, cohort, population, *, qualification=False, limit=None):
    if cohort not in ('DEV', 'NAVTEST') or not isinstance(qualification, bool):
        raise ValueError('Explicit DEV/NAVTEST role and qualification status required')
    expected = population['dev_tokens' if cohort == 'DEV' else 'navtest_tokens']
    if list(tokens) != list(expected) or len(set(tokens)) != len(tokens) or not tokens:
        raise ValueError('Evaluation manifest must be the entire fixed role, with unchanged scene order')
    if limit is not None and (not qualification or limit <= 0 or limit > len(tokens)):
        raise ValueError('Only explicit qualification may limit a positive fixed population prefix')
    return list(tokens if limit is None else tokens[:limit])


def validate_feedback(feedback, *, token, log, requested, protocol_sha256):
    if (feedback.get('status') != 'OK' or feedback.get('token') != token
            or feedback.get('log') != log or feedback.get('protocol_sha256') != protocol_sha256):
        raise ValueError('Official feedback scene/log/protocol join changed')
    if trajectory_hash(feedback['requested_trajectory']) != trajectory_hash(requested):
        raise ValueError('Official labels are not attached to the exact submitted FP32 trajectory')
    metrics = feedback['metrics']
    for name in (*OFFICIAL_METRIC_NAMES, 'score'):
        value = metrics[name]
        if not isinstance(value, (int, float, np.number)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid official metric is not a valid zero label: '+name)
    expected = metrics['no_at_fault_collisions']*metrics['drivable_area_compliance']*(
        5*metrics['ego_progress']+5*metrics['time_to_collision_within_bound']+2*metrics['comfort'])/12
    if abs(metrics['score']-expected) > 5e-15:
        raise ValueError('Canonical v1.1 5:5:2 aggregation changed')
    if not feedback.get('cache', {}).get('sha256'):
        raise ValueError('Official feedback has no immutable real metric-context identity')


def validate_scene_row(row, *, token, identity_sha256, protocol_sha256, expected_log):
    body = dict(row); saved = body.pop('sha256', None)
    if (fingerprint(body) != saved or body.get('identity') != identity_sha256
            or body.get('token') != token or body.get('log') != expected_log or body.get('status') != 'OK'):
        raise ValueError('Incomplete or changed prediction scene join')
    validate_feedback(body['feedback'], token=token, log=expected_log,
        requested=body['requested_trajectory'], protocol_sha256=protocol_sha256)
    return body


def sampling_role(cohort,seed,variation,qualification=False):
    if variation:
        if qualification or cohort!='NAVTEST' or seed not in (2102,2103):
            raise ValueError('Only preregistered post-selection NAVTEST sampling seeds2102/2103 may use the variation role')
        return 'post_selection_sampling_variation'
    if not qualification and seed!=2101:
        raise ValueError('Formal primary DEV/NAVTEST sampling seed remains2101')
    return 'qualification' if qualification else 'primary_seed2101'


def validate_dev_binding(path, *, model_sha256, protocol_sha256, seed, numerics, role='primary_seed2101'):
    # Import at call time: summarize reuses the scene validation helpers above.
    from tools.qwen_world_postcot.summarize import load
    identity, _ = load(path)
    variation=role=='post_selection_sampling_variation'and seed in (2102,2103)
    if (role not in ('primary_seed2101','post_selection_sampling_variation')
            or(role=='post_selection_sampling_variation'and not variation)
            or(role=='primary_seed2101'and seed!=2101)
            or identity['cohort'] != 'DEV' or identity['qualification']
            or identity['model_sha256'] != model_sha256
            or identity['protocol']['sha256'] != protocol_sha256
            or identity['seed'] != (2101 if variation else seed) or identity.get('numerics') != numerics
            or identity['inference'] != 'single_candidate_cleanx_K10_shift5'):
        raise ValueError('Navtest policy must be explicitly frozen through its own full canonical DEV evaluation')
    return dict(evaluation=str(Path(path).resolve()), identity_sha256=identity['sha256'],
                model_sha256=model_sha256, Navtest_used_for_selection=False,
                role=role,DEV_selection_seed=identity['seed'],evaluation_sampling_seed=seed)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint'); parser.add_argument('--export'); parser.add_argument('--output', required=True)
    parser.add_argument('--cohort', choices=['DEV', 'NAVTEST'], required=True); parser.add_argument('--data', required=True)
    parser.add_argument('--parent-selection'); parser.add_argument('--dev-evaluation')
    parser.add_argument('--user-interim-registration', help='Exact user-requested IL37k Navtest; grants no parent selection')
    parser.add_argument('--user-snapshot-registration', help='Exact user-pinned full IL/RL Test snapshot, without model selection or updates')
    parser.add_argument('--native-parity-output', help='Interim rank0 compares the actual export prediction with the native checkpoint')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world', type=int, default=1); parser.add_argument('--seed', type=int, default=2101)
    parser.add_argument('--qualification', action='store_true'); parser.add_argument('--limit', type=int)
    parser.add_argument('--post-selection-seed-variation',action='store_true',
                        help='Locked model only: additional NAVTEST sampling seed2102/2103; DEV selection stays2101')
    args = parser.parse_args()
    if args.user_interim_registration and (args.cohort != 'NAVTEST' or args.qualification or args.limit
            or args.seed != 2101 or args.post_selection_seed_variation or args.parent_selection
            or args.dev_evaluation or not args.export):
        raise ValueError('User interim role requires full seed2101 IL-export Navtest, without selection/qualification')
    if args.user_snapshot_registration and (args.user_interim_registration or args.cohort!='NAVTEST'
            or args.qualification or args.limit or args.seed!=2101 or args.post_selection_seed_variation
            or args.parent_selection or args.dev_evaluation or not args.export):
        raise ValueError('User snapshot requires one fixed full seed2101 Test export; no selection/qualification')
    if args.native_parity_output and (not (args.user_interim_registration or args.user_snapshot_registration) or args.rank != 0):
        raise ValueError('Native parity output is only for interim rank0')
    if bool(args.checkpoint) == bool(args.export):
        raise ValueError('Choose exactly one complete IL checkpoint or sealed deployment export')
    role=sampling_role(args.cohort,args.seed,args.post_selection_seed_variation,args.qualification)
    host = require_dispatch()
    data = WorldSceneData(args.data, role=args.cohort)
    population = json.loads(POPULATION.read_text())
    tokens = validate_population(data.tokens, args.cohort, population,
        qualification=args.qualification, limit=args.limit)
    if not 0 <= args.rank < args.world <= len(tokens):
        raise ValueError('Evaluation shard topology must be nonempty and inside the fixed population')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('FLASH_ATTENTION_DETERMINISTIC', '1')
    from starVLA.rl.flow_grpo.reproducibility import configure_numerics
    configure_numerics(); torch.set_num_threads(4); torch.cuda.set_device(0)
    roots = evaluation_caches(args.cohort)
    preflight_caches(data, tokens[args.rank::args.world], roots)
    model, info = load_export(args.export, 'cuda') if args.export else load_deployment_checkpoint(args.checkpoint, 'cuda')
    parent = info['IL_parent'] if args.export else info
    if not args.qualification and (parent['identity']['qualification'] or info.get('qualification')):
        raise ValueError('Diagnostic checkpoint cannot become a formal result')
    if args.cohort == 'NAVTEST':
        if args.user_snapshot_registration:
            pass  # Exact native/export/data authorization is validated below.
        elif args.user_interim_registration:
            if info.get('RL_parent') is not None: raise ValueError('This interim grant is IL-only')
        else:
            if not args.parent_selection:
                raise ValueError('Navtest waits for frozen DEV-only model selection')
            validate_rl_parent_selection(args.parent_selection, parent)
    processor = QwenObservationProcessor(str(Path(args.export)/'processor') if args.export
        else '/mnt/project/DriveDreamer-Policy/models/Qwen3-VL-2B-Instruct')
    service = RewardService(Path.cwd(), workers=1, cache_roots=roots)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True); (output/'scenes').mkdir(exist_ok=True)
    lock = (output/f'rank{args.rank}.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source_paths = list(Path('starVLA/model/modules/qwen_world_postcot').glob('*.py')) + [Path(path) for path in (
        'starVLA/rl/qwen_world_postcot/deployment.py', 'starVLA/rl/qwen_world_postcot/rl_actor.py',
        'starVLA/rl/qwen_world_postcot/loading.py', 'tools/qwen_world_postcot/evaluate.py',
        'starVLA/rl/safe_flow_improvement/v1_reward.py',
        'scripts/analysis/constrained_flow_noise/batched_canonical_score.py')]
    if args.user_interim_registration:
        source_paths.append(Path('starVLA/rl/qwen_world_postcot/interim_navtest.py'))
    if args.user_snapshot_registration:
        source_paths += [Path('starVLA/rl/qwen_world_postcot/snapshot_navtest.py'),
                         Path('starVLA/rl/qwen_world_postcot/scope_actor.py')]
    identity = dict(schema='world_postcot_evaluation_v1', model_sha256=info['model_sha256'],
        parent_model_sha256=parent['model_sha256'], cohort=args.cohort, tokens=tokens,
        token_sha256=fingerprint(tokens), count=len(tokens), data_sha256=data.sha,
        population_sha256=file_sha(POPULATION), seed=args.seed, protocol=service.protocol,
        qualification=args.qualification, sampling_role=role,inference='single_candidate_cleanx_K10_shift5',
        physical_logs=len({population['token_logs'][token].rsplit('_', 2)[0] for token in tokens}),
        numerics=dict(deterministic=torch.are_deterministic_algorithms_enabled(),
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
            bf16_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'], flash_deterministic=os.environ['FLASH_ATTENTION_DETERMINISTIC']),
        source={str(path): file_sha(path) for path in source_paths})
    identity['policy_lineage']=dict(IL_updates=info.get('IL_updates',parent['seal']['updates']),
        RL_updates=info.get('RL_updates'),IL_parent_model_sha256=parent['model_sha256'],
        precision='FP32 storage/master; BF16 Qwen and action forward; FP32 clean-x states/ODE',
        prediction_type='x0',action_config=parent['identity']['model_config']['action'],
        training_population='90890 physical-log-disjoint TRAIN; teacher pretraining exposure is separately disclosed')
    if args.user_snapshot_registration:
        from starVLA.rl.qwen_world_postcot.snapshot_navtest import validate_snapshot
        identity['user_requested_snapshot_navtest'] = validate_snapshot(args.user_snapshot_registration,
            parent=parent, export=info, data_sha256=data.sha, tokens=tokens,
            population_sha256=file_sha(POPULATION), protocol_sha256=service.protocol['sha256'],
            output=output, seed=args.seed)
    elif args.user_interim_registration:
        from starVLA.rl.qwen_world_postcot.interim_navtest import validate_interim
        identity['user_requested_interim_navtest'] = validate_interim(args.user_interim_registration,
            parent=parent, model_sha256=info['model_sha256'], data_sha256=data.sha, tokens=tokens,
            population_sha256=file_sha(POPULATION), protocol_sha256=service.protocol['sha256'],
            output=output, seed=args.seed)
    elif args.cohort == 'NAVTEST' and not args.qualification:
        if not args.dev_evaluation:
            raise ValueError('Formal Navtest requires --dev-evaluation for this exact IL/RL policy, not only its IL parent')
        identity['DEV_binding'] = validate_dev_binding(args.dev_evaluation,
            model_sha256=info['model_sha256'], protocol_sha256=service.protocol['sha256'],
            seed=args.seed, numerics=identity['numerics'],role=role)
    identity['sha256'] = fingerprint(identity)
    with (output/'identity.lock').open('a') as identity_lock:
        fcntl.flock(identity_lock, fcntl.LOCK_EX)
        if (output/'identity.json').exists() and json.loads((output/'identity.json').read_text()) != identity:
            raise ValueError('Evaluation identity changed')
        atomic_json(output/'identity.json', identity)
    record = dict(status='RUNNING', rank=args.rank, world=args.world, host=host, new=0, reused=0,
        forward_seconds=0., scoring_seconds=0., start_unix=time.time())
    started = time.monotonic()
    try:
        for token in tokens[args.rank::args.world]:
            if boundary_stop_due(host):
                record['status'] = 'STOPPED_AT_SCENE_BOUNDARY'; break
            destination = output/'scenes'/f'{token}.json'
            log = data.records[token]['log']
            if log != population['token_logs'][token]:
                raise ValueError('Current data source log differs from the fixed population')
            if destination.exists():
                validate_scene_row(json.loads(destination.read_text()), token=token,
                    identity_sha256=identity['sha256'], protocol_sha256=service.protocol['sha256'], expected_log=log)
                record['reused'] += 1; continue
            observation = data.observation(token); inputs = processor([observation])
            ego = torch.tensor([observation['proprio']], device='cuda', dtype=torch.float32)
            noise = torch.randn(1, 8, 3, generator=torch.Generator().manual_seed(
                seed_for('native_qwen_sourceflow_ODE', args.seed, token))).cuda()
            tick = time.monotonic(); prediction = model.predict(inputs, ego, noise); torch.cuda.synchronize()
            if args.native_parity_output and record['new'] == 0:
                import gc
                if args.user_snapshot_registration:
                    from starVLA.rl.qwen_world_postcot.snapshot_navtest import load_native_snapshot
                    native, native_info = load_native_snapshot(parent, info)
                else:
                    native, native_info = load_deployment_checkpoint(parent['path'], 'cuda')
                reference = native.predict(inputs, ego, noise); torch.cuda.synchronize()
                for key in ['normalized_action', 'requested_trajectory']:
                    if not torch.equal(prediction[key], reference[key]):
                        raise ValueError('Native/export actual prediction differs: ' + key)
                atomic_json(args.native_parity_output, dict(status='PASS', checkpoint=parent['path'],
                    native_model_sha256=native_info['model_sha256'], model_sha256=info['model_sha256'],
                    normalized_max_abs=0., requested_max_abs=0., extra_validation_predictions=1,
                    current_images_only=True, teacher_or_search_at_inference=False,
                    **(dict(native_RL_checkpoint=info.get('RL_checkpoint'), native_RL_updates=info.get('RL_updates'))
                       if args.user_snapshot_registration else {})))
                del reference, native; gc.collect(); torch.cuda.empty_cache()
            record['forward_seconds'] += time.monotonic()-tick
            record['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            requested = prediction['requested_trajectory'].cpu().numpy()
            tick = time.monotonic(); feedback = service.score((token, log, requested))
            record['scoring_seconds'] += time.monotonic()-tick
            if len(feedback) != 1:
                raise ValueError('Incomplete official score; the scene denominator is retained')
            validate_feedback(feedback[0], token=token, log=log, requested=requested[0], protocol_sha256=service.protocol['sha256'])
            row = dict(token=token, log=log, identity=identity['sha256'], status='OK',
                requested_trajectory=requested[0].tolist(), normalized_action=prediction['normalized_action'][0].cpu().tolist(),
                feedback=json_feedback(feedback[0]))
            row['sha256'] = fingerprint(row); atomic_json(destination, row)
            record['new'] += 1; atomic_json(output/f'rank{args.rank}.json', record)
        else:
            record['status'] = 'COMPLETE'
    except BaseException:
        record.update(status='FAILED', error=traceback.format_exc()); raise
    finally:
        record.update(wall_seconds=time.monotonic()-started, end_unix=time.time())
        atomic_json(output/f'rank{args.rank}.json', record); service.close()


if __name__ == '__main__':
    main()
