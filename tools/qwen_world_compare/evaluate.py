"""A/B/C/D single-candidate canonical DEV/Navtest workers and full summaries.

The native scorer, trajectory decoding, exact-scene validation and cluster
bootstrap are reused.  Only the architecture loader, model identity and cost
journal are new; no alternate score or candidate selection is introduced.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

if __package__ in (None, ''):
    script = Path(__file__).resolve()
    sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != script.parent]
    sys.path.insert(0, str(script.parents[2]))

import numpy as np
import torch

from starVLA.rl.qwen_world_compare.contracts import require_dispatch, boundary_stop_due, fingerprint, file_sha
from starVLA.rl.qwen_world_compare.deployment import load_export, load_native_policy
from starVLA.rl.qwen_world_compare.parent_selection import validate_evaluation_selection
from starVLA.rl.qwen_world_postcot.data import WorldSceneData, POPULATION
from starVLA.rl.qwen_world_postcot.checkpoint import atomic_json
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
from starVLA.rl.simwam_recipe.contracts import seed_for
from starVLA.rl.simwam_recipe.reward_readouts import RewardService
from scripts.analysis.qwen_sourceflow.evaluate_policy import evaluation_caches, preflight_caches, json_feedback
from tools.qwen_world_postcot.evaluate import validate_population, validate_feedback, validate_scene_row
from tools.qwen_world_postcot.summarize import METRICS, physical_log, paired_statistics

EVALUATION_SCHEMA = 'qwen_world_compare_evaluation_v1'


def read_evaluation_identity(root):
    identity = json.loads((Path(root)/'identity.json').read_text())
    body = dict(identity); saved = body.pop('sha256', None)
    if (fingerprint(body) != saved or body.get('schema') != EVALUATION_SCHEMA
            or body.get('architecture_id') not in ('A', 'B', 'C', 'D')
            or type(body.get('qualification')) is not bool
            or body.get('cohort') not in ('DEV', 'NAVTEST')
            or body.get('inference') != 'single_candidate_cleanx_K10_shift5'
            or body.get('seed') != 2101):
        raise ValueError('Changed four-framework evaluation schema/protocol')
    return identity


def load_evaluation(root, *, population=None):
    root = Path(root); identity = read_evaluation_identity(root)
    population = json.loads(POPULATION.read_text()) if population is None else population
    expected = population['dev_tokens' if identity['cohort'] == 'DEV' else 'navtest_tokens']
    tokens = identity['tokens']
    wanted = expected[:len(tokens)] if identity['qualification'] else expected
    if (not tokens or tokens != wanted or len(tokens) != len(set(tokens))
            or identity['count'] != len(tokens) or identity['token_sha256'] != fingerprint(tokens)):
        raise ValueError('Full evaluation must retain the exact fixed population')
    if {p.stem for p in (root/'scenes').glob('*.json')} != set(tokens):
        raise ValueError('Evaluation is incomplete; missing/foreign scenes are not removed from the denominator')
    rows = {token:validate_scene_row(json.loads((root/'scenes'/f'{token}.json').read_text()),
        token=token, identity_sha256=identity['sha256'], protocol_sha256=identity['protocol']['sha256'],
        expected_log=population['token_logs'][token]) for token in tokens}
    return identity, rows


def summarize(evaluation, baseline=None, *, population=None):
    identity, rows = load_evaluation(evaluation, population=population)
    result = dict(status='COMPLETE', architecture_id=identity['architecture_id'],
        cohort=identity['cohort'], scenes=len(rows), scene_denominator=len(rows), invalid_scenes=0,
        physical_logs=len({physical_log(row['log']) for row in rows.values()}),
        qualification=identity['qualification'], seed=identity['seed'],
        evaluation_identity_sha256=identity['sha256'], model_sha256=identity['model_sha256'],
        protocol_sha256=identity['protocol']['sha256'], token_sha256=identity['token_sha256'],
        means={name:float(np.mean([row['feedback']['metrics'][name] for row in rows.values()])) for name in METRICS},
        evaluation=str(Path(evaluation).resolve()))
    result['PDMS'] = 100*result['means']['score']
    if baseline is not None:
        base_id, base = load_evaluation(baseline, population=population)
        for key in ('cohort', 'token_sha256', 'data_sha256', 'seed', 'inference', 'qualification', 'numerics'):
            if identity.get(key) != base_id.get(key):
                raise ValueError('Paired comparison changed its common contract: '+key)
        if identity['protocol']['sha256'] != base_id['protocol']['sha256']:
            raise ValueError('Paired comparison changed canonical scoring')
        result['paired'] = paired_statistics(rows, base)
    result['sha256'] = fingerprint(result)
    return result


def _numerics():
    return dict(deterministic=torch.are_deterministic_algorithms_enabled(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        bf16_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        flash_deterministic=os.environ.get('FLASH_ATTENTION_DETERMINISTIC'),
        model_storage='float32', model_forward='bfloat16', ode_state='float32')


def _dev_binding(path, *, meta, protocol_sha256, numerics):
    identity, _ = load_evaluation(path)
    if (identity['cohort'] != 'DEV' or identity['qualification']
            or identity['model_sha256'] != meta['model_sha256']
            or identity['architecture_id'] != meta['architecture_id']
            or identity['protocol']['sha256'] != protocol_sha256
            or identity['numerics'] != numerics):
        raise ValueError('Navtest must bind its own full fixed-seed DEV policy')
    return dict(evaluation=str(Path(path).resolve()), identity_sha256=identity['sha256'],
                Navtest_used_for_selection=False)


def _cumulative_cost(output, rank, current):
    invocations = []
    for p in (output/'invocations').glob(f'rank{rank}_*.json'):
        row = json.loads(p.read_text())
        if row.get('invocation') == current['invocation']:
            row = current
        invocations.append(row)
    if not any(row['invocation'] == current['invocation'] for row in invocations):
        invocations.append(current)
    names = ('physical_inference_attempts', 'new', 'model_load_seconds', 'forward_seconds',
             'scoring_seconds', 'wall_seconds')
    return {name:sum(float(row.get(name, 0.)) for row in invocations) for name in names}


def worker(args):
    if args.seed != 2101:
        raise ValueError('This comparison has one fixed evaluation seed2101')
    if bool(args.export) == bool(args.checkpoint):
        raise ValueError('Choose exactly one sealed export or diagnostic native checkpoint')
    if not args.export and not args.qualification:
        raise ValueError('Formal DEV/Navtest uses a sealed standalone export')
    host = require_dispatch()
    output = Path(args.output);output.mkdir(parents=True, exist_ok=True)
    (output/'scenes').mkdir(exist_ok=True);(output/'invocations').mkdir(exist_ok=True)
    lock = (output/f'rank{args.rank}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    invocation = f'rank{args.rank}_{uuid.uuid4().hex}'
    journal = output/'invocations'/f'{invocation}.json'
    started = time.monotonic();service = None
    record = dict(invocation=invocation, status='STARTING', rank=args.rank, world=args.world,
        host=host, pid=os.getpid(), start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
        new=0, reused=0, physical_inference_attempts=0, model_load_seconds=0.,
        forward_seconds=0., scoring_seconds=0., start_unix=time.time(), shared_GPU_worker_time=True)
    atomic_json(journal, record)
    try:
        data = WorldSceneData(args.data, role=args.cohort)
        population = json.loads(POPULATION.read_text())
        tokens = validate_population(data.tokens, args.cohort, population,
            qualification=args.qualification, limit=args.limit)
        if not 0 <= args.rank < args.world <= len(tokens):
            raise ValueError('Invalid fixed-population shard topology')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        os.environ.setdefault('FLASH_ATTENTION_DETERMINISTIC', '1')
        from starVLA.rl.flow_grpo.reproducibility import configure_numerics
        configure_numerics();torch.set_num_threads(4);torch.cuda.set_device(0)
        roots = evaluation_caches(args.cohort)
        preflight_caches(data, tokens[args.rank::args.world], roots)
        tick = time.monotonic()
        if args.export:
            model, meta = load_export(args.export, 'cuda')
            parent = meta['IL_parent'];processor_path = Path(args.export)/'processor'
        else:
            model, sources = load_native_policy(args.checkpoint, device='cuda')
            parent = sources['parent']
            meta = dict(architecture_id=parent['identity']['architecture_id'], IL_parent=parent,
                model_sha256=parent['model_sha256'], IL_updates=parent['seal']['updates'],
                qualification=parent['identity']['qualification'], RL_parent=None)
            from starVLA.rl.qwen_sourceflow.contracts import QWEN_PATH
            processor_path = QWEN_PATH
        record['model_load_seconds'] = time.monotonic()-tick
        if not args.qualification and meta['qualification']:
            raise ValueError('Qualification models cannot produce a formal performance result')
        processor = QwenObservationProcessor(str(processor_path))
        service = RewardService(Path.cwd(), workers=1, cache_roots=roots)
        source_root = Path(__file__).resolve().parents[2]
        source_paths = list((source_root/'starVLA/model/modules/qwen_world_compare').glob('*.py'))
        source_paths += [source_root/name for name in (
            'starVLA/rl/qwen_world_compare/loading.py', 'starVLA/rl/qwen_world_compare/deployment.py',
            'tools/qwen_world_compare/evaluate.py', 'starVLA/rl/safe_flow_improvement/v1_reward.py',
            'scripts/analysis/constrained_flow_noise/batched_canonical_score.py')]
        identity = dict(schema=EVALUATION_SCHEMA, architecture_id=meta['architecture_id'],
            model_sha256=meta['model_sha256'], parent_model_sha256=parent['model_sha256'],
            IL_training_identity_sha256=fingerprint(parent['identity']),
            RL_checkpoint_identity_sha256=meta['RL_parent']['sha256'] if meta.get('RL_parent') else None,
            cohort=args.cohort, tokens=tokens, count=len(tokens), token_sha256=fingerprint(tokens),
            data_sha256=data.sha, population_sha256=file_sha(POPULATION), seed=2101,
            protocol=service.protocol, qualification=args.qualification,
            inference='single_candidate_cleanx_K10_shift5', numerics=_numerics(),
            source={str(path.relative_to(source_root)):file_sha(path) for path in source_paths},
            policy_lineage=dict(IL_updates=meta['IL_updates'], RL_updates=meta.get('RL_updates'),
                IL_parent_model_sha256=parent['model_sha256'], prediction_type='x0'))
        if args.cohort == 'NAVTEST' and not args.qualification:
            if getattr(args,'user_interim_registration',None):
                if args.selection or args.policy_role or args.dev_evaluation or args.limit:
                    raise ValueError('Interim observation cannot also select a policy or limit Test')
                from starVLA.rl.qwen_world_compare.interim_navtest import validate_registration
                identity['user_interim_observation'] = validate_registration(args.user_interim_registration,
                    meta, data_sha256=data.sha, protocol_sha256=service.protocol['sha256'])
            else:
                if not args.selection or not args.policy_role or not args.dev_evaluation:
                    raise ValueError('Formal Navtest needs frozen DEV selection, policy role and its own DEV evaluation')
                identity['selection'] = validate_evaluation_selection(args.selection, args.policy_role, meta)
                identity['DEV_binding'] = _dev_binding(args.dev_evaluation, meta=meta,
                    protocol_sha256=service.protocol['sha256'], numerics=identity['numerics'])
        identity['sha256'] = fingerprint(identity)
        with (output/'identity.lock').open('a') as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            if (output/'identity.json').exists() and json.loads((output/'identity.json').read_text()) != identity:
                raise ValueError('Evaluation identity changed across resume/shards')
            atomic_json(output/'identity.json', identity)
        record.update(status='RUNNING', identity=identity['sha256'])
        atomic_json(journal, record)
        for token in tokens[args.rank::args.world]:
            if boundary_stop_due(host):
                record['status'] = 'STOPPED_AT_SCENE_BOUNDARY';break
            path = output/'scenes'/f'{token}.json';log = data.records[token]['log']
            if log != population['token_logs'][token]:
                raise ValueError('Scene/log differs from the sealed population')
            if path.exists():
                validate_scene_row(json.loads(path.read_text()), token=token,
                    identity_sha256=identity['sha256'], protocol_sha256=service.protocol['sha256'], expected_log=log)
                record['reused'] += 1;continue
            inputs = processor([data.observation(token)])
            ego = torch.tensor([data.records[token]['proprio']], device='cuda', dtype=torch.float32)
            noise = torch.randn(1, 8, 3, generator=torch.Generator().manual_seed(
                seed_for('native_qwen_sourceflow_ODE', 2101, token))).cuda()
            tick = time.monotonic();record['physical_inference_attempts'] += 1
            prediction = model.predict(inputs, ego, noise);torch.cuda.synchronize()
            record['forward_seconds'] += time.monotonic()-tick
            record['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            if args.native_parity_output and args.rank == 0 and record['new'] == 0:
                proof_path = Path(args.native_parity_output)
                if proof_path.exists():
                    proof = json.loads(proof_path.read_text())
                    if proof.get('model_sha256') != meta['model_sha256'] or proof.get('status') != 'PASS':
                        raise ValueError('Native/export parity receipt does not bind this model')
                else:
                    native, _ = load_native_policy(parent['path'], meta.get('RL_checkpoint'), 'cuda')
                    record['physical_inference_attempts'] += 1
                    tick = time.monotonic();reference = native.predict(inputs, ego, noise);torch.cuda.synchronize()
                    record['forward_seconds'] += time.monotonic()-tick
                    for key in ('normalized_action', 'requested_trajectory'):
                        if not torch.equal(prediction[key], reference[key]):
                            raise ValueError('Native/export actual K10 prediction differs: '+key)
                    proof_path.parent.mkdir(parents=True, exist_ok=True)
                    atomic_json(proof_path, dict(status='PASS', model_sha256=meta['model_sha256'],
                        native_model_sha256=parent['model_sha256'], architecture_id=meta['architecture_id'],
                        normalized_max_abs=0., requested_max_abs=0., current_images_only=True,
                        extra_validation_predictions=1, teacher_or_search_at_inference=False))
                    del native, reference
                    import gc
                    gc.collect();torch.cuda.empty_cache()
            requested = prediction['requested_trajectory'].cpu().numpy()
            tick = time.monotonic();feedback = service.score((token, log, requested))
            record['scoring_seconds'] += time.monotonic()-tick
            if len(feedback) != 1:
                raise ValueError('Infrastructure failure is not a reward-zero scene')
            validate_feedback(feedback[0], token=token, log=log, requested=requested[0],
                              protocol_sha256=service.protocol['sha256'])
            row = dict(token=token, log=log, identity=identity['sha256'], status='OK',
                requested_trajectory=requested[0].tolist(),
                normalized_action=prediction['normalized_action'][0].cpu().tolist(),
                feedback=json_feedback(feedback[0]))
            row['sha256'] = fingerprint(row);atomic_json(path, row)
            record['new'] += 1
            record['wall_seconds'] = time.monotonic()-started
            atomic_json(journal, record);atomic_json(output/f'rank{args.rank}.json', record)
        else:
            record['status'] = 'COMPLETE'
    except BaseException:
        record.update(status='FAILED', error=traceback.format_exc());raise
    finally:
        cleanup_error = None
        already_failed = record.get('status') == 'FAILED'
        try:
            if service is not None:
                service.close()
        except BaseException as error:
            cleanup_error = error
            record.update(status='FAILED', cleanup_error=repr(error))
        finally:
            record.update(wall_seconds=time.monotonic()-started, end_unix=time.time())
            atomic_json(journal, record)
            latest = dict(record, cumulative_worker_cost=_cumulative_cost(output, args.rank, record))
            atomic_json(output/f'rank{args.rank}.json', latest)
        if cleanup_error is not None and not already_failed:
            raise cleanup_error


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint');parser.add_argument('--export');parser.add_argument('--output', required=True)
    parser.add_argument('--cohort', choices=('DEV', 'NAVTEST'));parser.add_argument('--data')
    parser.add_argument('--rank', type=int, default=0);parser.add_argument('--world', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2101)
    parser.add_argument('--qualification', action='store_true');parser.add_argument('--limit', type=int)
    parser.add_argument('--selection');parser.add_argument('--policy-role',
        choices=('il_endpoint', 'il_selected', 'rl_endpoint', 'rl_selected'))
    parser.add_argument('--dev-evaluation');parser.add_argument('--native-parity-output')
    parser.add_argument('--user-interim-registration',help='Exact full Test observation requested by user; no parent selection')
    parser.add_argument('--summarize', action='store_true');parser.add_argument('--evaluation');parser.add_argument('--baseline')
    args = parser.parse_args()
    if args.user_interim_registration and (args.cohort!='NAVTEST' or args.qualification or args.limit
            or args.selection or args.policy_role or args.dev_evaluation or not args.export):
        parser.error('Interim Test requires one formal export, full NAVTEST and no selection/qualification')
    if args.summarize:
        if not args.evaluation:
            parser.error('--summarize requires --evaluation')
        result = summarize(args.evaluation, args.baseline)
        output = Path(args.output);output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and json.loads(output.read_text()) != result:
            raise FileExistsError('Keep the previous summary; choose a new derived version')
        atomic_json(output, result);print(json.dumps(result))
    else:
        if not args.cohort or not args.data:
            parser.error('worker requires --cohort and --data')
        worker(args)


if __name__ == '__main__':
    main()
