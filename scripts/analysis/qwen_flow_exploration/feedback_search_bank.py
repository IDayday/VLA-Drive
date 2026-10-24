"""CPU-only F0-F4 search from the shared real native K32 candidate bank.

No model is loaded and no GPU is selected. F0 and the first eight candidates
reuse the common bank with their historical logical cost explicitly retained.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import resource
import time
import threading
import traceback
import uuid

import numpy as np
import torch

from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_sourceflow.contracts import file_sha, fingerprint
from starVLA.rl.simwam_recipe.reward_readouts import RewardService
from starVLA.rl.qwen_flow_exploration.protocol import read_protocol, require_dispatch, require_scene, require_budget_lock
from starVLA.rl.qwen_flow_exploration.feedback.process import NoveltyScale
from starVLA.rl.qwen_flow_exploration.feedback.process import process_summary
from starVLA.rl.qwen_flow_exploration.feedback.search import METHODS, discovery_budget_curve, run_search, select_teacher_candidates


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def indexed_paths(index, split):
    value = index.get('splits', index)[split]
    if isinstance(value, dict):
        value = list(value.values())
    result = []
    for row in value:
        path = row['path'] if isinstance(row, dict) else row
        result.append(Path(path))
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--protocol', required=True)
    p.add_argument('--bank-index', required=True)
    p.add_argument('--split', choices=['TRAIN', 'DEV_SELECT'], required=True)
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--novelty-scale')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--world', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Explicit CPU-only execution required; do not allocate unassigned GPUs')
    if a.workers < 1 or not 0 <= a.rank < a.world:
        raise ValueError('Invalid explicit CPU/shard allocation')
    require_dispatch()
    protocol = read_protocol(a.protocol)
    require_budget_lock(protocol,phase='DIAGNOSTIC_LOCKED')
    torch.set_num_threads(1)
    index_path = Path(a.bank_index)
    index = json.loads(index_path.read_text())
    paths = indexed_paths(index, a.split)[a.rank::a.world]
    if not paths:
        raise ValueError('No registered bank scenes in this shard')
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(schema='native_feedback_search_bank_v1', split=a.split, method=a.method,
                    bank_index_sha256=file_sha(index_path), seed=a.seed, rank=a.rank, world=a.world,
                    common_protocol_sha256=protocol['sha256'],
                    logical_candidate_budget_per_scene=32, source_file_sha256=file_sha(__file__))
    scale = None
    if a.method == 'F4_NOVELTY':
        if not a.novelty_scale:
            raise ValueError('F4 requires frozen TRAIN-only normalization')
        value = json.loads(Path(a.novelty_scale).read_text())
        body = dict(value)
        if (body.pop('sha256') != fingerprint(body) or value.get('fitted_split') != 'TRAIN' or
                value['common_protocol_sha256'] != protocol['sha256'] or
                value['base_model_sha256'] != protocol['base_model_sha256']):
            raise ValueError('Changed/non-TRAIN/other-base novelty calibration')
        scale = NoveltyScale(value['center'], value['scale'])
        identity['novelty_scale_sha256'] = file_sha(a.novelty_scale)
    identity['sha256'] = fingerprint(identity)
    identity_path = output/f'identity_rank{a.rank:03d}.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Refusing changed search identity')
    atomic_json(identity_path, identity)
    service = None if a.method == 'F0_IID' else RewardService(Path.cwd(), workers=a.workers,
                cache_roots=protocol['metric_cache_roots'])
    if service is not None and service.protocol['sha256'] != protocol['scorer']['sha256']:
        raise ValueError('Shared canonical protocol differs')
    started = time.monotonic()
    child_cpu0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    attempt = uuid.uuid4().hex
    invocation = dict(schema='native_feedback_CPU_search_invocation_v1',attempt=attempt,identity=identity['sha256'],
                      rank=a.rank,world=a.world,start_unix=time.time(),status='RUNNING',
                      physical_candidate_score_attempts=0,scorer_batch_call_attempts=0,scorer_wait_worker_seconds=0.)
    invfile = output/'invocations'/f'rank{a.rank:03d}_{attempt}.json'
    invfile.parent.mkdir(exist_ok=True)
    atomic_json(invfile,invocation)
    invlock = threading.Lock()

    def scene(path):
        bank = torch.load(path, map_location='cpu', weights_only=False)
        if bank['split'] != a.split or len(bank['candidate_records']) != 32:
            raise ValueError('Shared bank population/shape differs')
        token = bank['scene_token']
        require_scene(protocol,token,a.split)
        destination = output/'scenes'/f'{token}.json'
        if destination.exists():
            old = json.loads(destination.read_text())
            if old['identity'] != identity['sha256'] or old['status'] != 'COMPLETE':
                raise ValueError('Invalid reused search result')
            return dict(token=token, reused=True, costs=old['result']['costs'])
        rows = bank['candidate_records']
        if any(row['base_checkpoint_id'] != protocol['base_model_sha256'] for row in rows):
            raise ValueError('Candidate bank is not the single common base')
        feedback0 = rows[0]['feedback']
        if service is not None and feedback0['protocol_sha256'] != service.protocol['sha256']:
            raise ValueError('Common and search canonical scoring protocols differ')
        proprio = np.asarray(bank['proprio'], dtype=np.float64)
        if proprio.shape != (8,) or not np.isfinite(proprio).all():
            raise ValueError('Expected current vx/vy/ax/ay/4command proprio')
        scorer_cost = dict(actual_candidate_scores=0, scorer_batch_calls=0)

        def score_requests(requests):
            with invlock:
                invocation['physical_candidate_score_attempts'] += len(requests)
                invocation['scorer_batch_call_attempts'] += 1
            scoring_started = time.monotonic()
            try:
                records = service.score((token, bank['log_id'], requests))
            finally:
                with invlock:
                    invocation['scorer_wait_worker_seconds'] += time.monotonic()-scoring_started
            scorer_cost['actual_candidate_scores'] += len(records)
            scorer_cost['scorer_batch_calls'] += 1
            result = []
            for requested, feedback in zip(requests, records):
                if feedback['status'] != 'OK':
                    raise ValueError('Scorer exception is not a negative search sample')
                normalized = ActionCodec.norm_odo(torch.tensor(requested, dtype=torch.float32))
                back = ActionCodec.denorm_odo(normalized).numpy()
                if float(np.abs(back-requested).max()) > 1e-5:
                    raise ValueError('Native action codec roundtrip differs beyond tolerance')
                result.append(dict(scene_token=token, log_id=bank['log_id'], split=a.split,
                    checkpoint_id=bank['checkpoint_id'],base_checkpoint_id=protocol['base_model_sha256'],condition_id=bank['condition_id'],
                    experiment_id=identity['sha256'], seed=a.seed,
                    planned_trajectory=feedback['requested_trajectory'], simulated_trajectory=feedback['simulated_states'],
                    normalized_action=normalized.numpy(), raw_reward=float(feedback['metrics']['score']),
                    native_subscores=feedback['metrics'], feedback=feedback, status='OK',
                    scorer_config_hash=service.protocol['sha256'],candidate_attempts=1,scorer_calls=1,
                    scorer_batch_calls_amortized=1/len(records),model_forward_calls=0,
                    wall_time=float(feedback['scoring_seconds'])))
            return result

        def iid(round_index, count):
            assert count == 8
            return rows[8+8*round_index:16+8*round_index]

        seed = int(fingerprint(dict(seed=a.seed,token=token,namespace='shared_feedback_search'))[:16],16) % (2**32)
        result = run_search(a.method, rows[:8], initial_velocity_xy=proprio[:2],
                            sample_iid=iid, score_requests=score_requests, seed=seed, novelty_scale=scale)
        # Posthoc scoring diagnostics are deliberately outside F2's algorithm.
        for row in result['candidates']:
            if row.get('raw_reward') is None:
                row['physical_feedback_available_mask'] = dict(simulated_dynamics=False,non_drivable_mask=False)
                row['physical_feedback'] = None
                continue
            information = process_summary(row['feedback'])
            row['physical_feedback_available_mask'] = information['available_mask']
            row['physical_feedback'] = information['physical_feedback']
            row['feasibility_flags'] = dict(official_NC_DAC_TTC_all_pass=information['feasible'])
            if row['external_proposal']:
                row['sampler_config_hash'] = fingerprint(dict(method=a.method,proposal=result['proposal_identity']))
                row['latent_or_proposal_parameters'] = row['parameters']
        result['costs'].update(model_forward_calls=0,
            actual_new_candidate_scores=scorer_cost['actual_candidate_scores'], actual_new_scorer_batch_calls=scorer_cost['scorer_batch_calls'],
            logical_total_candidate_scores=32, cached_logical_candidate_scores=32 if a.method == 'F0_IID' else 8,
            common_bank_historical_cost_ledger=bank['cost_ledger'])
        curve = discovery_budget_curve(result, reference_reward=float(rows[0]['raw_reward']))
        teacher = select_teacher_candidates(result['candidates'],reference_reward=float(rows[0]['raw_reward']),split='TRAIN') if a.split == 'TRAIN' else []
        record = dict(status='COMPLETE',identity=identity['sha256'],scene_token=token,log_id=bank['log_id'],split=a.split,
                      common_protocol_sha256=protocol['sha256'],base_model_sha256=protocol['base_model_sha256'],
                      common_bank_file_sha256=file_sha(path),result=jsonable(result),budget_curve=curve,
                      teacher_candidates=jsonable(teacher),teacher_update_eligible=a.split == 'TRAIN',
                      no_GT_or_external_replacement_in_GRPO=True)
        record['sha256'] = fingerprint(record)
        destination.parent.mkdir(exist_ok=True)
        atomic_json(destination,record)
        return dict(token=token,reused=False,costs=result['costs'],teacher_count=len(teacher),budget_curve=curve)

    try:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            completed = list(pool.map(scene,paths))
        invocation['status'] = 'COMPLETE'
    except BaseException:
        invocation.update(status='FAILED',error=traceback.format_exc())
        raise
    finally:
        if service is not None:
            service.close()
        child_cpu1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        invocation.update(wall_seconds=time.monotonic()-started,end_unix=time.time(),
            scorer_child_CPU_seconds=(child_cpu1.ru_utime+child_cpu1.ru_stime)-(child_cpu0.ru_utime+child_cpu0.ru_stime))
        atomic_json(invfile,invocation)
    history = [json.loads(path.read_text()) for path in (output/'invocations').glob(f'rank{a.rank:03d}_*.json')]
    summary = dict(status='COMPLETE',identity=identity['sha256'],scenes=len(completed),split=a.split,method=a.method,
                   logical_candidate_scores=32*len(completed),
                   actual_new_candidate_scores=sum(row['costs']['actual_new_candidate_scores'] if not row['reused'] else 0 for row in completed),
                   actual_new_scorer_batch_calls=sum(row['costs']['actual_new_scorer_batch_calls'] if not row['reused'] else 0 for row in completed),
                   reused_scenes=sum(row['reused'] for row in completed),wall_seconds=time.monotonic()-started,
                   oracle_is_not_deployment=True,worker_threads=a.workers,invocation=str(invfile),
                   cumulative_physical_candidate_score_attempts=sum(row['physical_candidate_score_attempts'] for row in history),
                   cumulative_scoring_batch_call_attempts=sum(row['scorer_batch_call_attempts'] for row in history),
                   cumulative_scorer_child_CPU_seconds=sum(row['scorer_child_CPU_seconds'] for row in history if row['status'] != 'RUNNING'),
                   cumulative_CPU_controller_wall_seconds=sum(row['wall_seconds'] for row in history if row['status'] != 'RUNNING'))
    atomic_json(output/f'SUMMARY_rank{a.rank:03d}_{attempt}.json',summary)
    atomic_json(output/f'SUMMARY_rank{a.rank:03d}.json',summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
