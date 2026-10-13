"""Real bounded ODE/SDE/CEM generation plus independent canonical scoring.

The same collector supports D3 (64 logical proposals) and P/online banks (32).
All reference, valid, duplicate and physically rejected proposals are retained.
Scoring executes on the separate registered CPU host through ScoringClient.
"""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np
import torch

from scripts.analysis.flow_safe_improvement.qualify_rollout_batch import grouped
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.rollout import SamplingSpec
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.candidate_search import BoundedCEM, CEMSpec, smooth_behavior_request, safety_order
from starVLA.rl.safe_flow_improvement.contracts import PopulationContract, SFT_SHA256, file_sha, fingerprint, seed_for
from starVLA.rl.safe_flow_improvement.policy_adapter import encode_condition, sample_legacy_sde, decode_request, encode_external_request
from starVLA.rl.safe_flow_improvement.sampling import load_cached_policy, sample_scene
from starVLA.rl.safe_flow_improvement.scoring_queue import ScoringClient


def publish_scene(path, row, identity):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.attempt-'+uuid.uuid4().hex)
    with temp.open('xb') as stream:
        torch.save(row, stream)
        stream.flush(); os.fsync(stream.fileno())
    if path.exists():
        path.rename(path.with_suffix('.unsealed-'+uuid.uuid4().hex))
    os.replace(temp, path)
    atomic_json(path.with_suffix('.json'), {'identity_sha256': identity, 'payload_sha256': file_sha(path), 'token': row['token']})


def request_key(request):
    x = np.asarray(request, dtype='<f8').copy()
    x[x == 0] = 0
    import hashlib
    return hashlib.sha256(x.tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--population', required=True)
    p.add_argument('--cohorts', required=True)
    p.add_argument('--stage', choices=['D3', 'BANK', 'CALIBRATION', 'ONLINE'], required=True)
    p.add_argument('--method', choices=['ODE', 'SDE', 'LATENT', 'BEHAVIOR'], required=True)
    p.add_argument('--reference-scores', required=True)
    p.add_argument('--queue', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batch-qualification')
    p.add_argument('--physical-fallback-registration', help='Explicit 32-slot bank fallback for rejected behavior proposals')
    p.add_argument('--online-registration', help='Required fixed current-ODE/search quota and refresh contract for ONLINE')
    p.add_argument('--behavior')
    p.add_argument('--behavior-sha256')
    p.add_argument('--iteration', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after-for-qualification', type=int)
    p.add_argument('--config', default='configs/flow_grpo/frozen_action_head_batched_single8_epoch1.yaml')
    p.add_argument('--source-config', default='/mnt/project/DriveDreamer-Policy-flow-grpo/artifacts/action-only-source/checkpoint_code/action-only-checkpoints-v1/configs/frozen-visual-step100000-config.yaml')
    a = p.parse_args()
    fallback_spec = None
    if a.physical_fallback_registration:
        fallback_spec = json.loads(Path(a.physical_fallback_registration).read_text())
        if a.method != 'BEHAVIOR' or a.stage not in ('BANK', 'ONLINE') or fallback_spec['physical_failure_policy'] != 'uniform_fixed_reference_fallback':
            raise ValueError('Fallback is an explicitly registered behavior bank policy; D3 remains unchanged')
    online_spec = None
    if a.stage == 'ONLINE':
        online_spec = json.loads(Path(a.online_registration).read_text())
        if (a.iteration not in (1, 2, 3) or a.method not in ('ODE', 'BEHAVIOR')
                or online_spec['search_quota'] != dict(reference=4, current_ODE=12, behavior_search=16, search_rounds=4)
                or online_spec['ordinary_quota'] != dict(reference=4, current_ODE=28)
                or online_spec['behavior_center'] != 'safety_order_best_of_12_current_ODE'):
            raise ValueError('Online iteration must use the fixed current-policy sampling contract')
        if a.method == 'BEHAVIOR' and fallback_spec is None:
            raise ValueError('Online physical failures require the registered reference fallback')
    elif a.online_registration:
        raise ValueError('Online quota cannot silently change D3/P banks')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    configure_numerics()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    population, cohorts = [json.loads(Path(x).read_text()) for x in (a.population, a.cohorts)]
    if cohorts['population_identity'] != population['identity_sha256']:
        raise ValueError('Cohorts changed population')
    tokens = (population['calibration_tokens'] if a.stage == 'CALIBRATION'
              else cohorts['diagnostic_tokens' if a.stage == 'D3' else 'learning_tokens'])
    contract = PopulationContract(*[frozenset(population[k]) for k in ('train_tokens', 'dev_tokens', 'navtest_tokens')], population['token_logs'])
    contract.require_training(tokens, 'search/bank generation')
    budget, draws = (64, 15) if a.stage == 'D3' else (32, 7)
    mixed_online = online_spec is not None and a.method == 'BEHAVIOR'
    if mixed_online:
        draws = 4
    chunk = 1
    if a.batch_qualification:
        qualification = json.loads(Path(a.batch_qualification).read_text())
        chunk = qualification['selected_exploration_chunk']
        if qualification['status'] != 'COMPLETE' or not qualification['summaries'][str(chunk)]['passed']:
            raise ValueError('Exploration batching was not qualified')
    queue = json.loads((Path(a.queue) / 'manifest.json').read_text())
    scorer = ScoringClient(a.queue, queue['protocol_sha256'])
    policy, store, cfg, sft, behavior = load_cached_policy(a.config, a.source_config, behavior=a.behavior, behavior_sha256=a.behavior_sha256)
    from starVLA.rl.safe_flow_improvement import candidate_search, sampling, policy_adapter, scoring_queue
    from scripts.analysis.flow_safe_improvement import qualify_rollout_batch
    identity = {'schema': 'flow_safe_candidate_bank_v1', 'stage': a.stage, 'method': a.method,
                'population_identity': population['identity_sha256'], 'cohort_identity': cohorts['sha256'],
                'tokens': tokens, 'token_list_sha256': fingerprint(tokens), 'reference_SFT': SFT_SHA256,
                'behavior_sha256': behavior, 'iteration': a.iteration, 'seed': a.seed, 'feature_identity': store.identity_sha,
                'protocol_sha256': queue['protocol_sha256'], 'logical_budget_per_scene': budget, 'reference_draws': 4,
                'new_draws': budget-4, 'ODE_candidate_chunk': chunk, 'legacy_SDE_candidate_chunk': 1,
                # JSON serializes tuple bounds as lists. Normalize before
                # comparing another rank's manifest or an explicit resume.
                'search_spec': json.loads(json.dumps(asdict(CEMSpec()))), 'search_order': 'joint_NC_DAC_TTC_pass_then_max_cost_then_sum_cost_then_true_PDMS',
                'reference_scores_manifest_sha256': file_sha(Path(a.reference_scores)/'manifest.json'),
                'batch_qualification_sha256': file_sha(a.batch_qualification) if a.batch_qualification else None,
                'source_files': {Path(x).name: file_sha(x) for x in (__file__, candidate_search.__file__, sampling.__file__, policy_adapter.__file__, scoring_queue.__file__, qualify_rollout_batch.__file__)}}
    if fallback_spec:
        identity.update(physical_failure_policy='uniform_fixed_reference_fallback',
                        physical_fallback_registration_sha256=file_sha(a.physical_fallback_registration))
    if online_spec:
        identity.update(online_registration_sha256=file_sha(a.online_registration),
                        quota=online_spec['search_quota' if mixed_online else 'ordinary_quota'],
                        search_center=online_spec['behavior_center'] if mixed_online else None)
    identity['sha256'] = fingerprint(identity)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    with (out/'manifest.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (out/'manifest.json').exists():
            if json.loads((out/'manifest.json').read_text()) != identity:
                raise ValueError('Behavior, bank, search, protocol or source changed')
            if not a.resume and (out/f'rank{rank}.json').exists():
                raise ValueError('Existing bank requires explicit resume')
        else:
            atomic_json(out/'manifest.json', identity)
    lease = (out/f'rank{rank}.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = dict(status='RUNNING', identity_sha256=identity['sha256'], rank=rank, world_size=world,
                  requested_scenes=len(tokens[rank::world]), reused_scenes=0, new_scenes=0, logical_proposals=0,
                  velocity_forward_calls=0, actual_new_scoring_calls=0, reused_score_candidates=0,
                  invalid_physical_proposals=0, duplicate_proposals=0, GPU_generation_seconds=0.)
    started = time.monotonic()
    with torch.no_grad():
        for token in tokens[rank::world]:
            dst = out/'scenes'/token[:2]/(token+'.pt')
            if dst.with_suffix('.json').exists():
                seal = json.loads(dst.with_suffix('.json').read_text())
                if seal['identity_sha256'] != identity['sha256'] or seal['payload_sha256'] != file_sha(dst):
                    raise ValueError('Previously sealed bank changed')
                status['reused_scenes'] += 1
                continue
            log = population['token_logs'][token]
            reference_path = Path(a.reference_scores)/'scenes'/token[:2]/(token+'.pt')
            if reference_path.with_suffix('.json').exists():
                reference_seal = json.loads(reference_path.with_suffix('.json').read_text())
                if reference_seal['payload_sha256'] != file_sha(reference_path):
                    raise ValueError('Permanent reference changed')
                reference = torch.load(reference_path, map_location='cpu', weights_only=True)
                rs, feedback = reference['sample'], reference['feedback']
                if rs['behavior_sha256'] != SFT_SHA256 or [k['seed'] for k in rs['noise_keys']] != population['reference_seeds']:
                    raise ValueError('Reference identity/seeds changed')
                reference_new_calls, reference_reused_calls = 0, 4
            else:
                if behavior != SFT_SHA256:
                    raise ValueError('Missing permanent reference; never replace it by current behavior')
                rs = sample_scene(policy, store, token, population['reference_seeds'], mode='ode', role='reference', act_norm=int(sft.datasets.vla_data.act_norm))
                feedback, receipt = scorer.score(token, log, rs['requested_trajectories'].numpy(), caller_identity='permanent_SFT_reference_'+population['identity_sha256'], request_key='reference_4')
                reference_new_calls, reference_reused_calls = (0, 4) if receipt['reused'] else (4, 0)
                status['velocity_forward_calls'] += 40
            candidates, attempts, cached = [], [], {}
            for i, f in enumerate(feedback):
                record = dict(source='reference_ode', is_reference=True, behavior_sha256=SFT_SHA256,
                              normalized_action=rs['normalized_actions'][i].cpu(), requested_trajectory=rs['requested_trajectories'][i].cpu(),
                              feedback=f, noise_keys=rs['noise_keys'][i], source_parameters={}, on_policy_for_NFT=False)
                candidates.append(record); cached[request_key(record['requested_trajectory'])] = (len(candidates)-1, f)
            raw = store.get(token)['raw'].cuda()
            condition = encode_condition(policy, raw)
            diagnostics = []
            if mixed_online:
                # Genuine fresh current-behavior samples precede repair. Their
                # keys match the first three draws in each ordinary ODE round.
                t0 = time.monotonic()
                root_keys = [seed_for('bank_initial', a.seed, token, a.iteration, r, i) for r in range(4) for i in range(3)]
                roots = torch.cat([torch.randn((1, 8, 4), device='cuda', dtype=raw.dtype,
                                               generator=torch.Generator(device='cuda').manual_seed(k)) for k in root_keys])
                seed_actions = grouped(policy, condition, roots, chunk).float().cpu()
                seed_requests = decode_request(seed_actions.numpy(), act_norm=int(sft.datasets.vla_data.act_norm))
                torch.cuda.synchronize()
                status['GPU_generation_seconds'] += time.monotonic()-t0
                status['velocity_forward_calls'] += 10*((12+chunk-1)//chunk)
                seed_feedback, receipt = scorer.score(token, log, seed_requests, caller_identity=identity['sha256'], request_key=f'{token}_current_ODE12')
                diagnostics.append({'current_behavior_ODE_scoring': receipt})
                for i, f in enumerate(seed_feedback):
                    key = request_key(seed_requests[i])
                    duplicate = cached.get(key)
                    if duplicate is not None:
                        status['duplicate_proposals'] += 1
                    record = dict(source='ode', is_reference=False, behavior_sha256=behavior,
                                  normalized_action=seed_actions[i], requested_trajectory=torch.as_tensor(seed_requests[i]),
                                  feedback=f, noise_keys={'root_key': root_keys[i], 'initial_noise': roots[i].cpu()},
                                  source_parameters={'online_center_pool': True}, on_policy_for_NFT=True,
                                  duplicate_of=duplicate[0] if duplicate is not None else None)
                    candidates.append(record); cached[key] = (len(candidates)-1, f)
            search = None
            if a.method in ('LATENT', 'BEHAVIOR'):
                search = BoundedCEM('latent_search' if a.method == 'LATENT' else 'behavior_search', np.random.default_rng(seed_for('search', a.seed, token, a.method, a.iteration)))
            base_index = min(range(4, 16) if mixed_online else range(4), key=lambda i: safety_order(candidates[i]['feedback']['metrics']))
            base_request = np.asarray(candidates[base_index]['requested_trajectory'])
            state = np.asarray(candidates[base_index]['feedback']['simulated_states'])[0]
            # NAVSIM's bicycle state stores body-frame longitudinal/lateral
            # velocity although positions are global. See its X_dot=v_x*cos
            # heading, Y_dot=v_x*sin heading. Do not rotate velocity again.
            initial_velocity = state[3:5].copy()
            for round_index in range(4):
                t0 = time.monotonic()
                physical = []
                if search is not None:
                    parameters, draw_record = search.draw(draws)
                    diagnostics.append(draw_record)
                else:
                    parameters = None
                if a.method == 'BEHAVIOR':
                    proposals = [smooth_behavior_request(base_request, initial_velocity, value) for value in parameters]
                    requests = np.stack([r for r, _ in proposals])
                    physical = [p for _, p in proposals]
                    actions = torch.from_numpy(encode_external_request(requests, act_norm=int(sft.datasets.vla_data.act_norm)))
                    source, keys = 'behavior_search', [{} for _ in range(draws)]
                else:
                    group_size = 16 if a.method == 'SDE' else draws
                    keys = [seed_for('bank_initial', a.seed, token, a.iteration, round_index, i) for i in range(group_size)]
                    if a.method == 'LATENT':
                        roots = torch.tensor(parameters.reshape(draws, 8, 4), device='cuda', dtype=raw.dtype)
                    else:
                        roots = torch.cat([torch.randn((1, 8, 4), device='cuda', dtype=torch.float32 if a.method == 'SDE' else raw.dtype,
                                                       generator=torch.Generator(device='cuda').manual_seed(key)) for key in keys])
                    if a.method == 'SDE':
                        innovation = seed_for('bank_sde_innovation', a.seed, token, a.iteration, round_index)
                        sampled = sample_legacy_sde(policy, condition, roots[None], innovation_generator=torch.Generator(device='cuda').manual_seed(innovation),
                                                    spec=SamplingSpec(group_size=16, num_steps=10, temporal_noise_correlation=.8))
                        actions = sampled['actions'][0, :draws].float().cpu()
                        source = 'legacy_sde'
                        status['velocity_forward_calls'] += 160
                        diagnostics.append({'round': round_index, 'innovation_key': innovation, 'generated': 16, 'scored_quota': draws,
                                            'unscored_outputs': 16-draws, 'sampler': 'original_G16_K10_rho0.8_chunk1'})
                    else:
                        actions = grouped(policy, condition, roots, chunk).float().cpu()
                        source = 'latent_search' if a.method == 'LATENT' else 'ode'
                        status['velocity_forward_calls'] += 10*((draws+chunk-1)//chunk)
                    requests = decode_request(actions.numpy(), act_norm=int(sft.datasets.vla_data.act_norm))
                    keys = [{'root_key': key, 'initial_noise': roots[i].cpu()} for i, key in enumerate(keys[:draws])]
                torch.cuda.synchronize(); status['GPU_generation_seconds'] += time.monotonic()-t0
                new_indices, duplicate_of, valid, invalid_indices = [], {}, [], {}
                for i in range(draws):
                    if physical and physical[i]['status'] != 'PHYSICAL_PROPOSAL_VALID':
                        valid.append(False)
                        invalid_indices[i] = len(attempts)
                        attempts.append({'round': round_index, 'index': i, 'status': 'INVALID_PHYSICAL_PROPOSAL',
                                         'requested_trajectory': torch.as_tensor(requests[i]), 'normalized_action': actions[i], 'physical': physical[i],
                                         'consumes_logical_budget': True, 'official_score': None})
                        status['invalid_physical_proposals'] += 1
                        continue
                    valid.append(True)
                    key = request_key(requests[i])
                    if key in cached:
                        duplicate_of[i] = cached[key]
                        status['duplicate_proposals'] += 1
                    else:
                        new_indices.append(i)
                # Within a fresh round, exact duplicate draws are unusual. They
                # still receive individual official calls, remain marked by
                # the subsequent bank deduplicator, and count in real cost.
                results = {}
                if new_indices:
                    fresh, receipt = scorer.score(token, log, requests[new_indices], caller_identity=identity['sha256'], request_key=f'{token}_round{round_index}')
                    for i, f in zip(new_indices, fresh): results[i] = f
                    diagnostics.append({'scoring': receipt})
                for i, (_, f) in duplicate_of.items(): results[i] = f
                for i in range(draws):
                    if not valid[i]:
                        if fallback_spec:
                            # A rejected proposal still consumes its one slot.
                            # Returning an already-scored fixed SFT draw adds no
                            # score call or discovery. Its duplicate frequency
                            # enters empirical u, while q_ref remains four draws.
                            ref_index = seed_for('physical_rejection_reference_fallback', a.seed, token, a.iteration, round_index, i) % 4
                            record = dict(candidates[ref_index])
                            record.update(source='reference_fallback', is_reference=False,
                                          on_policy_for_NFT=False, duplicate_of=ref_index,
                                          source_parameters={'round': round_index, 'failure_status': 'INVALID_PHYSICAL_PROPOSAL',
                                                             'original_reference_index': ref_index,
                                                             'rejected_attempt_index': invalid_indices[i]})
                            attempts[invalid_indices[i]]['fallback_candidate_index'] = len(candidates)
                            candidates.append(record)
                            status['duplicate_proposals'] += 1
                        continue
                    f = results[i]
                    record = dict(source=source, is_reference=False, behavior_sha256=behavior,
                                  normalized_action=actions[i], requested_trajectory=torch.as_tensor(requests[i]),
                                  feedback=f, noise_keys=keys[i], on_policy_for_NFT=(source == 'ode'),
                                  source_parameters={'round': round_index, 'parameters': parameters[i].tolist() if parameters is not None else None,
                                                     'physical': physical[i] if physical else None,
                                                     **({'center_candidate_index': base_index, 'center_behavior_sha256': behavior} if mixed_online else {})},
                                  duplicate_of=duplicate_of[i][0] if i in duplicate_of else None)
                    candidates.append(record)
                    cached[request_key(requests[i])] = (len(candidates)-1, f)
                if search is not None:
                    search.update(parameters, [results[i]['metrics'] if valid[i] else None for i in range(draws)])
            row = {'token': token, 'log_name': log, 'bank_identity': identity['sha256'], 'candidates': candidates,
                   'invalid_attempts': attempts, 'search_diagnostics': diagnostics,
                   'budget': {'logical_candidate_budget': budget, 'reference_historical_scoring_calls': 4,
                              'reference_new_calls_this_run': reference_new_calls, 'reference_reused_calls': reference_reused_calls,
                              'finite_scored_records': len(candidates), 'invalid_physical_attempts': len(attempts)},
                   'candidate_scores_are_not_deployment_policy_scores': True}
            if fallback_spec:
                assert len(candidates) == budget and all('fallback_candidate_index' in x for x in attempts)
                row['budget']['fallback_reused_reference_records'] = len(attempts)
                row['budget']['invalid_attempts_are_the_same_slots_as_fallbacks'] = True
            else:
                assert len(candidates)+len(attempts) == budget
            publish_scene(dst, row, identity['sha256'])
            status['new_scenes'] += 1; status['logical_proposals'] += budget
            status.update(actual_new_scoring_calls=scorer.actual_new_scores, reused_score_candidates=scorer.reused_scores,
                          wall_seconds=time.monotonic()-started)
            atomic_json(out/f'rank{rank}.json', status)
            print(json.dumps(status), flush=True)
            if a.stop_after_for_qualification and status['new_scenes'] >= a.stop_after_for_qualification:
                status['status'] = 'QUALIFICATION_WINDOW_COMPLETE_FULL_POPULATION_PENDING'
                break
        else:
            status['status'] = 'ASSIGNED_POPULATION_COMPLETE'
    status['wall_seconds'] = time.monotonic()-started
    atomic_json(out/f'rank{rank}.json', status)


if __name__ == '__main__':
    main()
