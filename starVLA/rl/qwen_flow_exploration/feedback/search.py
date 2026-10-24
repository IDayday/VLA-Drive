"""K32 discovery with eight common roots and three rounds of eight proposals.

Callbacks own the mature native model and independent canonical scorer. This
module never receives GT trajectories, changes actor inputs, or creates policy
probabilities for external requests. F2 accesses only scalar rewards.
"""
import hashlib
import time

import numpy as np

from .process import process_summary
from .proposal import StructuredCEM, structured_proposal


METHODS = ('F0_IID', 'F1_BLIND', 'F2_PDMS', 'F3_FEEDBACK', 'F4_NOVELTY')


def reward(candidate):
    if 'raw_reward' in candidate:
        value = float(candidate['raw_reward'])
    else:
        value = float(candidate.get('feedback', candidate)['metrics']['score'])
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Scoring failure cannot become an official zero')
    return value


def tie_key(candidate_id, seed):
    return hashlib.sha256(f'{seed}:{candidate_id}'.encode()).hexdigest()


def scalar_rank(candidate, seed):
    # Deliberately no metrics, feasibility, trace or physical process access.
    return (-reward(candidate), tie_key(candidate['candidate_id'], seed))


def physical_rank(candidate, seed):
    summary = process_summary(candidate['feedback'])
    if summary['feasible']:
        return (0, -reward(candidate), tie_key(candidate['candidate_id'], seed))
    return (1, summary['search_violation_cost'], tie_key(candidate['candidate_id'], seed))


def trajectory_key(planned):
    return hashlib.sha256(np.ascontiguousarray(planned, dtype=np.float32).tobytes()).hexdigest()


def decorate(record, candidate_id, *, source, parent=None, parameters=None, diagnostic=None):
    if record.get('status', 'OK') != 'OK':
        raise ValueError('Invalid scorer status is not an official zero sample')
    planned = np.asarray(record.get('planned_trajectory', record.get('requested_trajectory')), dtype=np.float32)
    if planned.shape != (8, 3) or not np.isfinite(planned).all():
        raise ValueError('Native planned trajectory must be preserved as8x3')
    result = dict(record)
    result.update(candidate_id=candidate_id, proposal_source=source, parent_candidate_id=parent,
                  planned_trajectory=planned, raw_reward=reward(record),
                  parameters=None if parameters is None else np.asarray(parameters).tolist(),
                  proposal_validation=diagnostic, external_proposal=parameters is not None,
                  on_policy_logprob=None,
                  # ODE roots have a prior density but no ordinary Flow-GRPO
                  # stochastic-chain likelihood. Never forge an old chain.
                  can_enter_GRPO_ratio=False)
    if 'feedback' not in result:
        result['feedback'] = record
    return result


def qualified_novelty_archive(candidates, scale, *, max_entries=16):
    observed = []
    for candidate in candidates:
        if candidate.get('proposal_status') == 'INVALID_PHYSICAL_PROPOSAL':
            continue
        summary = process_summary(candidate['feedback'])
        if summary['feasible'] and summary['novelty_features'] is not None:
            observed.append((candidate, summary))
    if not observed:
        return []
    best = max(reward(candidate) for candidate, _ in observed)
    qualified = [(candidate, summary) for candidate, summary in observed
                 if reward(candidate) >= .9 or reward(candidate) >= best-.005]
    features = [scale.encode(summary['novelty_features']) for _, summary in qualified]
    rows = []
    seen = set()
    for index, ((candidate, summary), feature) in enumerate(zip(qualified, features)):
        key = trajectory_key(candidate['planned_trajectory'])
        if key in seen:
            continue
        seen.add(key)
        distances = [float(np.linalg.norm(feature-other)) for j, other in enumerate(features) if j != index]
        novelty = min(distances) if distances else 0.
        rows.append(dict(candidate=candidate, novelty=novelty, features=feature.tolist()))
    rows.sort(key=lambda row: (-row['novelty'], -reward(row['candidate']), row['candidate']['candidate_id']))
    return rows[:max_entries]


def run_search(method, base_candidates, *, initial_velocity_xy, sample_iid, score_requests,
               seed, novelty_scale=None):
    """Return every attempt; invalid proposals consume K and have reward=None.

    base_candidates are the SAME eight scored ODE candidates for all methods.
    sample_iid(round_index, count) -> list of complete public candidate records.
    score_requests(requests) -> list of records including requested/simulated
    trajectories and feedback. Root code owns CPU/GPU and exact cost ledgers.
    """
    if method not in METHODS or len(base_candidates) != 8:
        raise ValueError('Locked discovery methods require eight shared references')
    if method == 'F4_NOVELTY' and novelty_scale is None:
        raise ValueError('F4 needs frozen TRAIN-only feature scaling')
    started = time.monotonic()
    rng = np.random.default_rng(seed)
    cem = StructuredCEM(rng)
    candidates = [decorate(row, f'base{i:02d}', source='shared_ODE') for i, row in enumerate(base_candidates)]
    # Every structured arm uses the first shared trajectory, rather than
    # choosing a different scalar/safety-conditioned anchor for each method.
    anchor = candidates[0]
    cache = {trajectory_key(row['planned_trajectory']): row for row in candidates}
    if method in ('F2_PDMS', 'F3_FEEDBACK', 'F4_NOVELTY'):
        rank = scalar_rank if method == 'F2_PDMS' else physical_rank
        cem.elites = [(np.array([0., 1., 0., 0.]), rank(anchor, seed))]
    invalid_attempts, actual_scores, batch_calls, duplicate_attempts = 0, 0, 0, 0
    rounds = []
    for round_index in range(3):
        if method == 'F0_IID':
            rows = sample_iid(round_index, 8)
            if len(rows) != 8:
                raise ValueError('No hidden or infinite IID resampling')
            selected = [decorate(row, f'r{round_index}_p{i}', source=method) for i, row in enumerate(rows)]
            candidates.extend(selected)
            # Model/scorer callbacks account their actual batch calls. This
            # count is explicit candidate evaluation, not DiT batch invocations.
            actual_scores += 8
            batch_calls += 1
            rounds.append(dict(round=round_index, attempted=8, scored=8, invalid=0, novelty_proposals=0))
            continue
        parameters, draw = cem.draw(8)
        novelty_proposals = 0
        if method == 'F4_NOVELTY':
            archive = qualified_novelty_archive(candidates, novelty_scale)
            parents = [row for row in archive if row['candidate']['parameters'] is not None]
            if parents:
                # Same eight Gaussian innovations; replace only two centers.
                innovation = (parameters-cem.mean)/np.maximum(cem.std, cem.floor)
                for index in range(6, 8):
                    parent = parents[(index-6) % len(parents)]['candidate']
                    parameters[index] = np.clip(np.asarray(parent['parameters'])+innovation[index]*cem.std,
                                                cem.lower, cem.upper)
                novelty_proposals = 2
        attempted = []
        new_requests, pending_indices, pending_keys = [], [], []
        before_invalid = invalid_attempts
        before_duplicate = duplicate_attempts
        for index, pars in enumerate(parameters):
            candidate_id = f'r{round_index}_p{index}'
            requested, diagnostic = structured_proposal(anchor['planned_trajectory'], initial_velocity_xy, pars)
            row = dict(candidate_id=candidate_id, proposal_source=method,
                       parent_candidate_id=anchor['candidate_id'], planned_trajectory=requested,
                       parameters=pars.tolist(), proposal_validation=diagnostic,
                       external_proposal=True, on_policy_logprob=None, can_enter_GRPO_ratio=False)
            if diagnostic['status'] != 'PHYSICAL_PROPOSAL_VALID':
                row.update(proposal_status='INVALID_PHYSICAL_PROPOSAL', raw_reward=None, feedback=None,
                           invalid_reason='registered_dense_physical_proposal_limits', scorer_calls=0)
                invalid_attempts += 1
            else:
                key = trajectory_key(requested)
                if key in cache:
                    old = cache[key]
                    row.update(raw_reward=reward(old), feedback=old['feedback'],
                               duplicate_of=old['candidate_id'], scorer_calls=0, proposal_status='SCORED_OR_REUSED')
                    duplicate_attempts += 1
                elif key in pending_keys:
                    row.update(duplicate_of_pending=pending_indices[pending_keys.index(key)])
                    duplicate_attempts += 1
                else:
                    pending_indices.append(index)
                    pending_keys.append(key)
                    new_requests.append(requested)
            attempted.append(row)
        if new_requests:
            records = score_requests(np.stack(new_requests))
            if len(records) != len(new_requests):
                raise ValueError('Scoring failures cannot disappear from the denominator')
            batch_calls += 1
            actual_scores += len(records)
            for index, key, record in zip(pending_indices, pending_keys, records):
                row = attempted[index]
                enriched = decorate(record, row['candidate_id'], source=method,
                                    parent=anchor['candidate_id'], parameters=row['parameters'],
                                    diagnostic=row['proposal_validation'])
                error = float(np.max(np.abs(enriched['planned_trajectory']-row['planned_trajectory'])))
                if error > 1e-5:
                    raise ValueError('Native codec/scorer changed the original request beyond registered1e-5')
                enriched['proposal_preencoding_request'] = row['planned_trajectory']
                enriched['native_codec_roundtrip_max_abs'] = error
                row.update(enriched, scorer_calls=1, proposal_status='SCORED_OR_REUSED')
                cache[key] = row
        for row in attempted:
            if 'duplicate_of_pending' in row:
                previous = attempted[row.pop('duplicate_of_pending')]
                row.update(raw_reward=reward(previous), feedback=previous['feedback'],
                           duplicate_of=previous['candidate_id'], scorer_calls=0, proposal_status='SCORED_OR_REUSED')
        usable = [row for row in attempted if row['raw_reward'] is not None]
        if method in ('F2_PDMS', 'F3_FEEDBACK', 'F4_NOVELTY'):
            rank = scalar_rank if method == 'F2_PDMS' else physical_rank
            cem.update_ranked([(row['parameters'], rank(row, seed)) for row in usable])
        else:
            # Blind proposals keep the original distribution; round limits
            # still consume exactly the same three-round schedule.
            cem.round += 1
        candidates.extend(attempted)
        rounds.append(dict(round=round_index, attempted=8, scored=len(new_requests),
                           invalid=invalid_attempts-before_invalid, duplicates=duplicate_attempts-before_duplicate,
                           novelty_proposals=novelty_proposals, proposal_distribution=draw))
    if len(candidates) != 32:
        raise AssertionError('K32 includes all failed/invalid attempts')
    return dict(method=method, candidates=candidates, rounds=rounds, proposal_identity=cem.identity(),
                costs=dict(candidate_attempts=32, common_reference_logical_scores=8,
                           candidate_scores_requested_from_callbacks=actual_scores,
                           scorer_batch_callback_calls=batch_calls, invalid_attempts=invalid_attempts,
                           duplicate_attempts=duplicate_attempts,
                           model_forward_calls=None,
                           model_forward_counter_source='shared native model runner ledger required',
                           wall_seconds=time.monotonic()-started),
                oracle_is_finite_budget_discovery_not_deployed_policy=True)


def select_teacher_candidates(candidates, *, reference_reward, split, maximum=4, delta_useful=.005,
                              minimum_quality=.9, minimum_pairwise_ADE_m=.2):
    if split != 'TRAIN' or maximum not in (2, 3, 4):
        raise ValueError('Only qualified TRAIN candidates may enter supervised absorption')
    selected = []
    eligible = []
    for row in candidates:
        if row.get('raw_reward') is None:
            continue
        summary = process_summary(row['feedback'])
        if summary['feasible'] and reward(row) >= max(minimum_quality, reference_reward+delta_useful):
            eligible.append(row)
    eligible.sort(key=lambda row: (-reward(row), row['candidate_id']))
    for row in eligible:
        if selected and any(float(np.linalg.norm(np.asarray(row['planned_trajectory'])[:, :2]-np.asarray(old['planned_trajectory'])[:, :2], axis=1).mean()) < minimum_pairwise_ADE_m for old in selected):
            continue
        selected.append(row)
        if len(selected) == maximum:
            break
    return selected


def discovery_budget_curve(result, *, reference_reward, useful_delta=.005):
    """Posthoc audit may inspect true feedback, including for scalar-only F2.

    This function is never called by the F2 optimizer or proposal update.
    Missing scores for physical invalidity stay missing, not synthetic zero.
    """
    rows = result['candidates']
    if len(rows) != 32:
        raise ValueError('Exactly K32 attempts including invalid proposals are required')
    curve = []
    for budget in (8, 16, 24, 32):
        prefix = rows[:budget]
        scored = [row for row in prefix if row.get('raw_reward') is not None]
        feasible = [row for row in scored if process_summary(row['feedback'])['feasible']]
        curve.append(dict(K=budget, attempted=budget, scored=len(scored),
                          valid_official_score_fraction=len(scored)/budget,
                          feasible=bool(feasible), feasible_fraction=len(feasible)/budget,
                          oracle_PDMS=max((reward(row) for row in scored), default=None),
                          feasible_oracle_PDMS=max((reward(row) for row in feasible), default=None),
                          found_useful_safe_improvement=any(reward(row) >= reference_reward+useful_delta for row in feasible),
                          unique_scored_requests=len({trajectory_key(row['planned_trajectory']) for row in scored})))
    return curve
