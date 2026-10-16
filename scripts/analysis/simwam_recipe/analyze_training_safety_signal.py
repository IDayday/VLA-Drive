"""Finite TRAIN-only audit of completed journals and unsafe-positive chains.

No model, scorer, optimizer or deployment is executed. Counts of positive
advantages are not gradient contributions or a causal safety attribution.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from starVLA.rl.simwam_recipe.contracts import file_sha, require_dispatch
from starVLA.rl.simwam_recipe.data_stream import SceneStream
from starVLA.rl.simwam_recipe.invocation import validate_rows
from starVLA.rl.simwam_recipe.recovery_state import rollout_digest
from starVLA.rl.flow_grpo.transactions import atomic_json
from scripts.analysis.simwam_recipe.summarize_training import read_journal

SAFETY = {'NC': 'no_at_fault_collisions', 'DAC': 'drivable_area_compliance',
          'TTC': 'time_to_collision_within_bound'}
MEANS = ('canonical_random_mean', 'train_reward_mean', 'NC', 'DAC', 'TTC', 'EP', 'safe_fraction')
GROUP_FLAGS = ('all_full', 'all_zero', 'tied', 'zero_PG', 'progress_only', 'safety_differing')
POSITIVE_COUNTS = ('positive_safe', 'positive_still_unsafe', 'positive_unsafe_with_safe_alternative')


def safety_type(metrics):
    values = [float(metrics[k]) for k in SAFETY.values()]
    if not np.isfinite(values).all() or any(x < 0 or x > 1 for x in values):
        raise ValueError('Invalid safety submetric')
    return '+'.join(name for name, value in zip(SAFETY, values) if value < 1) or 'SFT_single_seed_safe'


def candidate_signal(records, advantage, mask):
    """Classify actual saved positives without discarding their negative peers."""
    if len(records) != 8:
        raise ValueError('The registered group contains exactly eight candidates')
    if any(r['status'] != 'OK' for r in records):
        raise ValueError('Infrastructure errors cannot enter a zero-score class')
    metrics = [r['metrics'] for r in records]
    types = [safety_type(m) for m in metrics]
    safe = np.array([t == 'SFT_single_seed_safe' for t in types])
    scores = np.array([float(m['score']) for m in metrics])
    a = np.asarray(advantage, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=np.float64).reshape(-1)
    if a.shape != (8,) or mask.shape != (8,) or not np.isfinite([*a, *scores, *mask]).all():
        raise ValueError('Nonfinite or incorrectly grouped saved material')
    if not np.isin(mask, [0, 1]).all() or len(set(mask)) != 1:
        raise ValueError('Source PG mask must cover the complete scene group')
    raw_positive = a > 0
    positive = raw_positive & (mask > 0)
    bad = positive & ~safe
    best_safe = float(scores[safe].max()) if safe.any() else None
    gaps = (scores[bad] - best_safe).tolist() if best_safe is not None else []
    # 1e-8 is a declared reporting comparison tolerance, not a reward change.
    tol = 1e-8
    return dict(
        raw_positive_unsafe=int((raw_positive & ~safe).sum()),
        masked_positive_unsafe=int((raw_positive & ~safe & (mask == 0)).sum()),
        active_positive_unsafe=int(bad.sum()),
        active_positive_safe=int((positive & safe).sum()),
        unsafe_positive_types=dict(Counter(t for t, b in zip(types, bad) if b)),
        active_positive_unsafe_with_safe_alternative=int(bad.sum()) if safe.any() else 0,
        active_positive_unsafe_without_safe_alternative=int(bad.sum()) if not safe.any() else 0,
        unsafe_positive_score_above_best_safe=sum(g > tol for g in gaps),
        unsafe_positive_score_equal_best_safe=sum(abs(g) <= tol for g in gaps),
        unsafe_positive_score_below_best_safe=sum(g < -tol for g in gaps),
        unsafe_positive_minus_best_safe=gaps,
        unsafe_positive_advantage_sum=float(a[bad].sum()),
        unsafe_positive_clipped_advantage_sum=float(np.clip(a[bad], -5, 5).sum()),
        selected_group_positive_advantage_sum=float(a[positive].sum()),
        selected_group_positive_clipped_advantage_sum=float(np.clip(a[positive], -5, 5).sum()),
        selected_group_safe_positive_advantage_sum=float(a[positive & safe].sum()),
        selected_group_unsafe_positive_with_nonunit_comfort=int(sum(
            b and m['comfort'] < 1 for b, m in zip(bad, metrics))),
        selected_group_safe_alternatives_all_comfort_one=bool(
            safe.any() and all(m['comfort'] == 1 for m, s in zip(metrics, safe) if s)),
        all_candidates_unsafe=not bool(safe.any()),
        score_comparison_tolerance=tol)


def aggregate(rows, population_count, logs):
    n = len(rows)
    ex = [r['exploration'] for r in rows]
    counts = {k: sum(int(e[k]) for e in ex) for k in GROUP_FLAGS}
    raw = {k: sum(int(e[k]) for e in ex) for k in POSITIVE_COUNTS}
    active = {k: sum(int(e[k]) for e in ex if not e['zero_PG']) for k in POSITIVE_COUNTS}
    unsafe = [int(round((1 - e['safe_fraction']) * 8)) for e in ex]
    if any(abs((1 - e['safe_fraction']) * 8 - c) > 1e-10 for e, c in zip(ex, unsafe)):
        raise ValueError('Nonintegral unsafe candidate count')
    positives = active['positive_safe'] + active['positive_still_unsafe']
    return dict(
        hard_population_scenes=population_count, scene_exposures=n,
        unique_scenes=len({r['token'] for r in rows}),
        unique_logs=len({logs[r['token']] for r in rows}), candidates=n * 8,
        groups=counts, raw_positive_candidate_counts=raw, active_PG_positive_candidate_counts=active,
        active_positive_unsafe_fraction=(active['positive_still_unsafe'] / positives if positives else None),
        unsafe_candidates=sum(unsafe), groups_with_unsafe_candidates=sum(x > 0 for x in unsafe),
        groups_with_only_unsafe_candidates=sum(x == 8 for x in unsafe),
        groups_with_safe_and_unsafe_candidates=sum(0 < x < 8 for x in unsafe),
        stochastic_mean={k: float(np.mean([e[k] for e in ex])) if n else None for k in MEANS})


def validate_saved(rollout, row, rank, identity, population, stream, *, reward_summary_device=None):
    from starVLA.rl.simwam_recipe.collection import exploration_summary
    from starVLA.rl.simwam_recipe.reward_readouts import readouts
    if rollout['token'] != row['token'] or rollout['behavior_version'] != row['behavior']:
        raise ValueError('Saved behavior is not the journal collection')
    if rollout['log'] != population['token_logs'][row['token']]:
        raise ValueError('Saved log differs from fixed TRAIN manifest')
    if rollout['protocol_sha256'] != identity['score_protocol']['sha256']:
        raise ValueError('Scoring identity mismatch')
    expected_token, position = stream.at((row['collection'] - 1) * 8, rank)
    if expected_token != row['token'] or position != rollout['stream_position']:
        raise ValueError('Saved stream position mismatch')
    if tuple(rollout['chain'].shape) != (8, 11, 8, 4) or rollout['stoch_idx'].tolist() != [7, 8, 9]:
        raise ValueError('This audit is registered only for the completed original-window controls')
    digest = rollout_digest(rollout)
    if 'chain_identity' in row:
        if row['chain_identity'] != digest or rollout.get('chain_identity') != digest:
            raise ValueError('Changed committed chain, probability or advantage')
    elif 'chain_identity' in rollout:
        raise ValueError('Unexpected partially attributed legacy material')
    for record in rollout['records']:
        if record['token'] != row['token'] or record['log'] != rollout['log']:
            raise ValueError('Mixed scene in candidate group')
        if record['protocol_sha256'] != rollout['protocol_sha256'] or readouts(record) != record['readouts']:
            raise ValueError('Changed canonical readout')
    stored_rewards = rollout['rewards'].float().reshape(-1)
    expected_rewards = torch.tensor([r['readouts']['canonical_PDMS'] for r in rollout['records']], dtype=torch.float32)
    if not torch.equal(stored_rewards, expected_rewards):
        raise ValueError('Saved reward differs from actual canonical score')
    values = stored_rewards.double(); offsets = values - values[0]
    centered = offsets - offsets.mean(); std = float(rollout['global_std'])
    expected_a = (centered / (std + 1e-8)).float()
    expected_mask = torch.full((8,), float(offsets.std(unbiased=False) > 1e-8))
    if not torch.equal(expected_a, rollout['advantages']) or not torch.equal(expected_mask, rollout['adv_mask']):
        raise ValueError('Saved advantage or whole-group mask is inconsistent')
    actual = exploration_summary(rollout)
    if reward_summary_device is not None:
        # The journal used a CUDA FP32 reduction. CPU and CUDA means of eight
        # identical FP32 rewards can differ by two ULPs. GPU diagnostics can
        # reproduce the original operation exactly, without relaxing tolerances
        # or changing any reward/advantage/chain field. Legacy CPU calls retain
        # their original behavior and may conservatively reject such a summary.
        if torch.device(reward_summary_device).type != 'cuda':
            raise ValueError('Explicit journal reduction replay requires CUDA')
        actual['train_reward_mean'] = float(rollout['rewards'].to(reward_summary_device).mean())
        if actual['train_reward_mean'] != row['exploration']['train_reward_mean']:
            raise ValueError('CUDA reward summary does not exactly reproduce journal')
    for key, value in actual.items():
        old = row['exploration'][key]
        if value is None or isinstance(value, bool):
            if value != old: raise ValueError('Changed group summary: ' + key)
        elif not np.isclose(value, old, atol=1e-7, rtol=0):
            raise ValueError('Changed group summary: ' + key)
    return digest


def analyze(registration_path, run_path, output):
    begin = time.monotonic()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('This read-only audit must run with CUDA_VISIBLE_DEVICES empty')
    host = require_dispatch()
    registration_path, run, output = map(Path, (registration_path, run_path, output))
    if output.exists(): raise FileExistsError('Preserve old analysis output')
    reg = json.loads(registration_path.read_text())
    locked = reg['runs'][str(run)]
    identity_path = run / 'identity.json'
    identity = json.loads(identity_path.read_text())
    if identity['sha256'] != locked['run_identity'] or file_sha(identity_path) != locked['identity_file_sha256']:
        raise ValueError('Run identity changed after registration')
    if identity['recipe']['variant'] not in ('R_CANONICAL', 'R_CANONICAL_X1') or identity['world'] != 8:
        raise ValueError('This finite audit does not admit additional recipes')
    for field in ('hard_manifest', 'population'):
        if file_sha(reg[field]['path']) != reg[field]['sha256']:
            raise ValueError('Registered data identity changed: ' + field)
    manifest = json.loads(Path(reg['hard_manifest']['path']).read_text())
    population = json.loads(Path(reg['population']['path']).read_text())
    if identity['hard_manifest_sha256'] != reg['hard_manifest']['sha256']:
        raise ValueError('Training used a different hard population')
    hard = {r['token']: r for r in manifest['rows']}
    if set(hard) != set(manifest['tokens']) or len(hard) != len(manifest['rows']):
        raise ValueError('Incomplete or duplicated baseline classification')
    if not set(hard) <= set(population['train_tokens']): raise ValueError('Non-TRAIN scene')
    held = set(population['dev_tokens']) | set(population['navtest_tokens'])
    if set(hard) & held: raise ValueError('Held-out data entered audit')
    train_logs = {population['token_logs'][t] for t in hard}
    if train_logs & {population['token_logs'][t] for t in held}:
        raise ValueError('Held-out log overlap')
    strata = {t: safety_type(r['metrics']) for t, r in hard.items()}
    population_types = Counter(strata.values())
    stream = SceneStream(reg['hard_manifest']['path'], 8)
    rows_by_rank, journal_hashes = [], {}
    for rank in range(8):
        path = run / f'metrics_rank{rank}.jsonl'
        digest = file_sha(path)
        if digest != locked['journal_sha256'][str(rank)]: raise ValueError('Journal changed')
        rows = read_journal(path); validate_rows(rows, reg['optimizer_updates'])
        result_path = run / f'result_rank{rank}.json'
        result = json.loads(result_path.read_text())
        if result['status'] != 'COMPLETE' or result['updates'] != reg['optimizer_updates']:
            raise ValueError('Only completed fixed-budget experiments are admitted')
        for i in range(0, len(rows), 4):
            group = rows[i:i+4]; first = group[0]
            token, _ = stream.at((first['collection'] - 1) * 8, rank)
            if token != first['token'] or first['behavior']['run'] != identity['sha256']:
                raise ValueError('Changed data order or behavior identity')
            for r in group:
                if any(r.get(k) != first.get(k) for k in ('token', 'behavior', 'chain_identity', 'exploration')):
                    raise ValueError('A fresh group changed across its four inner updates')
                if r['raw_scene_exposures'] != r['collection'] * 8 or r['new_candidates'] != r['collection'] * 64:
                    raise ValueError('Inner reuse counted as fresh samples')
        rows_by_rank.append(rows[::4]); journal_hashes[str(rank)] = digest
    for group in zip(*rows_by_rank):
        if len({r['exploration']['global_reward_std'] for r in group}) != 1:
            raise ValueError('Ranks did not share the same recorded raw-reward std')
        if len({r['behavior']['adapter_sha256'] for r in group}) != 1:
            raise ValueError('Ranks used different behavior versions')
    all_rows = [r for rows in rows_by_rank for r in rows]
    by_stratum = {key: aggregate([r for r in all_rows if strata[r['token']] == key], count, population['token_logs'])
                  for key, count in sorted(population_types.items())}
    selected = [(rank, r) for rank, rows in enumerate(rows_by_rank) for r in rows
                if r['exploration']['positive_still_unsafe'] > 0]
    if len(selected) > reg['maximum_selected_groups_per_run']:
        raise ValueError('Finite registered saved-chain read budget exceeded; do not truncate')
    output.mkdir(parents=True)
    sums = Counter(); type_counts = Counter(); scene_counts = Counter(); gaps = []
    read_bytes = 0; digested = 0; legacy = 0
    # This file contains private scene identities and must not be copied into git.
    with (output / 'selected_groups_private.jsonl').open('x') as evidence:
        for rank, row in selected:
            path = run / 'behavior' / f"collection{row['collection']:06d}" / f'rank{rank}.pt'
            rollout = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            digest = validate_saved(rollout, row, rank, identity, population, stream)
            signal = candidate_signal(rollout['records'], rollout['advantages'].numpy(), rollout['adv_mask'].numpy())
            if signal['raw_positive_unsafe'] != row['exploration']['positive_still_unsafe']:
                raise ValueError('Selected saved candidates do not match journal counts')
            digested += int('chain_identity' in row); legacy += int('chain_identity' not in row)
            read_bytes += path.stat().st_size
            for k, v in signal.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and k != 'score_comparison_tolerance': sums[k] += v
            type_counts.update(signal['unsafe_positive_types']); gaps.extend(signal['unsafe_positive_minus_best_safe'])
            scene_counts[row['token']] += signal['active_positive_unsafe']
            evidence.write(json.dumps(dict(rank=rank,collection=row['collection'],token=row['token'],
                log=rollout['log'],SFT_stratum=strata[row['token']],behavior_file_sha256=file_sha(path),
                digest=digest,preexisting_digest_available='chain_identity' in row,signal=signal),allow_nan=False)+'\n')
    overall = aggregate(all_rows, len(hard), population['token_logs'])
    if sums['active_positive_unsafe'] != overall['active_PG_positive_candidate_counts']['positive_still_unsafe']:
        raise ValueError('Selected-chain totals do not reconcile with complete journals')
    concentration = sorted(scene_counts.values(), reverse=True)
    summary = dict(schema='simwam_TRAIN_safety_signal_audit_v1',status='COMPLETE_READ_ONLY',
        captured_utc=datetime.now(timezone.utc).isoformat(),host=host,
        registration_sha256=file_sha(registration_path),analysis_source_sha256=file_sha(__file__),
        run_identity=identity['sha256'],variant=identity['recipe']['variant'],
        hard_manifest_sha256=reg['hard_manifest']['sha256'],population_sha256=reg['population']['sha256'],
        hard_scene_count=len(hard),hard_log_count=len(train_logs),full_TRAIN_count=len(population['train_tokens']),
        journal_sha256=journal_hashes,optimizer_updates=reg['optimizer_updates'],
        fresh_global_rollouts=reg['optimizer_updates']//4,overall=overall,by_SFT_single_seed_safety=by_stratum,
        matched_scene_order_digest=__import__('hashlib').sha256(json.dumps(
            [[r['token'] for r in rs] for rs in rows_by_rank],separators=(',',':')).encode()).hexdigest(),
        selected_unsafe_positive_groups=dict(groups=len(selected),candidate_records=len(selected)*8,
            preexisting_chain_digest_verified=digested,legacy_without_preexisting_chain_digest=legacy,
            counters=dict(sums),unsafe_positive_types=dict(type_counts),
            unique_positive_unsafe_scenes=sum(x>0 for x in concentration),
            active_unsafe_positive_counts_in_top1_top10_top50_scenes=[sum(concentration[:k]) for k in (1,10,50)],
            score_gap_to_best_safe_quantiles={str(q):float(np.quantile(gaps,q)) for q in (0,.05,.5,.95,1)} if gaps else {},
            private_evidence_sha256=file_sha(output/'selected_groups_private.jsonl')),
        cost=dict(worker_wall_seconds=time.monotonic()-begin,saved_file_bytes_read_and_hashed=read_bytes,
            new_scores=0,new_rollout_candidates=0,optimizer_updates=0,GPU_used=False),
        limitations=[
            'SFT safety is defined by one fixed seed; not safety for every root noise.',
            'Counts or advantage sums are not gradient norms, causal attribution, or deployment metrics.',
            'Selected-group advantage sums exclude groups with no unsafe positives; no global advantage-mass fraction is claimed.',
            'Global raw std is checked for saved-value and rank consistency; unselected chains are not reread to independently reconstruct it.',
            'Legacy canonical journals lack committed chain digests; file hashes, behavior/order, recomputed readouts/advantages and journal statistics are checked, without inventing historical hashes.',
            'No DEV/Navtest samples, parameter fitting, new reward, curriculum, or policy update are used.'])
    atomic_json(output/'summary.json',summary)
    print(json.dumps(dict(status=summary['status'],variant=summary['variant'],groups=len(all_rows),
        selected_groups=len(selected),active_unsafe_positive=sums['active_positive_unsafe'],
        above_best_safe=sums['unsafe_positive_score_above_best_safe'],seconds=summary['cost']['worker_wall_seconds'])))


def qualify():
    def records(scores, types=None):
        types=types or ['']*8
        return [dict(status='OK',metrics=dict(score=s,comfort=1,**{
            k:0 if name in typ.split('+') else 1 for name,k in SAFETY.items()})) for s,typ in zip(scores,types)]
    checks=[]
    assert safety_type(dict(zip(SAFETY.values(), [.5,1,1])))=='NC'; checks.append('NC_half_is_unsafe')
    r=records([.2,.8,0,0,0,0,0,0],['TTC']+['']*7)
    a=np.array([1,2,-.5,-.5,-.5,-.5,-.5,-.5])
    s=candidate_signal(r,a,np.ones(8)); assert s['active_positive_unsafe']==1 and s['unsafe_positive_score_below_best_safe']==1; checks.append('above_group_mean_is_not_above_best_safe')
    s=candidate_signal(r,a,np.zeros(8)); assert s['active_positive_unsafe']==0 and s['masked_positive_unsafe']==1; checks.append('masked_sign_is_not_active_PG')
    s=candidate_signal(records([0]*8,['DAC']*8),np.zeros(8),np.zeros(8)); assert s['raw_positive_unsafe']==0 and s['all_candidates_unsafe']; checks.append('all_zero_has_no_fake_positive')
    s=candidate_signal(records([.1,.2,.3,.4,.5,.6,.7,.8],['TTC']*8),np.arange(8)-3.5,np.ones(8)); assert s['active_positive_unsafe_without_safe_alternative']==4 and not s['unsafe_positive_minus_best_safe']; checks.append('all_unsafe_relative_ranking_is_preserved')
    s=candidate_signal(records([.8,.2,0,0,0,0,0,0],['TTC']+['']*7),a,np.ones(8)); assert s['unsafe_positive_score_above_best_safe']==1; checks.append('genuine_score_rank_inversion_is_recorded')
    s=candidate_signal(r,np.array([7,2,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5]),np.ones(8)); assert s['unsafe_positive_advantage_sum']==7 and s['unsafe_positive_clipped_advantage_sum']==5; checks.append('raw_and_source_clipped_advantage_are_separate')
    for name, mutate in [('infrastructure_error',lambda r:r[0].update(status='ERROR')),
                         ('invalid_metric',lambda r:r[0]['metrics'].update(no_at_fault_collisions=float('nan')))]:
        bad=records([.5]*8); mutate(bad)
        try: candidate_signal(bad,np.zeros(8),np.ones(8))
        except ValueError: checks.append(name+'_rejected')
        else: raise AssertionError(name)
    try: candidate_signal(r,a,np.array([0,1,1,1,1,1,1,1]))
    except ValueError: checks.append('partial_candidate_mask_rejected')
    else: raise AssertionError('partial mask')
    print(json.dumps(dict(status='PASS',checks=checks,new_model_execution=0,new_scores=0)))


def main():
    p=argparse.ArgumentParser(__doc__); sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('qualify')
    a=sub.add_parser('analyze');a.add_argument('--registration',required=True);a.add_argument('--run',required=True);a.add_argument('--output',required=True)
    args=p.parse_args()
    if args.command=='qualify':qualify()
    else:analyze(args.registration,args.run,args.output)


if __name__=='__main__':main()
