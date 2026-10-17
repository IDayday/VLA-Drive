"""Full-denominator, paired PDMS release analysis of saved single trajectories."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.analysis.fm_transfer.rescore_saved_navtest import authenticated, validate_input_manifest
from scripts.analysis.simwam_recipe.analyze_evaluation import METRICS, SAFETY
from starVLA.rl.safe_flow_improvement.diagnostics import log_bootstrap_difference
from starVLA.rl.simwam_recipe.contracts import POPULATION, file_sha, fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json


def validate_origin_ranks(source, ranks):
    """The sealed pre-ledger historical worker did not write a rank identity.

    Only that exact evaluator hash may use its recorded shard counts instead.
    Every original scene's manifest/payload hash is still checked below.
    """
    legacy = 'f9bfa2d205bd9548beca787730c8cf7508f2412998e72f7665848e062b218163'
    modes = []
    if len(ranks) != source['shards']: raise ValueError('Missing original rank')
    for rank, record in enumerate(ranks):
        expected = len(range(rank, source['scenes'], source['shards']))
        if record['status'] != 'ASSIGNED_COMPLETE' or record['new'] + record['reused'] != expected:
            raise ValueError('Original rank did not complete its assigned population')
        if 'identity_sha256' in record:
            if record['identity_sha256'] != source['sha256']: raise ValueError('Original rank identity changed')
            modes.append('rank_identity_and_per_scene_seals')
        elif source['source_sha256'] == legacy and record['direct_original_check']:
            modes.append('exact_legacy_worker_shard_counts_and_per_scene_seals')
        else:
            raise ValueError('Missing rank identity without the qualified historical worker schema')
    return sorted(set(modes))


def summarize(values, logs, reference=None):
    if values.shape != (len(logs), len(METRICS)) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Invalid complete score matrix')
    result = dict(mean={k: float(values[:, i].mean()) for i, k in enumerate(METRICS)},
        mean_percent={k: float(100 * values[:, i].mean()) for i, k in enumerate(METRICS)},
        zero_scenes=int((values[:, 0] == 0).sum()), fully_safe_scenes=int((values[:, 1:4] == 1).all(1).sum()),
        score_quantiles=dict(zip(['p01', 'p05', 'p10', 'p50'], np.quantile(values[:, 0], [.01, .05, .1, .5]).tolist())))
    if reference is not None:
        if reference.shape != values.shape or not np.isfinite(reference).all(): raise ValueError('Unmatched baseline')
        delta = values[:, 0] - reference[:, 0]; safe = (reference[:, 1:4] == 1).all(1)
        result['paired'] = dict(PDMS_log_cluster_bootstrap=log_bootstrap_difference(delta, logs),
            positive_gain_sum=float(delta.clip(0).sum()), negative_degradation_sum=float((-delta).clip(0).sum()),
            nonzero_to_zero=int(((reference[:, 0] > 0) & (values[:, 0] == 0)).sum()),
            zero_to_nonzero=int(((reference[:, 0] == 0) & (values[:, 0] > 0)).sum()),
            practical_improved_ge005=int((delta >= .005).sum()), practical_degraded_le_minus005=int((delta <= -.005).sum()),
            reference_safe_population=int(safe.sum()),
            reference_safe_new_joint_violation=int((safe & (values[:, 1:4] < 1).any(1)).sum()),
            safety_transitions={key: dict(new_violation=int(((reference[:, i] == 1) & (values[:, i] < 1)).sum()),
                recovery=int(((reference[:, i] < 1) & (values[:, i] == 1)).sum())) for i, key in enumerate(SAFETY, 1)})
    return result


def read_bank(root):
    root = Path(root); identity = authenticated(root / 'manifest.json')
    origin = Path(identity['original_evaluation']); source = authenticated(origin / 'manifest.json')
    population = json.loads(Path(POPULATION).read_text()); validate_input_manifest(source, population)
    if source['scoring'] != '2dee107a4e8358bc873530c0eb2ee16d8761e471eab1d80acc76022e789654e1':
        raise ValueError('Original bank is not the qualified canonical v1.1 protocol')
    if file_sha(origin / 'manifest.json') != identity['original_manifest_sha256'] or source['sha256'] != identity['original_manifest_identity']:
        raise ValueError('Original evaluation manifest changed')
    ranks = [json.loads((origin / f'rank{i}.json').read_text()) for i in range(source['shards'])]
    rank_validation = validate_origin_ranks(source, ranks)
    attempts = [json.loads(p.read_text()) for p in sorted((root / 'attempts').glob('*.json'))]
    if not attempts or any(r['status'] == 'RUNNING' for r in attempts): raise ValueError('Scoring workers still running')
    if any(r['identity_sha256'] != identity['sha256'] for r in attempts): raise ValueError('Mixed scoring attempts')
    if not any(r['status'] == 'ASSIGNED_COMPLETE' for r in attempts): raise ValueError('No completed scoring invocation')
    tokens = population['navtest_tokens']; logs = [population['token_logs'][t] for t in tokens]
    rows = {version: {mode: [] for mode in identity['modes']} for version in ('v1.0', 'v1.1')}
    if identity['protocol']['version'] != 'v1.0': raise ValueError('This analysis pairs native v1.0 rescoring with original canonical v1.1')
    seals = []
    for token, log in zip(tokens, logs):
        path = root / 'scenes' / token[:2] / (token + '.pt')
        original = origin / 'scenes' / token[:2] / (token + '.pt')
        seal = json.loads(path.with_suffix('.json').read_text()); old_seal = json.loads(original.with_suffix('.json').read_text())
        if (seal['token'] != token or seal['log'] != log or seal['identity_sha256'] != identity['sha256'] or
                seal['payload_sha256'] != file_sha(path) or old_seal['payload_sha256'] != file_sha(original) or
                seal['original_prediction_sha256'] != old_seal['payload_sha256'] or
                old_seal['identity_sha256'] != source['sha256']):
            raise ValueError('Changed/missing paired scene; never drop it from denominator')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        original_payload = torch.load(original, map_location='cpu', weights_only=False)
        if (payload['identity_sha256'] != identity['sha256'] or payload['token'] != token or payload['log'] != log or
                payload['original_v1_1_metrics'] != old_seal['metrics']):
            raise ValueError('Rescore payload mismatched')
        for mode in identity['modes']:
            row = payload['records'][mode]; old = original_payload['records'][mode]
            request = np.ascontiguousarray(old['requested_trajectory'])
            if (row['status'] != 'OK' or old['status'] != 'OK' or
                    row['metrics'] != seal['metrics'][mode] or old['metrics'] != old_seal['metrics'][mode] or
                    row['protocol_sha256'] != identity['protocol']['sha256'] or
                    row['request_sha256'] != hashlib.sha256(request.tobytes()).hexdigest()):
                raise ValueError('Official feedback or requested trajectory changed')
            rows['v1.0'][mode].append([row['metrics'][k] for k in METRICS])
            rows['v1.1'][mode].append([old['metrics'][k] for k in METRICS])
        seals.append((token, seal['payload_sha256'], old_seal['payload_sha256']))
    arrays = {v: {m: np.asarray(x, dtype=np.float64) for m, x in modes.items()} for v, modes in rows.items()}
    starts = [datetime.fromisoformat(r['started_utc']).timestamp() for r in attempts]
    ends = [datetime.fromisoformat(r['ended_utc']).timestamp() for r in attempts if 'ended_utc' in r]
    cost = dict(additional_model_predictions=0, original_rank_validation=rank_validation,
        logical_scored_trajectories=2 * len(tokens),
        committed_actual_scoring_calls=sum(r['physical_scoring_calls'] for r in attempts),
        cumulative_CPU_worker_scoring_seconds=sum(r['scoring_worker_seconds'] for r in attempts),
        cumulative_invocation_wall_seconds=sum(r['wall_seconds'] for r in attempts),
        elapsed_first_start_to_last_end_seconds=max(ends) - min(starts) if ends else None,
        failed_attempts=sum(r['status'] != 'ASSIGNED_COMPLETE' for r in attempts),
        incomplete_attempt_call_cost='UNKNOWN_NOT_ZERO' if any(r['status'] != 'ASSIGNED_COMPLETE' for r in attempts) else None,
        note='CPU worker sums are not GPU-hours or end-to-end wall. Original inference and one-time cache costs are separate.')
    return identity, source, logs, arrays, fingerprint(seals), cost


def main():
    p = argparse.ArgumentParser(__doc__); p.add_argument('--rescored', required=True)
    p.add_argument('--reference'); p.add_argument('--output', required=True)
    a = p.parse_args(); output = Path(a.output)
    if output.exists(): raise FileExistsError('Preserve prior derived summary')
    identity, origin, logs, arrays, digest, cost = read_bank(a.rescored)
    baseline = None; reference_identity = None
    if a.reference:
        ref_id, ref_origin, ref_logs, baseline, _, _ = read_bank(a.reference)
        for key in ('population_identity', 'tokens_sha256', 'scenes', 'logs', 'seed', 'modes', 'protocol', 'cache_identity'):
            if identity[key] != ref_id[key]: raise ValueError('Unmatched paired protocol: ' + key)
        if ref_logs != logs: raise ValueError('Unmatched scene/log pairing')
        if origin.get('stage') == 'RL' and origin.get('new_SFT') != ref_origin.get('new_SFT'):
            raise ValueError('RL increment must use its own exact SFT parent')
        reference_identity = ref_id['sha256']
    result = dict(schema='dual_release_full_Navtest_analysis_v1', status='COMPLETE_POPULATION',
        scenes=len(logs), logs=len(set(logs)), seed=2101, identity=identity,
        source_analysis_sha256=file_sha(__file__), all_payloads_digest=digest,
        reference=a.reference, reference_identity=reference_identity, cost=cost,
        uncertainty='Log-cluster bootstrap at fixed training/inference seed; not training or inference seed uncertainty',
        versions={v: {m: summarize(x, logs, baseline[v][m] if baseline is not None else None)
                      for m, x in modes.items()} for v, modes in arrays.items()},
        cross_version_score_change_not_RL_gain={m: log_bootstrap_difference(arrays['v1.1'][m][:, 0] - arrays['v1.0'][m][:, 0], logs)
                                               for m in identity['modes']})
    output.parent.mkdir(parents=True, exist_ok=True); atomic_json(output, result)
    print(json.dumps(dict(status=result['status'], scenes=len(logs), output=str(output),
        PDMS={v: {m: row['mean_percent']['score'] for m, row in modes.items()} for v, modes in result['versions'].items()})))


if __name__ == '__main__': main()
