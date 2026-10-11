"""Immutable prediction/attempt records and a verified completion receipt.

A retry never resamples a prediction. Invalid predictions are terminal quality
facts; failed official calls remain missing labels, not valid zero rewards.
"""
from dataclasses import asdict
from pathlib import Path
import time
import subprocess
import numpy as np
from .contracts import CandidateRecord, ScoreRecord, strict_record, require
from .io import atomic_json, digest, file_hash, read_json, ContractError


def verify_row(row, scene, candidate, protocol_hash):
    require(row['scene_id'] == scene.scene_id and row['observation_hash'] == scene.observation_hash
            and row['context_hash'] == scene.metric_context_hash, 'contract_mismatch: Base evaluation scene changed')
    require(file_hash(scene.metric_context_ref) == scene.metric_context_hash, 'contract_mismatch: metric context changed')
    require(file_hash(row['trajectory_ref']) == row['trajectory_hash'], 'contract_mismatch: prediction changed')
    if row.get('score'):
        score = strict_record(ScoreRecord, row['score'])
        require(score.candidate_key == candidate.key and score.score_protocol_hash == protocol_hash
                and score.metric_context_hash == scene.metric_context_hash, 'contract_mismatch: score dependency changed')


def candidate_for(row, scene, contract, pipeline, checkpoint_hash):
    return CandidateRecord(1, scene.scene_id, 'expert_0', checkpoint_hash, digest(contract),
        row['trajectory_hash'], row['trajectory_ref'], pipeline.trajectory_contract.raw_representation,
        'ego_relative_rear_axle', pipeline.trajectory_contract.horizon, pipeline.trajectory_contract.dt,
        row['finite'], pipeline.contract['iqe_code_hash'], pipeline.config['config_hash'], 'base_independent_evaluation')


def score_prediction(folder, row, scene, candidate, backend, *, budget, official):
    """Budget is calls per scene per invocation; previous failures remain immutable."""
    directory = folder/'scenes'/scene.scene_id
    directory.mkdir(parents=True, exist_ok=True)
    completed = directory/'SCORED.json'
    if completed.exists():
        saved = read_json(completed)
        require(saved['prediction_hash'] == digest(row), 'contract_mismatch: scored prediction identity')
        result = row | {'score': saved['score'], 'outcome': 'valid_scored'}
        verify_row(result, scene, candidate, backend.protocol_hash)
        require(result['score']['score_valid'], 'invalid completed score')
        return result
    if not row['finite']:
        return row | {'outcome': 'invalid_prediction', 'score': None}
    if not official:
        return row | {'outcome': 'prediction_only', 'score': None}
    attempts = sorted(directory.glob('attempt_*.json'))
    # A worker can crash after its successful attempt but before its receipt.
    for path in attempts:
        attempt = read_json(path)
        require(attempt['prediction_hash'] == digest(row), 'contract_mismatch: scoring attempt')
        if attempt.get('score') and attempt['score']['score_valid']:
            result = row | {'score': attempt['score'], 'outcome': 'valid_scored'}
            verify_row(result, scene, candidate, backend.protocol_hash)
            atomic_json(completed, {'prediction_hash': digest(row), 'score': attempt['score']}, immutable=True)
            return result
    last = None
    for index in range(len(attempts), len(attempts) + budget):
        start = time.time()
        score = None
        try:
            score = asdict(backend.score(candidate, scene))
            verify_row(row | {'score': score}, scene, candidate, backend.protocol_hash)
            error = score.get('error_reason')
        except (TimeoutError, subprocess.TimeoutExpired, ConnectionError, OSError) as exception:
            error = repr(exception)
        # ContractError/ValueError is intentionally NOT downgraded to a timeout.
        last = {'prediction_hash': digest(row), 'started_unix': start, 'finished_unix': time.time(),
                'score': score, 'error': error, 'outcome': 'valid_scored' if score and score['score_valid'] else 'transient_scoring_failure'}
        atomic_json(directory/f'attempt_{index:04d}.json', last, immutable=True)
        if score and score['score_valid']:
            atomic_json(completed, {'prediction_hash': digest(row), 'score': score}, immutable=True)
            return row | {'score': score, 'outcome': 'valid_scored'}
    if last is None and attempts:
        last = read_json(attempts[-1])
    return row | {'score': None, 'outcome': 'transient_scoring_failure',
                  'error': (last or {}).get('error', 'retry budget exhausted'), 'attempt_count': len(attempts) + budget}


def summarize(rows, components):
    scores = [r['score'] for r in rows if r.get('score') and r['score']['score_valid']]
    transient = sum(r['outcome'] == 'transient_scoring_failure' for r in rows)
    invalid = sum(r['outcome'] == 'invalid_prediction' for r in rows)
    value = float(np.mean([s['total_score_01'] for s in scores])) if scores else None
    out = {'scenes': len(rows), 'requested_scenes': len(rows), 'nonfinite': invalid,
           'invalid_predictions': invalid, 'transient_scoring_failures': transient, 'unfinished': transient,
           'valid_official_scores': len(scores), 'PDMS_01': value if len(scores) == len(rows) else None,
           'PDMS_valid_subset_01': value,
           'zero_rate': sum(s['total_score_01'] <= 1e-8 for s in scores)/len(scores) if len(scores) == len(rows) and scores else None,
           'zero_rate_valid_subset': sum(s['total_score_01'] <= 1e-8 for s in scores)/len(scores) if scores else None}
    for key in ('raw_IL_loss', 'ADE_m', 'FDE_m', 'yaw_error_rad'):
        values = [r[key] for r in rows if r[key] is not None]
        out[key] = float(np.mean(values)) if values else None
    out['components'] = {c: float(np.mean([s['named_components'][c] for s in scores if s['component_valid_masks'].get(c, False)]))
        for c in components if any(s['component_valid_masks'].get(c, False) for s in scores)}
    return out


def verify_completion(evaluation_path, *, require_all_valid=False):
    report = read_json(evaluation_path)
    require(report.get('status') == 'COMPLETE', 'BASE_EVALUATION_INCOMPLETE')
    summary = report['summary']
    require(summary.get('unfinished', 0) == 0 and summary.get('transient_scoring_failures', 0) == 0, 'BASE_EVALUATION_INCOMPLETE')
    rows = report['rows']
    require(len(rows) == summary['scenes'], 'evaluation row count mismatch')
    require(summary['valid_official_scores'] == sum(bool(r.get('score') and r['score']['score_valid']) for r in rows), 'evaluation score denominator mismatch')
    if require_all_valid:
        require(summary['valid_official_scores'] == summary['scenes'] and summary['nonfinite'] == 0, 'BASE_EVALUATION_INCOMPLETE_OR_INVALID')
    if report.get('schema_version') == 2:
        receipt = read_json(str(evaluation_path) + '.COMPLETE.json')
        require(receipt['evaluation_hash'] == file_hash(evaluation_path) and receipt['scene_ids'] == [r['scene_id'] for r in rows], 'evaluation completion identity mismatch')
        for path, checksum in receipt['outputs'].items():
            require(file_hash(path) == checksum, 'evaluation completed output changed')
    else:
        # Legacy COMPLETE with missing scores never becomes valid by migration.
        require(summary['valid_official_scores'] == summary['scenes'], 'LEGACY_INCOMPLETE_EVALUATION')
    return report


def complete_evaluation(path, rows):
    """Also repairs the atomic-report/receipt crash window after row validation."""
    folder = Path(path).parent
    files = list((folder/'scenes').rglob('*.json')) + [Path(r['trajectory_ref']) for r in rows]
    atomic_json(str(path)+'.COMPLETE.json', {'evaluation_hash':file_hash(path),
        'scene_ids':[r['scene_id'] for r in rows], 'outputs':{str(p):file_hash(p) for p in files}}, immutable=True)
