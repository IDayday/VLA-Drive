"""Sealed full-scene summary with explicit unmeasured transport-loss costs.

Original invocation JSON is never rewritten or declared successfully exited.
Only manifest-bound stale RUNNING records with released rank/worker locks are
accepted as abandoned; their measured compute remains a lower bound.
"""
import argparse
import hashlib
import json
from pathlib import Path


def invocation_costs(paths, abandoned):
    rows = []; lost = []; starts = []; ends = []
    for path in paths:
        raw = Path(path).read_bytes(); row = json.loads(raw)
        if row['status'] == 'RUNNING':
            proof = abandoned.get(str(Path(path).resolve()))
            if (proof is None or proof['sha256'] != hashlib.sha256(raw).hexdigest() or
                    not proof['rank_lock_unheld'] or not proof['worker_lock_unheld']):
                raise ValueError('Unaccounted live/stale invocation; never silently drop RUNNING costs')
            last = proof['last_observed_write_unix']
            starts.append(last - row['wall_seconds']); ends.append(last)
            lost.append(dict(path=str(path), known_wall_seconds=row['wall_seconds'],
                             actual_exit_unconfirmed=True, unmeasured_tail_seconds=None))
        else:
            starts.append(row['end_unix'] - row['wall_seconds']); ends.append(row['end_unix'])
        rows.append(row)
    if not rows:
        raise ValueError('No invocation ledger')
    result = dict(physical_inference_including_retries=sum(v['physical_inference'] for v in rows),
                  reused_scenes=sum(v['reused'] for v in rows),
                  model_loading_seconds=sum(v['model_loading_seconds'] for v in rows),
                  worker_wall_seconds=sum(v['wall_seconds'] for v in rows),
                  elapsed_span_seconds=max(ends) - min(starts),
                  forward_worker_seconds=sum(v['forward_seconds'] for v in rows),
                  scoring_wait_worker_seconds=sum(v['scoring_wait_seconds'] for v in rows),
                  transport_lost_invocations=lost, physical_cost_is_lower_bound=bool(lost),
                  GPU_cost_semantics='Measured shared-worker occupancy; unknown lost-host tail is not zero or exclusive GPU compute.')
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--evaluation', required=True); p.add_argument('--output', required=True)
    p.add_argument('--transport-evidence', required=True)
    a = p.parse_args(); root = Path(a.evaluation); dest = Path(a.output)
    if dest.exists(): raise FileExistsError('Preserve summaries')
    import numpy as np
    from scripts.analysis.qwen_sourceflow.evaluation_integrity import verify_identity, verify_scene
    from starVLA.rl.qwen_sourceflow.contracts import POPULATION, fingerprint, file_sha
    from starVLA.rl.flow_grpo.transactions import atomic_json
    evidence = json.loads(Path(a.transport_evidence).read_text()); body = dict(evidence)
    if body.pop('sha256') != fingerprint(body): raise ValueError('Changed transport evidence')
    identity = verify_identity(json.loads((root / 'identity.json').read_text()))
    pop = json.loads(POPULATION.read_text()); tokens = pop['navtest_tokens']
    if (identity['cohort'] != 'NAVTEST' or identity['count'] != 12146 or
            identity['qualification'] or not identity['full_population'] or
            identity['token_sha256'] != fingerprint(tokens)):
        raise ValueError('Not the complete registered Navtest population')
    files = list((root / 'scenes').glob('*.json'))
    if len(files) != len(tokens) or {f.stem for f in files} != set(tokens):
        raise ValueError('Missing/extra evaluation scenes')
    rows = [verify_scene(json.loads(f.read_text()), identity, token=f.stem, log=pop['token_logs'][f.stem]) for f in files]
    if len({r['log'] for r in rows}) != 136: raise ValueError('Wrong log coverage')
    means = {k: float(np.mean([r['feedback']['metrics'][k] for r in rows]))
             for k, value in rows[0]['feedback']['metrics'].items() if isinstance(value, (int, float))}
    export = identity.get('rl_export')
    result = dict(schema='native_evaluation_summary_transport_recovery_v4', status='COMPLETE',
                  evaluation=str(root.resolve()), scenes=len(rows), logs=136, means=means, PDMS=100 * means['score'],
                  integrity_analysis_sha256=file_sha(Path(__file__).with_name('evaluation_integrity.py')),
                  integrity_checks=dict(identity_seal=True, score_protocol_seal=True, inline_scene_seals=True,
                      scene_feedback_token_log_protocol=True, finite_native_8x3_requests=True, scored_request_equals_policy_request=True),
                  checkpoint=export['checkpoint'] if export else identity['checkpoint'],
                  IL_parent_checkpoint=identity['checkpoint'], policy_type='RL_MERGED' if export else 'IL',
                  model_sha256=export['model_sha256'] if export else identity['parent_model_sha256'],
                  rl_export=export, cohort=identity['cohort'], seed=identity['seed'], metric=identity['metric'],
                  inference=identity['inference'], qualification=False, full_population=True,
                  source_identity=identity['sha256'], logical_single_candidates=len(rows),
                  evaluation_role='USER_REQUESTED_INTERIM',
                  transport_evidence_sha256=evidence['sha256'], analysis_sha256=file_sha(__file__))
    result.update(invocation_costs(list((root / 'invocations').glob('*.json')), evidence['invocations']))
    result['sha256'] = fingerprint(result); atomic_json(dest, result)
    print(json.dumps(dict(status='COMPLETE', PDMS=result['PDMS'], scenes=result['scenes'], cost_lower_bound=result['physical_cost_is_lower_bound'])))


if __name__ == '__main__': main()
