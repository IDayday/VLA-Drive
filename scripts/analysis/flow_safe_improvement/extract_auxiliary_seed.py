"""Read one preselected seed from sealed official v2 log shards; no rescoring."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import file_sha, fingerprint
from starVLA.rl.safe_flow_improvement.evaluation_override import evaluation_seeds


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('source', 'population', 'evaluation-override', 'output'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args(); source = Path(a.source); out = Path(a.output)
    population = json.loads(Path(a.population).read_text()); seeds, override = evaluation_seeds(population, a.evaluation_override)
    identity = json.loads((source/'manifest.json').read_text()); sample = identity['sample_manifest']
    if (seeds != [2101] or sample['population_identity'] != population['identity_sha256']
            or sample['cohort'] != 'dev_tokens' or sample['mode'] != 'ode'
            or not set(seeds) <= set(sample['seeds']) or identity['protocol']['version'] != 'navsim_v2_official_one_stage'):
        raise ValueError('Only matched preselected original official v2 DEV outputs may be reused')
    tokens = population['dev_tokens']; logs = sorted({population['token_logs'][t] for t in tokens}); frames=[]; shards=[]
    for log in logs:
        csv_path = source/'logs'/f'2101_{fingerprint(log)}.csv'
        seal = json.loads(csv_path.with_suffix('.json').read_text())
        if seal['identity_sha256'] != identity['sha256'] or seal['sha256'] != file_sha(csv_path) or seal['seed'] != 2101:
            raise ValueError('Missing or changed v2 log seal')
        frame = pd.read_csv(csv_path, dtype={'token': str, 'log_name': str}, float_precision='round_trip')
        if set(frame.token) != {t for t in tokens if population['token_logs'][t] == log}:
            raise ValueError('Adjacent-frame aggregation log population changed')
        frames.append(frame); shards.append(seal)
    frame = pd.concat(frames, ignore_index=True)
    if len(frame) != len(tokens) or frame.token.duplicated().any() or not frame.valid.all() or not np.isfinite(frame.score).all():
        raise ValueError('Failures cannot disappear from the selected seed denominator')
    numeric = ['score', 'pdm_score', 'no_at_fault_collisions', 'drivable_area_compliance', 'driving_direction_compliance',
               'traffic_light_compliance', 'ego_progress', 'time_to_collision_within_bound', 'lane_keeping', 'history_comfort']
    report = dict(status='COMPLETE_OFFICIAL_V2_ONE_STAGE_AUXILIARY', scenes=len(tokens), seeds=seeds,
        source_identity_sha256=identity['sha256'], source_root=str(source), source_sample_manifest=sample,
        evaluation_override=override, selected_original_first_seed_without_score_based_selection=True,
        prior_five_seed_results_had_already_been_observed=True,
        protocol='navsim_v2_official_one_stage', means={k: float(frame[k].mean()) for k in numeric},
        two_frame_available_rows=int(frame.two_frame_extended_comfort.notna().sum()), total_scene_seed_rows=len(frame),
        missing_two_frame_kept_unavailable=True, full_logs_kept_together=True, new_scoring_calls=0,
        logical_scoring_calls=len(tokens), historical_selected_seed_CPU_seconds=sum(s['CPU_seconds'] for s in shards),
        v1_score_not_derived_from_v2_pdm_score=True, original_log_shard_hashes=[s['sha256'] for s in shards])
    report['sha256']=fingerprint(report); out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'report.json', report); frame['inference_seed']=2101; frame.to_csv(out/'seed2101.csv', index=False)
    print(json.dumps(dict(status=report['status'], EPDMS=report['means']['score'], new_scoring_calls=0)), flush=True)


if __name__ == '__main__':
    main()
