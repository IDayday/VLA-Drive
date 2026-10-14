"""Read-only saturation cross-tabs from sealed census records, without rescoring."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from starVLA.rl.constrained_flow_noise.contracts import file_sha, fingerprint, require_dispatch
from starVLA.rl.constrained_flow_noise.runtime import OrderedScenes
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require_dispatch()
    root, output = Path(args.root), Path(args.output)
    if output.exists():
        raise FileExistsError('Snapshots are immutable')
    registration = json.loads((root/'registration.json').read_text())
    if registration['sha256'] != fingerprint({k:v for k,v in registration.items() if k!='sha256'}):
        raise ValueError('Registration identity changed')
    population = json.loads(Path(registration['population']).read_text())
    stream = OrderedScenes(population['train_tokens'], registration['seed'])
    paths = sorted((root/'scenes').glob('*/*.json'))
    snapshot_started = datetime.now(timezone.utc).isoformat()
    groups = defaultdict(Counter)
    rare = defaultdict(Counter)
    metric_patterns = defaultdict(Counter)
    means = defaultdict(float)
    positions, logs, seals = set(), set(), {}
    flags = ('zero_final_advantage_groups', 'official_safety_cost_varies_groups',
             'geometric_cost_varies_groups', 'progress_only_official_reward_groups',
             'groups_with_safe_candidate')
    official = ('no_at_fault_collisions', 'drivable_area_compliance',
                'time_to_collision_within_bound', 'ego_progress', 'comfort',
                'driving_direction_compliance')
    for path in paths:
        contents = path.read_bytes()
        row = json.loads(contents)
        position = row['position']
        if (row['registration_sha256'] != registration['sha256'] or
                position in positions or not 0 <= position < registration['population_count'] or
                row['token'] != stream.at(position) or
                row['log'] != population['token_logs'][row['token']]):
            raise ValueError('Invalid or duplicated census scene')
        positions.add(position); logs.add(row['log']); seals[str(path)] = hashlib.sha256(contents).hexdigest()
        summary = row['summary']
        if summary['scenes'] != 1 or summary['candidates'] != 16:
            raise ValueError('Every group must preserve all 16 candidates')
        category = ('all_one' if summary['all_PDMS_one'] else
                    'all_zero' if summary['all_zero_PDMS_groups'] else
                    'other_tied' if summary['zero_PDMS_advantage_groups'] else 'mixed')
        groups[category]['scenes'] += 1
        for flag in flags:
            groups[category][flag] += summary[flag]
        groups[category]['nonzero_final_advantage_groups'] += 1-summary['zero_final_advantage_groups']
        for key, value in summary['official_means'].items():
            means[key] += value
        if category not in ('all_zero', 'other_tied'):
            continue
        # Explain rare ties from the original scored payload, never infer the
        # cause of a zero score from its scalar reward alone.
        payload_path = path.with_suffix('.pt')
        if file_sha(payload_path) != row['payload_sha256']:
            raise ValueError('Changed rare-group scored payload')
        payload = torch.load(payload_path, map_location='cpu', weights_only=False)
        values = {k:np.asarray([r['metrics'][k] for r in payload['records']]) for k in official}
        scores = np.asarray([r['metrics']['score'] for r in payload['records']])
        if len(scores) != 16 or not np.isfinite(scores).all() or not (scores == scores[0]).all():
            raise ValueError('Rare-group summary/payload disagree')
        if category == 'all_zero' and not (scores == 0).all():
            raise ValueError('Zero-group summary/payload disagree')
        for key, value in values.items():
            rare[category][key+'_all_zero_groups'] += int((value==0).all())
            rare[category][key+'_any_nonpass_groups'] += int((value<1).any())
            rare[category][key+'_varies_groups'] += int((value!=value[0]).any())
        nc, dac = (values[k] for k in ('no_at_fault_collisions', 'drivable_area_compliance'))
        if category == 'all_zero':
            # Locked v1.1 MultiMetricIndex contains NC and DAC. DDC is a
            # separately reported weighted metric with default weight zero.
            rare[category]['all_candidates_have_zero_multiplicative_factor'] += int(((nc*dac)==0).all())
        for i in range(16):
            pattern = ','.join(f'{k}={values[k][i]:.8g}' for k in official if values[k][i] != 1)
            metric_patterns[category][pattern or 'all_components_one'] += 1
    assert sum(x['scenes'] for x in groups.values()) == len(paths)
    report = dict(schema='cfn_population_saturation_snapshot_v1',
                  status='PARTIAL_POPULATION' if len(paths)<registration['population_count'] else 'FULL_COVERAGE_SNAPSHOT',
                  full_payload_verification_completed=(root/'COMPLETE').exists(),
                  snapshot_started_utc=snapshot_started, completed_scenes=len(paths),
                  registered_scenes=registration['population_count'], observed_logs=len(logs),
                  committed_candidate_scores=16*len(paths), groups={k:dict(v) for k,v in groups.items()},
                  rare_group_details={k:dict(v) for k,v in rare.items()},
                  rare_candidate_metric_patterns={k:dict(v) for k,v in metric_patterns.items()},
                  candidate_metric_means={k:v/len(paths) for k,v in means.items()},
                  registration_sha256=registration['sha256'], seal_manifest_sha256=fingerprint(seals),
                  analysis_source_sha256=file_sha(__file__),
                  policy='fixed SFT / original C_SDE / G16; not trained Flow-Noise or single ODE',
                  advantage='fixed initial lambda=[1,1,1,0.1,0.1]; not current training multipliers',
                  boundary='Observed sealed subset only until full coverage; no additional sampling or updates')
    report['sha256'] = fingerprint(report)
    atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
