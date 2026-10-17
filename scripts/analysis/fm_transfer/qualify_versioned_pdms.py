"""Real-cache oracle and request-isolation checks; no model inference."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import lzma
from pathlib import Path
import pickle

import numpy as np
import torch

from scripts.analysis.fm_transfer.build_versioned_pdms_cache import make_spec
from starVLA.rl.fm_transfer.versioned_pdms import initialize, score_requests, cache_path
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.simwam_recipe.contracts import POPULATION, file_sha


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--version', choices=['v1.0', 'v1.1'], required=True)
    p.add_argument('--cache-root', required=True)
    p.add_argument('--evaluation', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    output = Path(a.output)
    if output.exists(): raise FileExistsError('Preserve prior qualification')
    pop, identity, spec = make_spec(a.source, a.version, a.cache_root, POPULATION,
        '/mnt/project/DriveDreamer-Policy/navsim_raw/navsim_logs/test', '/mnt/navsim/maps')
    initialize(spec)
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.common.dataclasses import Trajectory
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    differences = []
    evaluation = Path(a.evaluation)
    manifest = json.loads((evaluation / 'manifest.json').read_text())
    if manifest['cohort'] != 'NAVTEST' or manifest['seed'] != 2101 or manifest['qualification_only']:
        raise ValueError('Use a complete fixed-seed saved Navtest evaluation')
    for token in pop['navtest_tokens'][:2]:
        path = evaluation / 'scenes' / token[:2] / (token + '.pt')
        seal = json.loads(path.with_suffix('.json').read_text())
        if seal['payload_sha256'] != file_sha(path) or seal['identity_sha256'] != manifest['sha256']:
            raise ValueError('Changed historical request')
        data = torch.load(path, map_location='cpu', weights_only=False)
        log = pop['token_logs'][token]
        requests = [data['records'][m]['requested_trajectory'] for m in manifest['modes']]
        rows = score_requests((token, log, requests))
        reversed_rows = score_requests((token, log, requests[::-1]))[::-1]
        changed_neighbor = np.zeros((8, 3))
        independent_rows = [score_requests((token, log, [x]))[0] for x in requests]
        neighbor_rows = score_requests((token, log, [requests[0], changed_neighbor]))
        with lzma.open(cache_path(token, log), 'rb') as stream: cache = pickle.load(stream)
        sampling = TrajectorySampling(num_poses=40, interval_length=.1)
        request_sampling = TrajectorySampling(num_poses=8, interval_length=.5)
        for i, request in enumerate(requests):
            oracle = asdict(pdm_score(cache, Trajectory(request, request_sampling), sampling,
                                     PDMSimulator(sampling), PDMScorer(sampling)))
            assert rows[i]['metrics'] == oracle == reversed_rows[i]['metrics'] == independent_rows[i]['metrics']
            if i == 0: assert neighbor_rows[0]['metrics'] == oracle
            expected = data['records'][manifest['modes'][i]]['metrics']
            maximum = max(abs(oracle[k] - expected[k]) for k in oracle)
            if a.version == 'v1.1' and maximum > 1e-12:
                raise ValueError('Rebuilt official v1.1 differs from locked historical evaluator')
            differences.append(dict(mode=manifest['modes'][i], max_component_delta_vs_v1_1=maximum))
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, dict(status='PASS_REAL_CACHE_OFFICIAL_ORACLE_AND_ISOLATION', version=a.version,
        source_identity=spec['protocol'], cache_identity=identity['sha256'],
        source_sha256=file_sha(__file__), scenes=2, model_forward_calls=0,
        oracle_equal=True, candidate_order_equal=True, split_equal=True, other_candidate_equal=True,
        original_v1_1_parity_checked=a.version == 'v1.1',
        per_request_v1_1_differences=differences,
        nuplan_distribution=importlib.metadata.version('nuplan-devkit'),
        note='Two real scenes qualify the adapter, not the complete Navtest result.'))
    print(json.dumps(dict(status='PASS', version=a.version, output=str(output))))


if __name__ == '__main__': main()
