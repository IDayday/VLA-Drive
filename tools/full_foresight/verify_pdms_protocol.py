"""Bounded real-cache regression for the map precision/reference-progress audit.

Does not modify caches, scoring mathematics, model weights or training source.
Checks every development cache hash, then scores fixed log representatives and
known historical discrepancy cases against the independently audited GT CSV.
"""
import argparse
import copy
import csv
from dataclasses import asdict
import hashlib
import json
import lzma
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('devkit', 'cache-root', 'audit-report', 'audit-csv',
                 'audit-comparison', 'audit-reference-cases', 'audit-cache-root', 'output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, a.devkit)
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from shapely import get_coordinates
    from tools.local_interaction_mask_v2.score_async import score_chunk, python_tree_digest

    class TracedScorer(PDMScorer):
        def _calculate_ego_area(self):
            self.audit_trace.append('ego_area')
            return super()._calculate_ego_area()

        def _calculate_no_at_fault_collision(self):
            self.audit_trace.append('collision')
            return super()._calculate_no_at_fault_collision()

        def _calculate_drivable_area_compliance(self):
            self.audit_trace.append('DAC')
            return super()._calculate_drivable_area_compliance()

        def _calculate_progress(self):
            self.audit_trace.append('progress')
            return super()._calculate_progress()

        def score_proposals(self, states, *args, **kwargs):
            self.audit_trace = []
            self.audit_state_shape = list(states.shape)
            assert states.shape[:2] == (2, 41), 'Exactly reference + one ego, 40x0.1s'
            result = super().score_proposals(states, *args, **kwargs)
            assert self.audit_trace == ['ego_area', 'collision', 'DAC', 'progress']
            return result

    root = Path(a.cache_root)
    index = json.loads((root / 'cache_index.json').read_text())
    generated = {r['token']: r for r in csv.DictReader((root / 'scenes.csv').open())}
    assert len(index) == len(generated) and len({r['token'] for r in index}) == len(index)
    for row in index:
        assert generated[row['token']]['status'] == 'ok'
        assert digest(row['cache_path']) == generated[row['token']]['sha256']
    source_identity = json.loads((root / 'identity.json').read_text())
    assert source_identity['navsim_python_tree_sha256'] == python_tree_digest(Path(a.devkit) / 'navsim')
    groups = {}
    for row in index:
        groups.setdefault(row['log'], []).append(row)
    # Fixed before examining this evaluator's outputs: one token-hash per log,
    # all historical EP cases in the dev population, and eight DAC cases.
    selected = {min(rows, key=lambda r: hashlib.sha256(r['token'].encode()).hexdigest())['token']
                for rows in groups.values()}
    population = {r['token'] for r in index}
    ep = {r['token'] for r in csv.DictReader(open(a.audit_reference_cases))} & population
    selected |= ep
    comparison = {r['token']: r for r in csv.DictReader(open(a.audit_comparison)) if r['token'] in population}
    dac = [t for t, r in comparison.items()
           if r.get('drivable_area_compliance_20260903') != r.get('drivable_area_compliance_20260927')]
    # Column names must be explicit, never silently treat missing columns as equal.
    required = {'drivable_area_compliance_20260903', 'drivable_area_compliance_20260927'}
    assert comparison and required <= next(iter(comparison.values())).keys()
    selected |= set(sorted(dac, key=lambda t: hashlib.sha256(t.encode()).hexdigest())[:8])
    expected = {r['token']: r for r in csv.DictReader(open(a.audit_csv)) if r['token'] in selected}
    sampling = TrajectorySampling(num_poses=40, interval_length=.1)
    records = []
    for item in [r for r in index if r['token'] in selected]:
        token = item['token']
        try:
            with lzma.open(item['cache_path'], 'rb') as stream:
                cache = pickle.load(stream)
            audit_path = Path(a.audit_cache_root) / (token + '.xz')
            with lzma.open(audit_path, 'rb') as stream:
                audited = pickle.load(stream)
            assert not hasattr(cache, 'pdm_progress'), 'Fixed legacy progress cache forbidden'
            assert len(cache.observation._occupancy_maps) == 51, 'Need full 5s scoring environment'
            geometries = cache.drivable_area_map.__reduce__()[1][2]
            coordinates = get_coordinates(geometries)
            assert coordinates.dtype == np.float64 and np.isfinite(coordinates).all()
            quantization = np.abs(coordinates - coordinates.astype(np.float32).astype(np.float64))
            assert np.count_nonzero(quantization) > 0, 'Already quantized map rejected'
            poses = np.asarray(audited['supervision'], dtype=np.float64)
            assert poses.shape == (8, 3) and np.isfinite(poses).all()
            scorer = TracedScorer(sampling)
            result = asdict(pdm_score(cache, Trajectory(poses), sampling, PDMSimulator(sampling), scorer))
            reference = asdict(pdm_score(copy.deepcopy(audited['cache']), Trajectory(poses),
                                        sampling, PDMSimulator(sampling), PDMScorer(sampling)))
            proposal = out / (token + '.npz')
            np.savez(proposal, trajectory=poses)
            worker = score_chunk([dict(item, variant='GT_protocol_audit', proposal_path=str(proposal),
                                      proposal_sha256=digest(proposal))])[0]
            assert worker['status'] == 'ok', worker
            differences = {k: max(abs(float(result[k]) - float(reference[k])),
                                  abs(float(result[k]) - float(expected[token][k])),
                                  abs(float(result[k]) - float(worker[k]))) for k in result}
            formula = result['no_at_fault_collisions'] * result['drivable_area_compliance'] * (
                5 * result['ego_progress'] + 5 * result['time_to_collision_within_bound'] +
                2 * result['comfort']) / 12
            assert abs(formula - result['score']) < 1e-12
            assert max(differences.values()) < 1e-8, differences
            record = dict(token=token, log=item['log'], status='ok', metrics=result,
                          max_difference=max(differences.values()), map_dtype=str(coordinates.dtype),
                          original_map_not_fp32_quantized=True, max_if_quantized_error_m=float(quantization.max()),
                          scorer_call_order=scorer.audit_trace, scored_shape=scorer.audit_state_shape,
                          reference_raw_progress=float(scorer._progress_raw[0]),
                          reference_effective_progress=float(scorer._progress_raw[0] * scorer._multi_metrics[:, 0].prod()),
                          metric_cache_sha256=digest(item['cache_path']), audited_cache_sha256=digest(audit_path))
        except Exception as error:
            record = dict(token=token, log=item['log'], status='failed', error=repr(error))
        records.append(record)
        with (out / 'rows.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
    failed = sum(r['status'] != 'ok' for r in records)
    summary = dict(passed=failed == 0, evaluated=len(records), failed=failed,
                   full_cache_population_hashed=len(index), logs=len(groups), ep_cases=len(ep),
                   selected_dac_cases=min(8, len(dac)), records=records,
                   evaluation_source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   audit_script_sha256=digest(__file__),
                   source_files={k: dict(path=getattr(a, k), sha256=digest(getattr(a, k)))
                                 for k in ('audit_report', 'audit_csv', 'audit_comparison', 'audit_reference_cases')},
                   metric_index_sha256=digest(root / 'cache_index.json'),
                   navsim_tree_sha256=python_tree_digest(Path(a.devkit) / 'navsim'),
                   worker_sha256=digest(Path(__file__).parents[1] / 'local_interaction_mask_v2/score_async.py'),
                   scope='Full dev cache hashes; bounded GT parity against audited full-precision/reference path. '
                         'Not a full-population HumanAgent rerun, not model performance, not Navtest validation.',
                   protocol='v1; 8x0.5s ego; 40x0.1s simulation; 5s full observations; '
                            'per-scene reference plus one ego; NC*DAC*(5EP+5TTC+2Comfort)/12; no EPDMS mixing')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'records'}))
    if failed:
        raise RuntimeError('Scoring protocol failed; all failed rows retained')


if __name__ == '__main__':
    main()
