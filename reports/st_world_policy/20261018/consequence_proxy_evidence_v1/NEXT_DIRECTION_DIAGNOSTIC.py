"""Read-only feasibility audit. Does NOT generate perturbations or call a scorer.

Only the fixed C1 boundary and candidate zero are admissible. Payloads are trusted
local project artifacts and are deserialized only after receipt SHA256 validation.
The budget counts file reads; decompressed allocations have a separate cap.
"""
import argparse
import hashlib
import io
import json
import lzma
import os
from pathlib import Path
import pickle
import sys
import time

ROOT = Path('/mnt/project/st-world-policy-20261016')
RUN = ROOT / 'formal/consequence/runs/C1_fulltrain_seed42_official_4096_v1'
RUNTIME = Path('/mnt/project/VLA-Drive-st-flow-context-runtime-v3-20261017')
TRAIN = Path('/mnt/project/ddp-foresight-artifacts/20260928/student_train_v1')
FEEDBACK = RUN / 'official_feedback/27ed91c86ce0f1cb430689d1e49ff7d949cf114956393406a95b06777a327b37'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def identity(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


class Budget:
    def __init__(self):
        self.limit = 64 * 2**20
        # Conservative allowance for discovery and the earlier stopped probe.
        # This is an upper bound, not a claim of exact measured discovery IO.
        self.used = 35 * 2**20
        self.entries = []

    def charge(self, name, size, digest=None):
        if self.used + size > self.limit:
            raise RuntimeError('READ_BUDGET_EXCEEDED: ' + name)
        self.used += size
        self.entries.append({'path': name, 'bytes': size, 'sha256': digest})

    def read(self, path):
        path = Path(path)
        size = path.stat().st_size
        if self.used + size > self.limit:
            raise RuntimeError('READ_BUDGET_EXCEEDED: ' + str(path))
        data = path.read_bytes()
        if len(data) != size:
            raise RuntimeError('INPUT_CHANGED: ' + str(path))
        self.charge(str(path), size, sha(data))
        return data

    def json(self, path):
        return json.loads(self.read(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('Set CUDA_VISIBLE_DEVICES to the empty string; CPU-only audit.')
    if args.output.exists():
        parser.error('Output exists: choose a new attempt path; historical evidence is immutable.')
    started = time.time()
    import numpy as np
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(RUNTIME / 'navsim_v1.1/navsim'))
    b = Budget()
    reg = b.json(ROOT / 'formal/consequence/registrations/C1_fulltrain_seed42_official_4096_v1/REGISTRATION.json')
    run_identity = b.json(RUN / 'identity.json')
    result = b.json(RUN / 'RESULT.json')
    current = b.json(TRAIN / 'identity.json')
    index = b.json(TRAIN / 'index.json')
    assert current['split'] == 'train'
    assert identity(index) == current['index_sha256']
    train_lookup = {r['token']: r['log'] for r in index}
    selected = []
    receipts = []
    # Selection is fixed before any official metric values are inspected.
    for rank in range(4):
        receipt_path = RUN / f'rollouts/boundary_004095/rank_{rank}_COMPLETE.json'
        receipt = b.json(receipt_path)
        data = b.read(receipt['payload']['path'])
        assert sha(data) == receipt['payload']['sha256']
        payload = torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
        rollout = payload['rollout']
        assert payload['identity'] == receipt['identity']
        assert rollout['policy']['optimizer_step'] == 4095
        assert rollout['policy'] == result['last_completed_policy']
        assert rollout['flow_steps'] == 10
        assert rollout['raw_representation'] == 'normalized_xy_sin_cos'
        assert tuple(rollout['raw_actions'].shape) == (8, 4, 8, 4)
        assert torch.isfinite(rollout['raw_actions']).all().item()
        receipts.append({'path': str(receipt_path), 'payload_sha256': sha(data)})
        for slot, scene in enumerate(rollout['scene_ids']):
            selected.append({'rank': rank, 'slot': slot, 'candidate': 0,
                             'scene_id': scene,
                             'candidate_id': f"{rollout['rollout_id']}:{slot}:0",
                             'observation_hash': rollout['observation_hashes'][slot],
                             'condition_hash': rollout['condition_hashes'][slot],
                             'raw_sha256': sha(rollout['raw_actions'][slot, 0].numpy().tobytes())})
    assert len(selected) == len({s['scene_id'] for s in selected}) == 32
    selection_sha256 = identity(selected)
    requests = b.json(FEEDBACK / 'REQUESTS.json')
    assert str(requests['policy_version']) == '4095'
    request_lookup = {r['candidate_id']: r for r in requests['requests']}
    complete = b.json(FEEDBACK / 'attempt_00/COMPLETE.json')
    admitted = b.read(FEEDBACK / 'attempt_00/ADMITTED.json')
    assert sha(admitted) == complete['admission_sha256']
    stored = b.read(complete['result']['path'])
    assert sha(stored) == complete['result']['sha256']
    records = torch.load(io.BytesIO(stored), map_location='cpu', weights_only=False)
    record_lookup = {r['identity']['candidate_id']: r for r in records}
    # No scorer object is instantiated, and no model/checkpoint is loaded.
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    scorer_vehicle = get_pacifica_parameters()
    for item in selected:
        request = request_lookup[item['candidate_id']]
        record = record_lookup[item['candidate_id']]
        assert request == record['identity']
        assert request['scene_id'] == item['scene_id']
        assert request['source_log'] == train_lookup[item['scene_id']]
        assert record['status'] == 'VALID_SCORED' and record['valid']
        poses = np.ascontiguousarray(record['requested_trajectory'], dtype=np.float64)
        h = hashlib.sha256()
        h.update(json.dumps({'dtype': poses.dtype.str, 'shape': poses.shape}, sort_keys=True).encode())
        h.update(poses.tobytes())
        assert h.hexdigest() == request['trajectory_sha256']
        assert poses.shape == (8, 3) and np.isfinite(poses).all()
        assert record['executed_states'].shape == (41, 11)
        compressed = b.read(request['metric_cache_path'])
        assert sha(compressed) == request['metric_cache_sha256']
        dims = lambda v: {k: float(getattr(v, k)) for k in ('half_length', 'half_width', 'rear_axle_to_center', 'wheel_base')}
        schema_probe = {'status': 'NOT_DECOMPRESSED_THIS_SUCCESSFUL_AUDIT',
                        'prior_canonical_record_valid': True}
        if item is selected[0]:
            # One schema probe only. Full content hashes are checked for all32.
            decoder = lzma.LZMADecompressor()
            unpacked = decoder.decompress(compressed, max_length=2 * 2**20)
            if not decoder.eof:
                raise RuntimeError('CACHE_SCHEMA_PROBE_DECOMPRESSION_CAP: ' + item['scene_id'])
            cache = pickle.loads(unpacked)
            assert type(cache).__name__ == 'MetricCache'
            assert set(vars(cache)) == {'file_path', 'trajectory', 'ego_state', 'observation', 'centerline', 'route_lane_ids', 'drivable_area_map'}
            area = cache.drivable_area_map
            assert len(area._geometries) == len(area.tokens) == len(area.map_types)
            assert all(g.is_valid for g in area._geometries)
            assert all(g.geom_type in ('Polygon', 'MultiPolygon') for g in area._geometries)
            vehicle = cache.ego_state.car_footprint.vehicle_parameters
            schema_probe = {'status': 'VERIFIED', 'decompressed_bytes': len(unpacked),
                            'map_polygon_count': len(area._geometries),
                            'map_types': sorted({int(v) for v in area.map_types}),
                            'vehicle': dims(vehicle), 'scorer_vehicle': dims(scorer_vehicle),
                            'vehicle_matches_scorer': dims(vehicle) == dims(scorer_vehicle)}
        item.update({'source_log': request['source_log'], 'role': 'train',
                     'trajectory_sha256': request['trajectory_sha256'],
                     'metric_cache_path': request['metric_cache_path'],
                     'metric_cache_sha256': request['metric_cache_sha256'],
                     'metric_cache_compressed_bytes': len(compressed),
                     'schema_probe': schema_probe,
                     'original_DAC': record['metrics']['drivable_area_compliance'],
                     'old_official_record_valid': True,
                     'score_protocol_hash': request['score_protocol_hash']})
    relative_sources = [
        'navsim_v1.1/navsim/navsim/evaluate/pdm_score.py',
        'navsim_v1.1/navsim/navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py',
        'navsim_v1.1/navsim/navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py',
        'navsim_v1.1/navsim/navsim/planning/simulation/planner/pdm_planner/simulation/batch_lqr.py',
        'navsim_v1.1/navsim/navsim/planning/simulation/planner/pdm_planner/simulation/batch_kinematic_bicycle.py',
        'navsim_v1.1/navsim/navsim/planning/simulation/planner/pdm_planner/utils/pdm_array_representation.py',
        'navsim_v1.1/navsim/navsim/planning/simulation/planner/pdm_planner/observation/pdm_occupancy_map.py',
        'starVLA/model/modules/st_world/consequence.py',
    ]
    sources = [{'path': str(RUNTIME / p), 'sha256': sha(b.read(RUNTIME / p))} for p in relative_sources]
    huge = ROOT / 'audits/flow_policy/full_train_contexts_v2/TRAINING_MANIFEST.json'
    out = {'schema': 'road_command_direction_diagnostic_feasibility_v1',
           'status': 'ASSETS_LOCATED_DIRECTION_UNTESTED',
           'independent_review': 'PENDING', 'dispatch_authorized': False,
           'source_runtime_commit': '6714b1e9fb2e0c8059b90b8b4d48cd40796fe345',
           'source_runtime': str(RUNTIME), 'run_dir': str(RUN),
           'policy': result['last_completed_policy'],
           'selection_rule': 'boundary_004095; ranks 0..3; saved slot order 0..7; candidate 0; no score filtering',
           'selection_sha256_before_scores': selection_sha256,
           'selection_count': len(selected), 'unique_logs': len({x['source_log'] for x in selected}),
           'train_index_identity': current, 'rollout_receipts': receipts,
           'native_rollout_schema': 'fpo_rollout_v1; normalized_xy_sin_cos; [8,4,8,4]/rank; Euler steps=10',
           'stored_official_result': complete,
           'selected': selected,
           'existing_DAC_counts': {str(v): sum(x['original_DAC'] == v for x in selected) for v in (0., 1.)},
           'sources': sources,
           'large_manifest': {'path': str(huge), 'bytes': huge.stat().st_size,
                              'content_read_this_audit': False,
                              'registered_sha256': 'b810cafbdaa42afd8554c08be443779e7d66edce56292c0c30940e728c0ed656',
                              'limitation': 'Registered manifest hash not freshly verified; membership independently checked against hashed TRAIN current index and rollout receipt.'},
           'budget': {'hard_read_limit_bytes': b.limit, 'prior_discovery_and_stopped_attempt_reserve_bytes': 35 * 2**20,
                      'accounted_bytes_including_reserve': b.used,
                      'entries': b.entries, 'official_calls': 0, 'GPU_calls': 0,
                      'train_updates': 0, 'elapsed_seconds': time.time() - started},
           'cuda_initialized': torch.cuda.is_initialized()}
    assert not out['cuda_initialized']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(out, indent=2, sort_keys=True) + '\n'
    if args.output.exists():
        raise FileExistsError('Do not overwrite previous audit evidence; choose a new attempt path.')
    args.output.write_text(encoded)
    print(json.dumps({'status': out['status'], 'scenes': len(selected),
                      'logs': out['unique_logs'], 'read_bytes_with_reserve': b.used,
                      'DAC': out['existing_DAC_counts'], 'official_calls': 0}))


if __name__ == '__main__':
    main()
