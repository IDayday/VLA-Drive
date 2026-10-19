"""Official single-reference PDMS plus read-only event recording.

Instrument two method ASTs in memory: append a trace only when the official
method updates a collision/TTC event index. No score expression is changed.
The original package on disk is never modified. Verify against an unmodified
scorer on fresh metric caches before formal use.
"""
import ast
from dataclasses import asdict
import importlib
import inspect
import lzma
import pickle
from pathlib import Path
import sys
import textwrap
import numpy as np
from common import digest, file_hash, poses_valid, wrap

_API = None


def api(config):
    global _API
    if _API is not None:
        assert _API['root'] == str(Path(config['navsim_root']).resolve())
        return _API
    root = Path(config['navsim_root']).resolve()
    sys.path.insert(0, str(root))
    # Existing team cache was serialized with NumPy 2; this environment is 1.26.
    # Aliasing the same numeric module changes deserialization lookup, not data.
    if np.__version__.startswith('1.'):
        import numpy.core.numeric
        import numpy.core.multiarray
        sys.modules.setdefault('numpy._core.numeric', numpy.core.numeric)
        sys.modules.setdefault('numpy._core.multiarray', numpy.core.multiarray)
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.common.dataclasses import Trajectory
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import EgoAreaIndex, StateIndex
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    module = importlib.import_module(PDMScorer.__module__)
    assert Path(module.__file__).resolve().is_relative_to(root)

    class DiagnosticScorer(PDMScorer):
        def _record_event(self, kind, proposal_idx, time_idx, token, **details):
            obj = self._observation.unique_objects[token]
            self.events.append(dict(kind=kind, proposal_index=int(proposal_idx), time_index=int(time_idx),
                                    object_token=token, object_type=obj.tracked_object_type.name, **details))

    def instrument(method_name, attr, call):
        original = getattr(PDMScorer, method_name)
        tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
        found = 0

        class Recorder(ast.NodeTransformer):
            def visit_Assign(self, node):
                nonlocal found
                if any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Attribute)
                       and t.value.attr == attr for t in node.targets):
                    found += 1
                    return [node, ast.parse(call).body[0]]
                return self.generic_visit(node)

        tree = Recorder().visit(tree)
        assert found == 1, f'Unsupported official implementation: {method_name}, {found} event sites'
        namespace = dict(original.__globals__)
        exec(compile(ast.fix_missing_locations(tree), f'<trace-only:{method_name}>', 'exec'), namespace)
        setattr(DiagnosticScorer, method_name, namespace[method_name])

    instrument('_calculate_no_at_fault_collision', '_collision_time_idcs',
               "self._record_event('NC', proposal_idx, time_idx, token, collision_type=collision_type.name)")
    instrument('_calculate_ttc', '_ttc_time_idcs',
               "self._record_event('TTC', proposal_idx, time_idx, token, projection_horizon_s=float(future_time_idx)*self.proposal_sampling.interval_length, observation_time_index=int(current_time_idx))")
    _API = dict(root=str(root), score=pdm_score, trajectory=Trajectory, sampling=TrajectorySampling,
                simulator=PDMSimulator, scorer=PDMScorer, diagnostic=DiagnosticScorer,
                config=PDMScorerConfig, ego_area=EgoAreaIndex, state=StateIndex)
    return _API


def protocol(config):
    a = api(config)
    # Pin the scoring implementation tree, including tracking simulation.
    root = Path(a['root']) / 'navsim'
    files = sorted((root/'planning/simulation/planner/pdm_planner').rglob('*.py'))
    files += [root/'evaluate/pdm_score.py']
    value = dict(name='NAVSIM_v1.1_PDMS', sampling=config['sampling'], scorer=asdict(a['config']()),
                 candidate_batch='fresh_reference_plus_one_candidate', scale='0..1',
                 source_hashes={str(p):file_hash(p) for p in files})
    value['sha256'] = digest(value)
    return value


def score_one(poses, metric_path, config, diagnostic=True):
    a = api(config)
    poses = poses_valid(poses)
    with lzma.open(metric_path, 'rb') as f:
        cache = pickle.load(f)
    sampling = a['sampling'](**config['sampling'])
    scorer = (a['diagnostic'] if diagnostic else a['scorer'])(sampling, a['config']())
    scorer.events = []
    result = a['score'](cache, a['trajectory'](poses, a['sampling'](num_poses=8, interval_length=.5)),
                        sampling, a['simulator'](sampling), scorer)
    metrics = dict(PDMS=float(result.score), NC=float(result.no_at_fault_collisions),
                   DAC=float(result.drivable_area_compliance), TTC=float(result.time_to_collision_within_bound),
                   EP=float(result.ego_progress), CF=float(result.comfort), DDC=float(result.driving_direction_compliance))
    assert all(np.isfinite(x) and 0 <= x <= 1 for x in metrics.values())
    origin = cache.ego_state.rear_axle
    origin_xy = np.array([origin.x, origin.y]); c, s = np.cos(origin.heading), np.sin(origin.heading)
    simulated = scorer._states[1, :, a['state'].STATE_SE2].copy()
    local_xy = (simulated[:, :2] - origin_xy) @ np.array([[c,-s],[s,c]])
    local = np.column_stack((local_xy, wrap(simulated[:, 2] - origin.heading)))
    events = [dict(e) for e in scorer.events if e['proposal_index'] == 1]
    off = np.flatnonzero(scorer._ego_areas[1, :, a['ego_area'].NON_DRIVABLE_AREA])
    if len(off):
        events.append(dict(kind='DAC', time_index=int(off[0]), proposal_index=1))
    for e in events:
        t = e['time_index'] * sampling.interval_length
        e.update(time_s=float(t), nearest_waypoint_1based=int(np.clip(np.floor(t/.5+.5), 1, 8)),
                 ego_rear_axle_local=local[e['time_index']].tolist(),
                 coordinate_source='official_simulated_rear_axle_not_contact_point',
                 at_initial_state=bool(e['time_index']==0))
    # Keep first event per metric/object; retain distinct actors and TTC projections.
    events.sort(key=lambda e:(e['time_s'], e['kind'], e.get('object_token','')))
    seen=set(); first=[]
    for e in events:
        key=(e['kind'],e.get('object_token'))
        if key not in seen: first.append(e);seen.add(key)
    return dict(metrics=metrics, raw_progress_m=float(scorer._progress_raw[1]),
                reference_raw_progress_m=float(scorer._progress_raw[0]),
                initial_global_rear_axle=[origin.x,origin.y,origin.heading],
                first_collision_time_s=float(scorer.time_to_at_fault_collision(1)) if np.isfinite(scorer.time_to_at_fault_collision(1)) else None,
                first_ttc_violation_time_s=float(scorer.time_to_ttc_infraction(1)) if np.isfinite(scorer.time_to_ttc_infraction(1)) else None,
                events=first, simulated_trajectory=local.tolist(),
                simulation_dt=sampling.interval_length,
                formula='NC * DAC * (5*EP + 5*TTC + 2*CF + 0*DDC) / 12',
                ep_note='EP is normalized centerline progress after multiplying raw progress by NC*DAC.',
                source='official_pdms_fresh_reference_plus_candidate')


def verify_equivalence(poses, metric_path, config):
    original = score_one(poses, metric_path, config, diagnostic=False)
    instrumented = score_one(poses, metric_path, config, diagnostic=True)
    assert original['metrics'] == instrumented['metrics'], 'Trace changed official metrics'
    assert original['simulated_trajectory'] == instrumented['simulated_trajectory'], 'Trace changed simulation'
    assert original['raw_progress_m'] == instrumented['raw_progress_m']
    for kind,key in [('NC','first_collision_time_s'),('TTC','first_ttc_violation_time_s')]:
        times=[e['time_s'] for e in instrumented['events'] if e['kind']==kind]
        assert (min(times) if times else None)==original[key]
    return dict(metrics_exactly_equal=True, simulated_states_exactly_equal=True)
