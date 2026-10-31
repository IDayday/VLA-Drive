"""Canonical v1.1 isolated CPU workers; the old v2 RewardService is untouched."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import fcntl
import importlib.metadata
import json
import lzma
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np

from .contracts import V1_PROTOCOL, file_sha, fingerprint

SPEC = None


def protocol_identity(repo):
    repo = Path(repo).resolve()
    root = repo / 'navsim_v1.1/navsim'
    sources = {str(p.relative_to(root)): file_sha(p) for p in sorted((root / 'navsim').rglob('*.py'))}
    result = {'name': V1_PROTOCOL, 'sources': sources,
              'proposal_pool': ['fixed_official_pdm_reference', 'one_requested_candidate'],
              'request_sampling': {'poses': 8, 'interval_seconds': .5},
              'simulation_sampling': {'poses': 40, 'interval_seconds': .1},
              'score_units': '0..1', 'missing_two_frame_comfort': 'NOT_APPLICABLE_TO_V1',
              'dependencies': {k: importlib.metadata.version(k) for k in ('numpy', 'scipy', 'shapely')}}
    result['sha256'] = fingerprint(result)
    return result


def initialize(spec):
    global SPEC
    SPEC = spec
    # Compare real paths on both sides when an immutable runtime links its
    # unchanged scorer tree to a frozen base.
    root = (Path(spec['repo']).resolve() / 'navsim_v1.1/navsim').resolve()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['NUPLAN_MAPS_ROOT'] = spec['maps_root']
    os.environ['NUPLAN_MAP_VERSION'] = 'nuplan-maps-v1.0'
    sys.path.insert(0, str(root))
    import navsim
    if not Path(navsim.__file__).resolve().is_relative_to(root):
        raise RuntimeError('v1/v2 module collision; refusing to score in a mixed interpreter')
    if protocol_identity(spec['repo'])['sha256'] != spec['protocol_sha256']:
        raise RuntimeError('Canonical scorer source/environment identity changed')


def load_cache(token, log):
    from navsim.planning.metric_caching.metric_cache import MetricCache
    for root in SPEC['cache_roots']:
        path = Path(root) / log / 'unknown' / token / 'metric_cache.pkl'
        if path.exists():
            break
    else:
        raise FileNotFoundError(f'Canonical v1 cache missing: {log}/{token}; build it explicitly, never substitute v2')
    with lzma.open(path, 'rb') as stream:
        cache = pickle.load(stream)
    expected = {'file_path', 'trajectory', 'ego_state', 'observation', 'centerline', 'route_lane_ids', 'drivable_area_map'}
    if type(cache) is not MetricCache or set(vars(cache)) != expected:
        raise ValueError('Cache is not the locked v1 dataclass (v2 conversion is prohibited)')
    return cache, {'path': str(path), 'sha256': file_sha(path)}


def build_log_cache(job):
    """Reuse the official processor from real logged scenes, protected per log."""
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    log, tokens = job
    root = Path(SPEC['build_cache_root'])
    locks = root / '.locks'
    locks.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (locks / (fingerprint(log) + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        loader = SceneLoader(Path(SPEC['raw_logs']), None,
                             SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1,
                                         log_names=[log], tokens=tokens), SensorConfig.build_no_sensors())
        if set(loader.tokens) != set(tokens):
            raise ValueError(f'Raw scene population differs for {log}')
        processor = MetricCacheProcessor(str(root), False)
        entries = []
        for token in tokens:
            scene = loader.get_scene_from_token(token)
            scenario = NavSimScenario(scene, map_root=SPEC['maps_root'], map_version='nuplan-maps-v1.0')
            entry = processor.compute_metric_cache(scenario)
            path = Path(entry.file_name)
            entries.append({'token': token, 'log': log, 'path': str(path), 'sha256': file_sha(path)})
    return {'log': log, 'entries': entries, 'seconds': time.monotonic() - started,
            'protocol_sha256': SPEC['protocol_sha256']}


def score_scene(job):
    """Batch the execution only: call official pdm_score once PER candidate."""
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import EgoAreaIndex
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    token, log, requests = job
    cache, identity = load_cache(token, log)
    simulation_sampling = TrajectorySampling(num_poses=40, interval_length=.1)
    request_sampling = TrajectorySampling(num_poses=8, interval_length=.5)
    simulator, scorer = PDMSimulator(simulation_sampling), PDMScorer(simulation_sampling)
    records = []
    for requested in requests:
        requested = np.asarray(requested)
        if requested.shape != (8, 3) or not np.isfinite(requested).all():
            raise ValueError('Invalid request; never convert infrastructure/shape errors to reward zero')
        start = time.monotonic()
        feedback = asdict(pdm_score(cache, Trajectory(requested, request_sampling), simulation_sampling, simulator, scorer))
        if not all(np.isfinite(x) and 0 <= x <= 1 for x in feedback.values()):
            raise ValueError('Nonfinite/out-of-range canonical v1 feedback')
        def event(value):
            return float(value) if np.isfinite(value) else None
        records.append({'token': token, 'log': log, 'status': 'OK', 'protocol': V1_PROTOCOL,
                        'protocol_sha256': SPEC['protocol_sha256'], 'cache': identity, 'metrics': feedback,
                        'requested_trajectory': requested.copy(),
                        'simulated_states': scorer._states[1].copy(),
                        'simulated_body_coordinates': scorer._ego_coords[1].copy(),
                        'non_drivable_body_by_time': scorer._ego_areas[1, :, EgoAreaIndex.NON_DRIVABLE_AREA].copy(),
                        'official_first_at_fault_collision_seconds': event(scorer.time_to_at_fault_collision(1)),
                        'official_first_ttc_infraction_seconds': event(scorer.time_to_ttc_infraction(1)),
                        'raw_progress_meters': float(scorer._progress_raw[1]),
                        'geometric_margins_status': 'NOT_YET_EXTRACTED_NOT_OFFICIAL_RISK_PROBABILITY',
                        'scoring_seconds': time.monotonic() - start})
    return records


class V1RewardService:
    def __init__(self, spec, workers):
        self.spec = dict(spec)
        self.spec['protocol_sha256'] = protocol_identity(spec['repo'])['sha256']
        self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                        initializer=initialize, initargs=(self.spec,))

    def score(self, jobs):
        # Exceptions retain the scene denominator and abort the affected update.
        return list(self.pool.map(score_scene, jobs, chunksize=1))

    def build(self, jobs):
        return self.pool.map(build_log_cache, jobs, chunksize=1)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
