"""Isolated official NAVSIM releases for evaluation only.

No scorer formula is reimplemented. Each process imports exactly one upstream
tree and calls its unmodified cache processor / pdm_score. In particular v1.0
must not use a v1.1 PDM reference cache, or v1.1's DDC readout.
"""
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import lzma
import os
from pathlib import Path
import pickle
import sys
import time
import uuid

import numpy as np

from starVLA.rl.flow_grpo.transactions import atomic_json, publication_lock
from starVLA.rl.fm_transfer.contracts import require_dispatch
from starVLA.rl.safe_flow_improvement.contracts import file_sha, fingerprint

RELEASES = {
    'v1.0': ('9f3ea68ffda0c6afe37b51b402f3245ea9936e18',
             '93294ead28bb4f079c2d96af40cc5724538bbc1da78b4bf16092641f7f91297e'),
    'v1.1': ('0811876c274e8b058ab2be9b3dcd4d37bd23f177',
             '02ab075017d4edb1c8bba6476ef1674b84d43dbca0f599df87cfe052b2dd5211'),
}
SCORER = 'navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py'
SPEC = None


def upstream_identity(source, version):
    source = Path(source).resolve()
    receipt = json.loads((source / 'SOURCE_COMPLETE.json').read_text())
    commit, scorer_sha = RELEASES[version]
    if receipt['commit'] != commit or receipt['tag'] != version or file_sha(source / SCORER) != scorer_sha:
        raise ValueError('Wrong official scoring release')
    sources = {str(p.relative_to(source)): file_sha(p)
               for p in sorted((source / 'navsim').rglob('*.py'))}
    if sources != {k: v for k, v in receipt['files'].items() if k.startswith('navsim/') and k.endswith('.py')}:
        raise ValueError('Official NAVSIM Python source changed')
    value = dict(version=version, commit=commit, sources=sources,
                 dependencies={k: importlib.metadata.version(k) for k in ('numpy', 'scipy', 'shapely')},
                 request_sampling=dict(poses=8, interval_seconds=.5),
                 simulation_sampling=dict(poses=40, interval_seconds=.1),
                 proposal_pool=['version_specific_fixed_PDM_reference', 'one_requested_candidate'],
                 score_units='0..1', source_modifications=[])
    value['sha256'] = fingerprint(value)
    return value


def initialize(spec):
    global SPEC
    require_dispatch()
    source = Path(spec['source']).resolve()
    if upstream_identity(source, spec['version']) != spec['protocol']:
        raise ValueError('Scoring source/environment changed')
    if any(k == 'navsim' or k.startswith('navsim.') for k in sys.modules):
        raise RuntimeError('Use a fresh isolated process, never mix NAVSIM releases')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['NUPLAN_MAPS_ROOT'] = spec['maps_root']
    os.environ['NUPLAN_MAP_VERSION'] = 'nuplan-maps-v1.0'
    sys.path.insert(0, str(source))
    import navsim
    if not Path(navsim.__file__).resolve().is_relative_to(source):
        raise RuntimeError('Wrong NAVSIM import path')
    SPEC = spec


def cache_path(token, log):
    return Path(SPEC['cache_root']) / log / 'unknown' / token / 'metric_cache.pkl'


def validate_cache(path, token, log):
    seal = json.loads(path.with_suffix('.json').read_text())
    if (seal['token'] != token or seal['log'] != log or
            seal['cache_identity'] != SPEC['cache_identity'] or
            seal['sha256'] != file_sha(path)):
        raise ValueError('Incomplete, mixed-version or changed metric cache')
    return seal


def build_cache_log(job):
    """One log owner; atomic per-scene publication and explicit retry accounting."""
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    require_dispatch()
    log, tokens = job
    root = Path(SPEC['cache_root'])
    started = time.monotonic()
    new = reused = 0
    entries = []
    with publication_lock(root / '.locks' / (fingerprint(log) + '.lock')):
        loader = SceneLoader(Path(SPEC['raw_logs']), None,
            SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1,
                        log_names=[log], tokens=tokens), SensorConfig.build_no_sensors())
        if set(loader.tokens) != set(tokens):
            raise ValueError('Raw scene membership differs from fixed Navtest')
        # Keep unsealed attempts as evidence. The original processor writes its
        # staging filename into the pickle; that inert field is never rewritten.
        stage = root / '.building' / uuid.uuid4().hex
        processor = MetricCacheProcessor(str(stage), False)
        raw_sha = file_sha(Path(SPEC['raw_logs']) / (log + '.pkl'))
        for token in tokens:
            path = cache_path(token, log)
            if path.with_suffix('.json').exists():
                seal = validate_cache(path, token, log)
                if seal['raw_log_sha256'] != raw_sha:
                    raise ValueError('Raw log changed')
                reused += 1
            else:
                require_dispatch()
                scenario = NavSimScenario(loader.get_scene_from_token(token),
                    map_root=SPEC['maps_root'], map_version='nuplan-maps-v1.0')
                entry = processor.compute_metric_cache(scenario)
                temporary = Path(entry.file_name)
                seal = dict(token=token, log=log, cache_identity=SPEC['cache_identity'],
                    protocol_sha256=SPEC['protocol']['sha256'], raw_log_sha256=raw_sha,
                    sha256=file_sha(temporary), bytes=temporary.stat().st_size)
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    # A crash before publishing the seal is not a reusable cache.
                    path.rename(path.with_name(path.name + '.unsealed-' + uuid.uuid4().hex))
                os.replace(temporary, path)
                atomic_json(path.with_suffix('.json'), seal)
                new += 1
            entries.append(seal)
    return dict(log=log, entries=entries, new=new, reused=reused,
                worker_seconds=time.monotonic() - started)


def score_requests(job):
    """Every request gets its own [PDM reference, candidate] simulation/score."""
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    require_dispatch()
    token, log, requests = job
    path = cache_path(token, log)
    seal = validate_cache(path, token, log)
    with lzma.open(path, 'rb') as stream:
        cache = pickle.load(stream)
    sampling = TrajectorySampling(num_poses=40, interval_length=.1)
    request_sampling = TrajectorySampling(num_poses=8, interval_length=.5)
    rows = []
    for requested in requests:
        requested = np.asarray(requested)
        if requested.shape != (8, 3) or not np.isfinite(requested).all():
            raise ValueError('Invalid requested trajectory; do not return reward zero')
        start = time.monotonic()
        simulator, scorer = PDMSimulator(sampling), PDMScorer(sampling)
        metrics = asdict(pdm_score(cache, Trajectory(requested, request_sampling), sampling, simulator, scorer))
        if not all(np.isfinite(v) and 0 <= v <= 1 for v in metrics.values()):
            raise ValueError('Invalid official result')
        rows.append(dict(metrics=metrics, protocol_sha256=SPEC['protocol']['sha256'],
            version=SPEC['version'], status='OK', cache_sha256=seal['sha256'],
            request_sha256=hashlib.sha256(np.ascontiguousarray(requested).tobytes()).hexdigest(),
            raw_progress_meters=float(scorer._progress_raw[1]),
            simulated_states=scorer._states[1].copy(),
            scoring_seconds=time.monotonic() - start))
    return rows
