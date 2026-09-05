#!/usr/bin/env python3
"""Fingerprint actual imports and one immutable logged metric cache, not planned paths."""
import argparse
import hashlib
import inspect
import json
import lzma
from pathlib import Path
import pickle
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fingerprint(value):
    cls = value if isinstance(value, type) else type(value)
    file = Path(inspect.getfile(cls))
    return dict(python_class=f'{cls.__module__}.{cls.__qualname__}', file=str(file),
                sha256=hashlib.sha256(file.read_bytes()).hexdigest())


def run(args):
    from navsim.agents.EpisodeDrive.score_module import compute_navsim_score as runtime
    from navsim.agents.EpisodeDrive.layers.world_model.minimal_physical_targets import logged_occupancy_frames
    if args.output.exists():
        raise FileExistsError(args.output)
    with lzma.open(args.metric_cache, 'rb') as stream:
        cache = pickle.load(stream)
    observation = cache.observation
    frames = logged_occupancy_frames(observation)
    aggregate = inspect.getsource(type(runtime.scorer)._aggregate_scores)
    ttc = inspect.getsource(type(runtime.scorer)._calculate_ttc)
    reference_url = ('https://raw.githubusercontent.com/autonomousvision/navsim/'
                     '3e8291bfa89ff247231e0227778840cd0a036896/'
                     'navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py')
    reference = urllib.request.urlopen(reference_url, timeout=30).read()
    files = ['navsim/agents/EpisodeDrive/score_module/scorer.py',
             'navsim/agents/EpisodeDrive/transformer_decoder.py',
             'navsim/agents/EpisodeDrive/layers/losses/episode_drive_loss.py',
             'navsim/agents/EpisodeDrive/score_module/compute_navsim_score.py',
             'navsim/agents/EpisodeDrive/score_module/train_pdm_scorer.py']
    core = {}
    for file in files:
        base = subprocess.check_output(['git', 'show', f'e85e1a1797f1a26303e9ee81d9f3d1231bc59978:{file}'], cwd=ROOT)
        actual = (ROOT/file).read_bytes()
        core[file] = dict(sha256=hashlib.sha256(actual).hexdigest(), unchanged_from_v1p1=actual == base)
    report = dict(
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        metric_cache=str(args.metric_cache.resolve()),
        metric_cache_sha256=hashlib.sha256(args.metric_cache.read_bytes()).hexdigest(),
        runtime={name: fingerprint(value) for name, value in dict(simulator=runtime.simulator,
            scorer=runtime.scorer, official_scorer=runtime.OfficialPDMScorer, metric_cache=cache,
            observation=observation, centerline=cache.centerline, drivable_map=cache.drivable_area_map).items()},
        occupancy_frames=len(observation._occupancy_maps),
        observation_sample_res=observation._observation_sample_res,
        observation_sample_interval=observation._sample_interval,
        coverage_4p9=frames[49] is not None,
        ttc_source_time_max_sec=4.0 if 'range(self.proposal_sampling.num_poses + 1)' in ttc else None,
        ttc_max_lag_sec=.9, ttc_required_environment_end_sec=4.9,
        custom_ddc_multiplier_detected=('DRIVING_DIRECTION' in aggregate),
        training_ddc_weight=runtime.config.driving_direction_weight,
        aggregate_source=aggregate,
        note='Runtime protocol only; external custom DDC evaluators are not silently imported or mixed with Lite labels.',
        upstream_navsim=dict(url=reference_url, sha256=hashlib.sha256(reference).hexdigest()),
        scorer_source_commit='valeoai/DrivoR@fc6e5aa144bbcb5a046e22c18f1bd5cf3af8634a',
        scorer_core_files=core,
        label_protocol='projected_geometric_gap != official TTC/NC; road_margin != DAC; route_progress_m != normalized EP',
        actor_provenance_basis='Local train_cache_processor._interpolate_gt_observation: logged objects, 5 seconds, interpolated at 10 Hz; audited train cache path.',
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metric-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
