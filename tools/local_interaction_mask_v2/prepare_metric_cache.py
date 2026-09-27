"""Build full official metric caches from raw logs in a new, independent directory."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
from tools.local_interaction_mask_v2.score_async import digest, python_tree_digest, atomic_json


def initialize(devkit):
    sys.path.insert(0, devkit)


def cache_log(task):
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    log, tokens, args = task
    try:
        loader = SceneLoader(Path(args['raw_log_root']), None,
            SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True,
                        log_names=[log], tokens=tokens), SensorConfig.build_no_sensors())
        processor = MetricCacheProcessor(str(Path(args['output'])/'cache'), force_feature_computation=False)
    except Exception as error:
        return [{'token':token, 'log':log, 'status':'failed', 'error':repr(error)} for token in tokens]
    rows = []
    for token in tokens:
        row = {'token': token, 'log': log, 'status': 'ok'}
        try:
            scene = loader.get_scene_from_token(token)
            scenario = NavSimScenario(scene, map_root=args['map_root'], map_version='nuplan-maps-v1.0')
            result = processor.compute_metric_cache(scenario)
            path = Path(result.file_name)
            row.update(cache_path=str(path), sha256=digest(path))
        except Exception as error: row.update(status='failed', error=repr(error))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('devkit', 'index', 'raw-log-root', 'map-root', 'output'): parser.add_argument('--'+key, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16: raise ValueError('Bound CPU map/cache workers')
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    index = json.loads(Path(args.index).read_text()); grouped = {}
    if len({row['token'] for row in index}) != len(index): raise ValueError('Duplicate cache population')
    for row in index: grouped.setdefault(row['log'], []).append(row['token'])
    initialize(args.devkit)
    import nuplan
    identity = dict(code_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        arguments=vars(args), index_sha256=digest(args.index), navsim_python_tree_sha256=python_tree_digest(Path(args.devkit)/'navsim'),
        nuplan_python_tree_sha256=python_tree_digest(Path(nuplan.__file__).parent),
        raw_log_sha256={log: digest(Path(args.raw_log_root)/(log+'.pkl')) for log in grouped},
        purpose='offline official metric targets only; never a model input', original_caches_modified=False)
    atomic_json(output/'identity.json', identity)
    rows = []; started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize, initargs=(args.devkit,)) as pool:
        for result in pool.map(cache_log, [(log, tokens, vars(args)) for log, tokens in grouped.items()]):
            rows.extend(result)
            with (output/'progress.jsonl').open('a') as handle:
                for row in result: handle.write(json.dumps(row)+'\n')
            print(json.dumps({'completed':len(rows), 'expected':len(index)}), flush=True)
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with (output/'scenes.csv').open('w') as handle:
        writer = csv.DictWriter(handle, keys); writer.writeheader(); writer.writerows(rows)
    failed = sum(row['status'] != 'ok' for row in rows)
    atomic_json(output/'summary.json', dict(requested=len(index), completed=len(rows), failed=failed,
        logs=len(grouped), wall_seconds=time.monotonic()-started, identity=identity))
    if failed:
        atomic_json(output/'status.json', {'status':'failed', 'scenes':len(rows), 'failed':failed})
        raise RuntimeError('Missing/failed scenes retained; cache population invalid')
    lookup = {row['token']:row for row in rows}
    atomic_json(output/'cache_index.json', [{key:lookup[row['token']][key] for key in ('token','log','cache_path')} for row in index])
    atomic_json(output/'status.json', {'status':'complete', 'scenes':len(rows), 'failed':0})


if __name__ == '__main__': main()
