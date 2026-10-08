"""Export input-only RGB/calibration/state; all GT stays in its original cache."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.structured_world.build_cache import atomic_json, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--navsim-image-root', type=Path)
    parser.add_argument('--debug-limit', type=int, default=0)
    args = parser.parse_args()
    source = json.loads((args.cache/'identity.json').read_text())
    done = json.loads((args.cache/'COMPLETE.json').read_text())
    if done['identity'] != source['identity'] or done['scenes'] != done['expected_scenes']:
        raise ValueError('Incomplete source input population')
    rows = json.loads((args.cache/'index.json').read_text())
    if args.debug_limit: rows = rows[:args.debug_limit]
    declaration = {'schema': 'structured_world_current_inputs_v2', 'dataset': source['dataset'],
        'cameras': 3 if source['dataset'] == 'navsim' else 6, 'source_cache_identity': source['identity'],
        'population_kind': 'debug_only' if args.debug_limit else source['population_kind'], 'index': rows,
        'exporter_sha256': digest(Path(__file__)), 'Qwen_image_hashes_verified': True,
        'inputs': 'current RGB, local calibration, legal past/current state, fixed navigation text'}
    identity = hashlib.sha256(json.dumps(declaration, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'identity.json').exists() and json.loads((args.output/'identity.json').read_text())['identity'] != identity:
        raise ValueError('Foreign current-only export')
    atomic_json(args.output/'identity.json', {'identity': identity, **declaration})
    atomic_json(args.output/'index.json', rows)
    for name in ('current', 'pixels'): (args.output/name).mkdir(exist_ok=True)
    for row in rows:
        token = row['token']; metadata = json.loads((args.cache/'records'/(token+'.json')).read_text())
        current = metadata['current_record']
        paths = current['image_paths']
        if source['dataset'] == 'navsim':
            from starVLA.dataloader.foresight_dataset import current_observation
            observation = current_observation(current, args.navsim_image_root)
            state, lang = observation['state'], observation['lang']
            if args.navsim_image_root:
                paths = [str(args.navsim_image_root/(hashlib.sha256(name.encode()).hexdigest()+'.jpg')) for name in paths]
        else:
            state, lang = current['state'], current['lang']
        source_file = args.cache/'labels'/(token+'.npz')
        if digest(source_file) != metadata['label_sha256']: raise ValueError('Changed input source container')
        with np.load(source_file, allow_pickle=False) as arrays:
            keys = ['geometry_rgb', 'geometry_pixel_valid']+['calibration_'+key for key in
                     ('sensor2ego', 'intrinsics', 'post_rots', 'post_trans')]
            inputs = {key: arrays[key].copy() for key in keys}
        target = args.output/'pixels'/(token+'.npz'); temporary = target.with_suffix('.tmp')
        with temporary.open('wb') as stream: np.savez_compressed(stream, **inputs)
        temporary.replace(target)
        atomic_json(args.output/'current'/(token+'.json'), {'identity': identity, 'token': token,
            'image_paths': paths, 'image_sha256': [digest(Path(p)) for p in paths],
            'state': np.asarray(state).tolist(), 'lang': lang, 'pixels_sha256': digest(target)})
    atomic_json(args.output/'COMPLETE.json', {'identity': identity, 'scenes': len(rows), 'contains_GT': False})
    print(json.dumps({'identity': identity, 'scenes': len(rows), 'contains_GT': False}), flush=True)


if __name__ == '__main__': main()
