"""Real NAVSIM input equivalence and strict deployment file access audit."""
import argparse
import builtins
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from starVLA.dataloader.structured_world.current_inputs import CurrentInputs
from starVLA.dataloader.foresight_dataset import current_observation
from tools.structured_world.build_cache import atomic_json


def main():
    parser = argparse.ArgumentParser()
    for name in ('inputs', 'cache', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(1)
    data = CurrentInputs(args.inputs)
    allowed = set()
    for row in data.index:
        record = json.loads((args.inputs/'current'/(row['token']+'.json')).read_text())
        allowed.update(Path(p).resolve() for p in record['image_paths'])
    original_builtin, original_io = builtins.open, io.open
    accessed = set()
    def guard(original):
        def opened(file, *values, **options):
            if isinstance(file, (str, bytes, Path)):
                path = Path(file).resolve()
                if not path.is_relative_to(args.inputs.resolve()) and path not in allowed:
                    raise AssertionError('Non-current deployment file access: '+str(path))
                accessed.add(path)
            return original(file, *values, **options)
        return opened
    observations = []
    with patch('builtins.open', guard(original_builtin)), patch('io.open', guard(original_io)):
        checked = CurrentInputs(args.inputs)
        for i in range(len(checked)): observations.append(checked[i])
    rows = []
    for row, observation in zip(data.index, observations):
        token = row['token']; meta = json.loads((args.cache/'records'/(token+'.json')).read_text())
        previous = current_observation(meta['current_record'])
        np.testing.assert_array_equal(observation['state'], previous['state'])
        if observation['lang'] != previous['lang']: raise AssertionError('Navigation changed')
        for current, old in zip(observation['image'], previous['image']): np.testing.assert_array_equal(current, old)
        with np.load(args.cache/'labels'/(token+'.npz'), allow_pickle=False) as labels:
            rgb = torch.from_numpy(labels['geometry_rgb'].copy()).float()/255.
            normalized = (rgb-rgb.new_tensor([.485, .456, .406])[None, :, None, None])/rgb.new_tensor([.229, .224, .225])[None, :, None, None]
            torch.testing.assert_close(normalized, observation['geometry_images'], rtol=0., atol=0.)
            for key in observation['calibration']:
                torch.testing.assert_close(torch.from_numpy(labels['calibration_'+key]), observation['calibration'][key], rtol=0., atol=0.)
            np.testing.assert_array_equal(labels['geometry_pixel_valid'], observation['geometry_pixel_valid'].numpy())
        rows.append({'token': token, 'all_current_input_fields_bitwise_equal': True})
    atomic_json(args.output, {'scope': 'real current input equivalence, no GT file accessible during reader execution',
        'scenes': len(rows), 'opened_current_files': len(accessed), 'maps_boxes_lidar_GT_labels_opened': 0, 'rows': rows})
    print(json.dumps({'scenes': len(rows), 'bitwise_equal': True, 'non_current_files': 0}), flush=True)


if __name__ == '__main__': main()
