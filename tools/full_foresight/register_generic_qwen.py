"""Verify existing pinned generic Qwen weights offline; never download a model."""
import argparse
import json
from pathlib import Path

from tools.ddpolicy_vehicle.prepare_data import atomic_json
from starVLA.model.modules.vehicle_joint.initialization import verify_generic_source


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--qwen-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Generic source manifest is immutable')
    reference = Path(__file__).resolve().parents[2] / 'reports/ddpolicy_vehicle_from_scratch/GENERIC_SOURCES.json'
    sources = json.loads(reference.read_text())
    qwen_root = str(Path(args.qwen_root).resolve())
    verify_generic_source(qwen_root, sources['qwen'])
    sources['qwen']['root'] = qwen_root
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, sources)
    print(json.dumps({'sources': str(output), 'qwen': sources['qwen']['repository'],
                      'revision': sources['qwen']['revision'], 'downloaded': False,
                      'video_depth_assets_required': False}))


if __name__ == '__main__':
    main()
