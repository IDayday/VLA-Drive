"""New clip-model cost accounting; actual shared windows, no historical speed reuse."""
import argparse
import json
from pathlib import Path
from tools.full_foresight.summarize_profiles import summarize, stable_steps, concurrent_rate
from tools.ddpolicy_vehicle.prepare_data import atomic_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--campaign-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.campaign_root)
    population = json.loads((root/'clip_index_v1/identity.json').read_text())
    scene_count = json.loads((root/'clip_index_v1/COMPLETE.json').read_text())['coverage']['train']['scenes']
    profiles = []
    for run in sorted((root/'students').glob('profile_*')):
        if not (run/'identity.json').exists():
            continue
        identity = json.loads((run/'identity.json').read_text())
        if identity['data']['index_sha256'] != population['index_hashes']['train']:
            raise ValueError('Profile uses a different training population')
        profiles.append(summarize(run, scene_count))
    paired = [root/'students'/name for name in ('profile_S0_paired4_v1', 'profile_S4_paired4_v1')]
    concurrent = concurrent_rate([[dict(row, batch_scenes=32) for row in stable_steps(run)] for run in paired])
    atomic_json(args.output, {'profiles': profiles, 'paired4': concurrent,
        'selection_basis': 'Measured speed and capacity, never profile planning quality',
        'precision': 'Live BF16 timing is separate from FP32-master deployment timing',
        'cost_scope': 'Optimizer speed estimates; all loading, saving and failed attempts are also metered'})


if __name__ == '__main__':
    main()
