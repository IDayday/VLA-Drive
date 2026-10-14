"""Verify the specified original-GT S0 endpoint and existing input/label assets."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.decision_local.common import atomic_json, digest, identity, read, source_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = Path('/mnt/project/VLA-Drive-action-video-source-1493ded')
    sha = '1493deda247107efe076b9294863a6753391298a'
    if source_sha(source) != sha or subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=source).strip():
        raise ValueError('Pinned S runtime changed')
    archive = ROOT/'reports/experiment_results/20261014/evidence/C_S_COMPLETED_RUNS.json'
    historical = next(v for v in read(archive)['runs'].values() if v['run_id'] == 'formal_S0_seed42_full100k_v2')
    run = Path('/mnt/project/action-video-foresight-artifacts/20261002/students/formal_S0_seed42_full100k_v2')
    training = read(run/'identity.json')
    checkpoint = historical['checkpoint']
    folder = run/'checkpoints'/checkpoint['tag']
    complete = read(folder/'COMPLETE.json')
    if training != historical['identity'] or complete['completed'] != 100000 or training['source_sha'] != sha:
        raise ValueError('Original-GT S0 endpoint identity mismatch')
    if complete['identity'] != training['identity'] or set(p.name for p in folder.glob('*.pt')) != set(checkpoint['files']):
        raise ValueError('Incomplete original native checkpoint')
    started = time.monotonic()
    for name, expected in checkpoint['files'].items():
        if digest(folder/name) != expected:
            raise ValueError('Original checkpoint hash mismatch: '+name)
        print('verified', name, flush=True)
    current = {'train': '/mnt/project/ddp-foresight-artifacts/20260928/student_train_v1',
               'dev': '/mnt/project/ddp-foresight-artifacts/20260928/student_dev_v1',
               'navtest': '/mnt/project/ddpolicy-vehicle-joint-artifacts/20260928/current_navtest_v1'}
    base = Path('/mnt/project/structured-world-fgtr-round1-artifacts/20261008')
    labels = {'train': str(base/'navsim_train_full_label_v5_centered'),
              'dev': str(base/'navsim_dev_label_v1_centered')}
    populations = {}
    for split, root in current.items():
        root = Path(root)
        meta, index = read(root/'identity.json'), read(root/'index.json')
        if meta['split'] != split or len(index) != {'train': 101592, 'dev': 1696, 'navtest': 12146}[split]:
            raise ValueError('Population changed: '+split)
        if split == 'train' and meta != training['data']:
            raise ValueError('Anchor and new training population differ')
        populations[split] = {'current_root': str(root), 'identity': meta, 'index_sha256': digest(root/'index.json'),
                              'scenes': len(index), 'logs': len({r['log'] for r in index})}
        if split in labels:
            label = Path(labels[split])
            li, lc = read(label/'identity.json'), read(label/'COMPLETE.json')
            if li['source_current_identity'] != meta or lc['identity'] != li['identity']:
                raise ValueError('Foreign geometry labels')
            if read(label/'index.json') != index:
                raise ValueError('Geometry population/order differs')
            populations[split].update(label_root=str(label), label_identity=li['identity'],
                                      label_identity_file_sha256=digest(label/'identity.json'))
    fixed = Path('/mnt/project/action-video-foresight-artifacts/20261002/C1_100k_representations_v1/train_representation_queries.json')
    selected = read(fixed)
    allowed = {r['token'] for r in read(Path(current['train'])/'index.json')}
    if len(selected) != 4096 or len({r['token'] for r in selected}) != 4096 or not all(r['token'] in allowed for r in selected):
        raise ValueError('Existing fixed training diagnostic subset invalid')
    declaration = {'schema': 'decision_local_frozen_anchor_v1', 'source_root': str(source),
        'training_source_sha': sha, 'run': str(run), 'tag': checkpoint['tag'], 'checkpoint': checkpoint,
        'training_identity': training['identity'], 'data': populations,
        'diagnostic_train_index': str(fixed), 'diagnostic_train_index_sha256': digest(fixed),
        'base_training_GPU_hours': sum(m['gpu_hours'] for m in historical['meters']),
        'shared_base_training_seed': 42, 'registered_anchor': 'original_GT_S0_common_100000_not_reselected',
        'precision': 'FP32 native optimizer master reconstruction, strict=True, eval, TF32 off',
        'proposal_seeds_train': [42, 43], 'primary_sampling_seed': 42,
        'historical_receipt_sha256': digest(archive), 'registration_source_sha': source_sha(ROOT)}
    declaration['identity'] = identity(declaration)
    declaration['verification_seconds'] = time.monotonic()-started
    if args.output.exists():
        raise FileExistsError('Preserve the original registration')
    atomic_json(args.output, declaration)
    print(json.dumps({'status': 'PASS', 'anchor_identity': declaration['identity'],
                      'checkpoint_sha256': checkpoint['sha256'], 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
