"""Prepare optimized labels and launch the registered S3 recipe (CPU preflight first)."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from .labels import digest, identity_hash, load_overlay, prepare, read, training_population, write

SOURCE = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = '/mnt/project/s3-optimized-training-artifacts'
DEFAULT_DATA = '/mnt/project/ddp-foresight-artifacts/20260928/student_train_v1'
DEFAULT_CAMPAIGN = '/mnt/project/DriveVLA-M0/outputs/robust_navtrain_campaign_20261003_simscale_v1'


def baseline(variant='s3'):
    family = 's0' if variant == 's0' else 's3'
    cfg = read(SOURCE / f'configs/{family}_optimized/baseline_config.json')
    reg = read(SOURCE / f'configs/{family}_optimized/baseline_registration.json')
    if reg['arm'] != family.upper() or identity_hash(cfg) != reg['config_sha256']:
        raise ValueError('Frozen baseline configuration changed')
    if variant == 'action-only':
        # R is the existing native DDP path: no W tokens or auxiliary modules.
        # Keep the original Qwen/history/DiT recipe and public initialization.
        cfg['framework']['name'] = 'DDPForesight'
        cfg['foresight'] = {'arm': 'R', 'num_queries': 0, 'lambda_vis': 0., 'lambda_int': 0.,
                            'enable_current_dino': False, 'enable_future_dino': False,
                            'enable_interaction': False, 'gradient_checkpointing': True}
        reg = {**reg, 'arm': 'DDP_ACTION_ONLY', 'config_sha256': identity_hash(cfg),
               'dino': None, 'interaction': None, 'future_clip': None,
               'definition': 'native DDP Qwen + history state + original DiT; zero W tokens; ego FM loss only'}
    elif variant not in ('s3', 's0'):
        raise ValueError('Unknown training variant')
    return cfg, reg


def apply_objective(cfg, path):
    """Validate an opt-in recipe before allocating a model or registering a run."""
    if not path:
        return cfg
    from dataclasses import asdict
    from starVLA.model.modules.foresight.trajectory_objective import TrajectoryObjectiveConfig
    import copy
    objective = TrajectoryObjectiveConfig.from_mapping(read(path))
    if objective.risk_overshoot_weight:
        raise ValueError('This dataset has no audited ego_risk_mask cache; risk loss is an API for a future identified cache')
    cfg = copy.deepcopy(cfg)
    cfg['trajectory_objective'] = asdict(objective)
    # JSON canonicalization keeps resume equality stable for tuple-valued fields.
    cfg['trajectory_objective'] = json.loads(json.dumps(cfg['trajectory_objective']))
    return cfg


def preflight(a):
    """Load metadata and label hashes before torchrun/CUDA; no model allocation."""
    from starVLA.dataloader.action_video_foresight_dataset import ActionVideoForesightDataset
    from tools.foresight.student_state import validate_rank_batches
    variant = getattr(a, 'variant', 's3')
    cfg, reference = baseline(variant)
    cfg = apply_objective(cfg, getattr(a, 'objective_config', None))
    current, ego, index = training_population(a.data)
    if len(index) != reference['scene_count'] or identity_hash(index) != reference['scene_index']:
        raise ValueError('S3 train/dev population changed')
    overlay, _, _ = load_overlay(a.labels, current, ego, index)
    f = cfg['foresight']
    if variant in ('s3', 's0'):
        dataset = ActionVideoForesightDataset(
            a.data, ego_relabel_root=a.labels, candidate=f['candidate'], current=True, future=True,
            dino_root=a.dino_root, dino_index=a.dino_index, expected_dino=reference['dino'],
            interaction_root=a.interaction_root, expected_interaction=reference['interaction'],
            clip_root=a.clip_root, expected_clip=reference['future_clip'],
            future_type=f['future_target_type'], expected_teacher=f['video_teacher_identity'],
            allow_partial=False, image_root=a.local_image_root)
    else:
        from starVLA.dataloader.foresight_dataset import ForesightTrainingDataset
        dataset = ForesightTrainingDataset(a.data, ego_relabel_root=a.labels)
        dataset.local_image_root = a.local_image_root
        if a.local_image_root:
            stage = read(Path(a.data) / 'local_stage.json')
            if (stage['source_identity'] != current['identity'] or stage['scenes'] != len(index)
                or Path(stage['image_root']) != Path(a.local_image_root)):
                raise ValueError('Incomplete or foreign local image staging')
    validate_rank_batches(len(dataset), reference['global_batch'], a.gpus, a.micro_batch)
    # Materialize actual current images + every target type for a few scenes.
    # This catches missing staged data and schema incompatibilities before GPUs.
    for i in sorted({0, len(dataset) // 2, len(dataset) - 1}):
        observation, targets = dataset[i]
        if observation['token'] != index[i]['token'] or 'ego_video' not in targets:
            raise ValueError('Planning/video label routing failed')
        if variant == 'action-only' and set(targets) != {'ego', 'ego_video'}:
            raise ValueError('Native action-only may load only ego planning labels')
    return cfg, reference, overlay, {
        'scene_count': len(dataset), 'optimized_accepted': overlay['accepted_scenes'],
        'original_fallback': overlay['fallback_scenes'], 'excluded_source_scenes': overlay['excluded_source_scenes'],
        'ego_relabel_identity': overlay['identity'], 'variant': variant,
        'auxiliary_targets_unchanged': variant in ('s3', 's0'),
        'future_target_type': f.get('future_target_type', 'disabled'),
        'video_condition': 'original_logged_ego' if variant == 's3' else 'disabled',
        'initialization': reference['initialization'],
        **({'trajectory_objective': cfg['trajectory_objective']} if 'trajectory_objective' in cfg else {}),
        'sampled_dataset_items': 3}


def command(a, reference, config_path, registration_path):
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node', str(a.gpus),
           '--master_port', str(a.master_port), '-m', 'tools.foresight.train_student']
    values = {'config': config_path, 'data': a.data, 'ego-relabel-root': a.labels,
              'campaign-root': a.output, 'run-id': a.run_id, 'global-batch': reference['global_batch'],
              'micro-batch': a.micro_batch, 'updates': reference['updates'],
              'schedule-updates': reference['schedule_updates'], 'warmup': 5000,
              'save-every': 200, 'milestones': ','.join(map(str, reference['milestones'])),
              'loader-workers': a.loader_workers, 'scope': 'formal', 'registration': registration_path}
    if reference['arm'] in ('S3', 'S0'):
        values.update({'dino-root': a.dino_root, 'dino-index': a.dino_index, 'dino-identity': reference['dino'],
                       'interaction-root': a.interaction_root, 'interaction-identity': reference['interaction'],
                       'clip-root': a.clip_root, 'clip-identity': reference['future_clip']})
    if a.local_image_root:
        values['local-image-root'] = a.local_image_root
    for key, value in values.items():
        cmd += ['--' + key, str(value)]
    if a.resume:
        cmd += ['--resume', '--acknowledge-stop']
    if getattr(a, 'stop_after', 0):
        cmd += ['--stop-after', str(a.stop_after)]
    return cmd


def registration(a, cfg, reference, overlay, source):
    return {**reference, 'run_id': a.run_id, 'training_source_sha': source,
            'world_size': a.gpus, 'micro_batch': a.micro_batch, 'local_image_root': a.local_image_root,
            'config_sha256': identity_hash(cfg), 'baseline_training_source_sha': reference['training_source_sha'],
            'baseline_run_id': reference['run_id'], 'ego_relabel_identity': overlay['identity'],
            'ego_relabel': overlay,
            **({'trajectory_objective': cfg['trajectory_objective'],
                'definition': 'baseline architecture with explicitly registered opt-in ego fitting objectives',
                'baseline_definition': reference.get('definition')}
               if 'trajectory_objective' in cfg else {}),
            'video_condition': 'original_logged_ego' if reference['arm'] == 'S3' else 'disabled'}


def freeze(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError('Frozen run registration/config changed: ' + str(path))
    else:
        write(path, value)


def launch(a):
    if not a.run_id or Path(a.run_id).name != a.run_id or a.run_id in ('.', '..'):
        raise ValueError('run-id must be a single directory name')
    cfg, reference, overlay, report = preflight(a)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=SOURCE).strip():
        raise ValueError('Training requires a clean, committed source snapshot')
    root = Path(a.output).resolve()
    config_path = root / 'registrations' / (a.run_id + '_config.json')
    registration_path = root / 'registrations' / (a.run_id + '.json')
    reg = registration(a, cfg, reference, overlay, source)
    cmd = command(a, reference, config_path, registration_path)
    report.update(command=shlex.join(cmd), source_sha=source, run_id=a.run_id)
    print(json.dumps(report, indent=2), flush=True)
    if a.dry_run:
        return
    root.mkdir(parents=True, exist_ok=True)
    # Keep a host-independent filesystem lock while the launcher and ranks live.
    lock = (root / (a.run_id + '.launch.lock')).open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run = root / 'students' / a.run_id
    if run.exists():
        if not a.resume:
            raise FileExistsError('Run exists; use a new run-id or --resume for this same data version')
        state = read(run / 'status.json')
        if state['status'] == 'COMPLETE':
            raise ValueError('Completed run cannot be resumed')
        previous = read(run / 'identity.json')
        if previous.get('ego_relabel') != overlay or previous.get('source_sha') != source:
            raise ValueError('Resume cannot change optimized labels or training source')
    elif a.resume:
        raise ValueError('No interrupted run to resume')
    config_path.parent.mkdir(parents=True, exist_ok=True)
    freeze(config_path, cfg)
    freeze(registration_path, reg)
    write(root / (a.run_id + '_preflight.json'), report)
    env = dict(os.environ)
    env.setdefault('OMP_NUM_THREADS', '2')
    env.setdefault('TOKENIZERS_PARALLELISM', 'false')
    # exec preserves terminal signals. The lock descriptor remains open across
    # exec so a second invocation cannot start duplicate ranks.
    os.set_inheritable(lock.fileno(), True)
    os.chdir(SOURCE)
    os.execvpe(cmd[0], cmd, env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('prepare', help='CPU only: validate release and snapshot train labels')
    p.add_argument('--campaign', default=DEFAULT_CAMPAIGN)
    p.add_argument('--data', default=DEFAULT_DATA)
    p.add_argument('--labels', default=DEFAULT_OUTPUT + '/labels_v1')
    for name in ('check', 'train'):
        p = sub.add_parser(name, help='CPU data preflight' if name == 'check' else 'Launch S3; --dry-run checks without training')
        p.add_argument('--data', default=DEFAULT_DATA)
        p.add_argument('--labels', default=DEFAULT_OUTPUT + '/labels_v1')
        p.add_argument('--dino-root', default='/mnt/project/ddp-full-foresight-study-artifacts/20260929/dino_targets_v1/C1')
        p.add_argument('--dino-index', default='/mnt/project/ddp-full-foresight-study-artifacts/20260929/dino_index_v1')
        p.add_argument('--interaction-root', default='/mnt/project/ddp-foresight-artifacts/20260928/interaction_train_v1')
        p.add_argument('--clip-root', default='/mnt/project/action-video-foresight-artifacts/20261002/targets_video_clip_train_v1')
        p.add_argument('--local-image-root')
        p.add_argument('--gpus', type=int, default=8)
        p.add_argument('--micro-batch', type=int, default=4)
        p.add_argument('--variant', choices=('s3', 'action-only', 's0'), default='s3')
        p.add_argument('--objective-config', help='Opt-in objective JSON; a new run-id/source snapshot is required')
        if name == 'train':
            p.add_argument('--output', default=DEFAULT_OUTPUT)
            p.add_argument('--run-id', default='S3_optimized_seed42_100k_v1')
            p.add_argument('--master-port', type=int, default=29952)
            p.add_argument('--loader-workers', type=int, default=2)
            p.add_argument('--resume', action='store_true', help='Resume only this same optimized-data run')
            p.add_argument('--dry-run', action='store_true')
            p.add_argument('--stop-after', type=int, default=0, help='Save a recoverable pause at this update')
    a = parser.parse_args()
    try:
        if a.mode == 'prepare':
            result = prepare(a.campaign, a.data, a.labels)
            print(json.dumps(result, indent=2))
        elif a.mode == 'check':
            print(json.dumps(preflight(a)[-1], indent=2))
        else:
            if not 1 <= a.master_port <= 65535 or not 0 <= a.loader_workers <= 8:
                raise ValueError('Invalid port or loader worker count')
            launch(a)
    except (ValueError, FileNotFoundError, FileExistsError, BlockingIOError) as error:
        parser.exit(2, str(error) + '\n')


if __name__ == '__main__':
    main()
