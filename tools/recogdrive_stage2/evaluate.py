"""Strict official ReCogDrive inference; teachers and future labels are absent.

The Stage1 backbone retains its official BF16 computation and full-sequence
FP32 feature interface. Stage2 uses the trained FP32 master tensors, TF32 off,
the unchanged official five-step DDIM sampler and one trajectory per scene.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time

from .assets import atomic_json, check_official, digest, read


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def signature(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def scene_seed(token, seed):
    return int.from_bytes(hashlib.sha256(f'recogdrive:{seed}:{token}'.encode()).digest()[:8], 'big') % (2**31)


def validate_population(current, metric):
    a = {r['token']: r['log'] for r in current}
    b = {r['token']: r['log'] for r in metric}
    if len(a) != len(current) or len(b) != len(metric) or a != b:
        raise ValueError('Canonical current/metric token and log identities differ')
    if len(a) != 12146 or len(set(a.values())) != 136:
        raise ValueError('This entrypoint requires the complete canonical Navtest')


def build_manifest(args):
    source = check_official(args.official_source, args.official_revision)
    sys.path.insert(0, str(source))
    from navsim.common.dataclasses import AgentInput, SensorConfig
    import numpy as np
    current_root = Path(args.current_root)
    current = read(current_root / 'index.json')
    metric = read(args.metric_index)
    validate_population(current, metric)
    training = read(args.training_manifest)
    if set(r['log'] for r in training['rows']) & set(r['log'] for r in current):
        raise ValueError('Navtest logs overlap the Stage2 training/validation population')
    groups = defaultdict(list)
    for r in current:
        groups[r['log']].append(r)
    result = {}
    for log, rows in sorted(groups.items()):
        with (Path(args.logs) / (log + '.pkl')).open('rb') as f:
            frames = pickle.load(f)
        positions = {x['token']: i for i, x in enumerate(frames)}
        for r in rows:
            i = positions[r['token']]
            history = frames[i - 3:i + 1]
            if len(history) != 4:
                raise ValueError('Incomplete legal current history')
            inputs = AgentInput.from_scene_dict_list(history, Path(args.sensors), 4,
                SensorConfig.build_no_sensors(), load_image_path=True)
            now = inputs.ego_statuses[-1]
            raw = frames[i]['cams']['CAM_F0']['data_path']
            image = Path(args.sensors) / raw
            if not image.is_file():
                image = Path(read(current_root / 'current' / (r['token'] + '.json'))['image_paths'][0])
                if not str(image).endswith('/' + raw):
                    raise ValueError('Raw front-image identity differs')
            if not image.is_file():
                raise FileNotFoundError(image)
            row = dict(token=r['token'], log=log, image=str(image),
                history=[x.ego_pose.tolist() for x in inputs.ego_statuses],
                command=np.asarray(now.driving_command).tolist(), velocity=now.ego_velocity.tolist(),
                acceleration=now.ego_acceleration.tolist())
            if not all(np.isfinite(np.asarray(row[k], dtype=float)).all()
                       for k in ('history', 'command', 'velocity', 'acceleration')):
                raise ValueError('Nonfinite current observation')
            result[r['token']] = row
        print('Current inputs', len(result), flush=True)
    value = dict(identity=dict(schema='official_recogdrive_current_inference_v1', split='navtest',
        stage1=training['identity']['stage1'], official_revision=args.official_revision,
        current_index_sha256=digest(current_root / 'index.json'), metric_index_sha256=digest(args.metric_index),
        training_manifest_sha256=digest(args.training_manifest), scenes=len(current), logs=len(groups),
        current_views=['cam_f0'], history=4, future_inputs=False, targets=False),
        rows=[result[r['token']] for r in current])
    if Path(args.output).exists() and read(args.output) != value:
        raise ValueError('Immutable evaluation manifest already exists with another identity')
    atomic_json(args.output, value)


def make_agent_input(row):
    from navsim.common.dataclasses import AgentInput, Cameras, Camera, EgoStatus
    statuses = [EgoStatus(ego_pose=x, ego_velocity=row['velocity'], ego_acceleration=row['acceleration'],
                         driving_command=row['command']) for x in row['history']]
    cameras = []
    for i in range(4):
        c = Cameras(**{k: Camera() for k in Cameras.__dataclass_fields__})
        if i == 3:
            c.cam_f0.image = Path(row['image'])
        cameras.append(c)
    return AgentInput(statuses, cameras, [])


def load_agent(args, manifest):
    import torch
    from .compatibility import install_annotation_compatibility
    source = check_official(args.official_source, args.official_revision)
    sys.path.insert(0, str(source))
    install_annotation_compatibility(source)
    from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
    from navsim.agents.recogdrive.recogdrive_features import ReCogDriveFeatureBuilder
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    identity = checkpoint['recogdrive_identity']
    if identity != read(Path(args.training_run) / 'identity.json') or identity['scope'] != 'formal':
        raise ValueError('Checkpoint and completed formal training identity differ')
    if identity['stage1'] != manifest['identity']['stage1'] or identity['official_revision'] != args.official_revision:
        raise ValueError('Stage1 or official model source changed')
    if read(Path(args.training_run) / 'status.json')['status'] != 'COMPLETE':
        raise ValueError('This evaluation locks a completed training run')
    agent = ReCogDriveAgent(TrajectorySampling(num_poses=8, interval_length=.5),
        vlm_path=identity['stage1']['path'], vlm_type='internvl', dit_type='small',
        sampling_method='ddim', cache_hidden_state=True, cache_mode=False, vlm_size='small', grpo=False)
    state = checkpoint['state_dict']
    if any(not k.startswith('agent.') for k in state):
        raise ValueError('Unexpected non-agent checkpoint tensors')
    if any(v.is_floating_point() and v.dtype != torch.float32 for v in state.values()):
        raise ValueError('Stage2 checkpoint is not the original FP32 master state')
    agent.load_state_dict({k[len('agent.'):]: v for k, v in state.items()}, strict=True)
    agent.float().eval()
    if str(agent.action_head.config) != identity['architecture']:
        raise ValueError('Official architecture or sampler changed')
    for name, value in agent.state_dict().items():
        if not torch.isfinite(value).all():
            if not (name == 'action_head.eta.eta_logit' and torch.isposinf(value).all()
                    and not dict(agent.named_parameters())[name].requires_grad):
                raise FloatingPointError('Nonfinite checkpoint tensor: ' + name)
    if digest(Path(identity['stage1']['path']) / 'model.safetensors') != identity['stage1']['sha256']:
        raise ValueError('Public Stage1 weight contents changed')
    builder = ReCogDriveFeatureBuilder(cache_hidden_state=True, cache_mode=True, model_type='internvl',
        checkpoint_path=identity['stage1']['path'], device='cuda:0')
    from safetensors import safe_open
    with safe_open(Path(identity['stage1']['path']) / 'model.safetensors', framework='pt') as weights:
        if set(weights.keys()) != set(builder.backbone.model.state_dict()):
            raise ValueError('Stage1 core tensor inventory differs from the public file')
    builder.backbone.eval()
    for p in builder.backbone.parameters():
        p.requires_grad_(False)
    agent.get_feature_builders = lambda: [builder]
    return agent, builder, checkpoint


def export(args):
    import numpy as np
    import torch
    os.environ['LOCAL_RANK'] = '0'
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = read(args.manifest)
    bank = Path(args.output)
    identity = read(bank / 'identity.json')
    if identity['manifest_sha256'] != digest(args.manifest) or identity['checkpoint_sha256'] != digest(args.checkpoint):
        raise ValueError('Locked checkpoint or current-input manifest changed')
    if identity['world_size'] != args.world_size or identity['sampling_seed'] != args.sampling_seed:
        raise ValueError('Locked scene partition or RNG protocol changed')
    agent, builder, checkpoint = load_agent(args, manifest)
    identity_sha = signature(identity)
    rows = manifest['rows'][:args.limit] if args.limit else manifest['rows']
    assigned = rows[args.rank::args.world_size]
    begin = time.time()
    completed = 0
    bank.joinpath('predictions').mkdir(parents=True, exist_ok=True)
    for row in assigned:
        if time.time() - begin > args.max_seconds or (bank / 'STOP_REQUESTED').exists():
            atomic_json(bank / f'shard_{args.rank}.json', dict(status='paused', completed=completed,
                requested=len(assigned), identity_sha256=identity_sha))
            return
        token = row['token']
        input_hash = signature(dict(row=row, image_sha256=digest(row['image'])))
        target = bank / 'predictions' / (token + '.npz')
        receipt = target.with_suffix('.json')
        if receipt.exists():
            old = read(receipt)
            if old['identity_sha256'] != identity_sha or old['input_sha256'] != input_hash or old['proposal_sha256'] != digest(target):
                raise ValueError('Foreign or mutated prediction receipt')
            completed += 1
            continue
        current = make_agent_input(row)
        with torch.inference_mode():
            features = builder.compute_features(current)
            if set(features) != {'history_trajectory', 'high_command_one_hot', 'status_feature', 'last_hidden_state'}:
                raise ValueError('Non-current feature branch entered inference')
            for v in features.values():
                if not torch.isfinite(v).all():
                    raise FloatingPointError('Nonfinite legal current feature')
            seed = scene_seed(token, args.sampling_seed)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            poses = agent.forward(dict(features))['pred_traj'].float().cpu().squeeze(0).numpy()
            if poses.shape != (8, 3) or not np.isfinite(poses).all():
                raise FloatingPointError('Illegal official trajectory')
            if completed == 0:
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                reference = agent.compute_trajectory(current).poses
                if isinstance(reference, torch.Tensor):
                    reference = reference.detach().cpu().numpy()
                error = float(np.max(np.abs(poses - reference)))
                if error > 1e-6:
                    raise ValueError('Original compute_trajectory parity failed')
                atomic_json(bank / f'inference_parity_rank{args.rank}.json', dict(token=token,
                    maximum_pose_error=error, passed=True, original_path='unmodified official compute_trajectory',
                    precision=identity['precision'], scene_seed=seed, identity_sha256=identity_sha))
        tmp = target.with_suffix(f'.{os.getpid()}.tmp')
        with tmp.open('wb') as stream:
            np.savez(stream, trajectory=poses)
        tmp.replace(target)
        atomic_json(receipt, dict(token=token, log=row['log'], status='ok', input_sha256=input_hash,
            proposal_sha256=digest(target), identity_sha256=identity_sha, scene_seed=seed))
        completed += 1
        if completed == 1 or completed % 25 == 0:
            atomic_json(bank / f'progress_rank{args.rank}.json', dict(status='RUNNING', completed=completed,
                requested=len(assigned), seconds=time.time()-begin, peak_allocated=torch.cuda.max_memory_allocated(),
                optimizer_updates=0, updated_unix=time.time()))
    atomic_json(bank / f'shard_{args.rank}.json', dict(status='complete', completed=completed,
        requested=len(assigned), identity_sha256=identity_sha, seconds=time.time()-begin, optimizer_updates=0))


def main():
    p = argparse.ArgumentParser(__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    m = sub.add_parser('manifest')
    for name in ('official-source', 'official-revision', 'training-manifest', 'current-root',
                 'metric-index', 'logs', 'sensors', 'output'):
        m.add_argument('--' + name, required=True)
    e = sub.add_parser('export')
    for name in ('official-source', 'official-revision', 'training-run', 'checkpoint', 'manifest', 'output'):
        e.add_argument('--' + name, required=True)
    e.add_argument('--sampling-seed', type=int, default=42)
    e.add_argument('--rank', type=int, default=0)
    e.add_argument('--world-size', type=int, default=1)
    e.add_argument('--limit', type=int, default=0)
    e.add_argument('--max-seconds', type=float, default=21600)
    args = p.parse_args()
    if args.command == 'manifest':
        build_manifest(args)
    else:
        if not 0 <= args.rank < args.world_size or args.limit < 0:
            raise ValueError('Invalid deterministic scene partition')
        export(args)


if __name__ == '__main__':
    main()
