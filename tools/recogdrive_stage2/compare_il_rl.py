"""Paired released IL/RL checkpoints on held-out Navtest, without training.

Both policies run the unmodified official model and sampler. The original IL
forward is instrumented only to retain per-draw MSE and the sampled noise/time.
GT is a separate loss/evaluation target; it never enters trajectory sampling.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

from .assets import atomic_json, check_official, digest, read
from .evaluate import make_agent_input, scene_seed, signature, validate_population

SOURCE = '/mnt/project/recogdrive-stage2-optimized-artifacts/20261005/reference_sources/recogdrive'
REVISION = '6b8d8f5e01346c71094651c81dcaf66405dbc04e'
MANIFEST = '/mnt/project/recogdrive-stage2-evaluation-artifacts/20261006/navtest_current_manifest_v1.json'
METRIC = '/mnt/project/local-interaction-mask-v2-artifacts/20260927/navtest_cache_index.json'
DEVKIT = '/mnt/project/local-interaction-mask-v2-artifacts/20260927/reference_sources/navsim-v1.1'
CKPT_ROOT = Path('/mnt/project/VLA-AD/checkpoints/recogdrive')
HASHES = dict(IL='4569221da6962d4e26d21a463cd70c06a7ad4d76654943fb3870086f43bdb443',
              RL='23fb949b10387efeff87e093703d5faf41b61b9e5ead16d4076661e3c7c08a20')
REVISIONS = dict(IL='0fa96f2159d5bc0f816ecd3818f00541db9db6ff',
                 RL='7346ede73f03ed577811024f1a5d16fb063a47b9')


def setup_source():
    from .compatibility import install_annotation_compatibility
    source = check_official(SOURCE, REVISION)
    sys.path.insert(0, str(source))
    install_annotation_compatibility(source)


def checkpoint_path(name):
    return CKPT_ROOT / f'ReCogDrive-2B-{name}/ReCogDrive_Diffusion_Planner_2B_{name}.ckpt'


def prepare(args):
    import numpy as np
    import torch
    setup_source()
    from navsim.common.dataclasses import Scene, SceneMetadata
    from navsim.agents.recogdrive.recogdrive_features import TrajectoryTargetBuilder
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    manifest = read(MANIFEST); metric = read(METRIC)
    validate_population(manifest['rows'], metric)
    checkpoints = {}; states = {}
    for name in ('IL', 'RL'):
        p = checkpoint_path(name)
        if digest(p) != HASHES[name]:
            raise ValueError('Released weight hash mismatch: ' + name)
        c = torch.load(p, map_location='cpu', weights_only=False)
        states[name] = c['state_dict']
        checkpoints[name] = dict(path=str(p), sha256=HASHES[name], revision=REVISIONS[name],
            repo=f'owl10/ReCogDrive-2B-{name}', epoch=c['epoch'], global_step=c['global_step'])
    differences = [k for k, v in states['IL'].items() if not torch.equal(v,
        states['RL'][k.replace('agent.action_head.', 'agent.action_head.old_policy.', 1)])]
    if differences:
        raise ValueError('Released IL is not the frozen reference inside this RL checkpoint')
    del states
    groups = defaultdict(list)
    for row in manifest['rows']: groups[row['log']].append(row)
    builder = TrajectoryTargetBuilder(TrajectorySampling(num_poses=8, interval_length=.5))
    targets = {}; log_hashes = {}
    for log, rows in sorted(groups.items()):
        path = Path('/mnt/navsim/test_navsim_logs/test') / (log + '.pkl')
        log_hashes[log] = digest(path)
        with path.open('rb') as f: frames = pickle.load(f)
        indices = {f['token']: i for i, f in enumerate(frames)}
        for row in rows:
            i = indices[row['token']]; window = frames[i-3:i+9]
            if len(window) != 12: raise ValueError('Missing original future')
            # No map or sensor is needed by the official trajectory target API.
            metadata = SceneMetadata(log_name=log, scene_token=frames[i]['scene_token'],
                map_name=frames[i]['map_location'], initial_token=row['token'],
                num_history_frames=4, num_future_frames=8)
            scene = Scene(metadata, None, [SimpleNamespace(ego_status=Scene._build_ego_status(f)) for f in window])
            gt = builder.compute_targets(scene)['trajectory'].numpy()
            if gt.shape != (8, 3) or not np.isfinite(gt).all(): raise ValueError('Bad GT')
            targets[row['token']] = gt.tolist()
        print('original GT', len(targets), flush=True)
    atomic_json(root/'targets.json', dict(schema='official_logged_navtest_ego_gt_v1',
        targets=targets, source_log_sha256=log_hashes, source=SOURCE, revision=REVISION,
        generator='Scene.get_future_trajectory + TrajectoryTargetBuilder; no optimized targets'))
    atomic_json(root/'requested_index.json', metric)
    identity = dict(schema='recogdrive_released_IL_RL_paired_navtest_v1', checkpoints=checkpoints,
        official_source=SOURCE, official_revision=REVISION, current_manifest=MANIFEST,
        manifest_sha256=digest(MANIFEST), target_sha256=digest(root/'targets.json'),
        metric_index=METRIC, metric_sha256=digest(METRIC), stage1=manifest['identity']['stage1'],
        scenes=12146, logs=136, split='navtest', draws=args.draws, sampling_seed=42,
        loss='unmodified original forward: normalized-GT epsilon prediction MSE, Uniform integer t in [0,99]',
        loss_mode='eval/no_grad; same scene-bound seed and same vectorized Monte Carlo draws for IL/RL',
        trajectory='unmodified get_action: five-step DDIM, deterministic=False, one candidate, scene seed42',
        precision='official BF16 VLM -> full FP32 hidden sequence; FP32 planner/loss; TF32 disabled',
        RL_frozen_reference_bitwise_equals_IL=True, reference_tensors=347, optimizer_updates=0,
        official_reported_PDMS=dict(IL=86.5, RL=90.8),
        official_reported_source='https://github.com/xiaomi-research/recogdrive#checkpoint')
    atomic_json(root/'IDENTITY.json', identity)


def load_models(identity):
    import torch
    setup_source()
    from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
    from navsim.agents.recogdrive.recogdrive_features import ReCogDriveFeatureBuilder
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    models = {}
    for name in ('IL', 'RL'):
        cp = identity['checkpoints'][name]
        if digest(cp['path']) != cp['sha256']: raise ValueError('Checkpoint changed')
        model = ReCogDriveAgent(TrajectorySampling(num_poses=8, interval_length=.5),
            vlm_path=identity['stage1']['path'], cache_hidden_state=True, cache_mode=False,
            vlm_size='small', dit_type='small', sampling_method='ddim', grpo=False)
        raw = torch.load(cp['path'], map_location='cpu', weights_only=False)['state_dict']
        prefix = 'agent.action_head.'
        if any(not k.startswith(prefix) for k in raw): raise ValueError('Unknown checkpoint prefix')
        state = {k[len(prefix):]: v for k, v in raw.items() if not k.startswith(prefix+'old_policy.')}
        if len(state) != 347 or any(v.is_floating_point() and v.dtype != torch.float32 for v in state.values()):
            raise ValueError('Expected original FP32 planner tensors')
        model.action_head.load_state_dict(state, strict=True)
        model.float().eval().requires_grad_(False)
        models[name] = model
    builder = ReCogDriveFeatureBuilder(cache_hidden_state=True, cache_mode=True,
        model_type='internvl', checkpoint_path=identity['stage1']['path'], device='cuda:0')
    builder.backbone.eval().requires_grad_(False)
    for model in models.values(): model.get_feature_builders = lambda: [builder]
    return models, builder


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def original_il_loss(model, features, gt, draws, seed):
    """Observe native forward; do not substitute its noise or time distribution."""
    import torch
    from transformers.feature_extraction_utils import BatchFeature
    head = model.action_head
    hidden = features['last_hidden_state'].cuda().float().unsqueeze(0).expand(draws, -1, -1)
    history = features['history_trajectory'].cuda().float().reshape(1, 12).expand(draws, -1)
    status = features['status_feature'].cuda().float().reshape(1, 8).expand(draws, -1)
    action = torch.as_tensor(gt, device='cuda', dtype=torch.float32).unsqueeze(0).expand(draws, -1, -1)
    inputs = BatchFeature(dict(his_traj=history, status_feature=status,
        state=torch.cat((status, history), dim=1), action=action))
    captured = {}; native_randn = torch.randn_like; native_time = head.sample_time
    def noise(*a, **kw):
        value = native_randn(*a, **kw); captured['noise'] = value; return value
    def times(*a, **kw):
        value = native_time(*a, **kw); captured['time'] = value; return value
    handle = head.action_decoder.register_forward_hook(lambda mod, ins, out: captured.update(pred=out))
    try:
        seed_all(seed)
        with patch('torch.randn_like', side_effect=noise), patch.object(head, 'sample_time', side_effect=times):
            native_loss = head(hidden, inputs)['loss']
        error = (captured['pred'] - captured['noise']).square()
        torch.testing.assert_close(native_loss, error.mean(), atol=1e-7, rtol=1e-6)
        if not torch.isfinite(error).all(): raise FloatingPointError('Nonfinite original IL loss')
        return dict(mean=float(native_loss), draws=error.mean((1, 2)).cpu().tolist(),
            per_channel=error.mean((0, 1)).cpu().tolist(), per_horizon=error.mean((0, 2)).cpu().tolist(),
            times=captured['time'].cpu().tolist(),
            noise_sha256=hashlib.sha256(captured['noise'].cpu().numpy().tobytes()).hexdigest())
    finally:
        handle.remove()


def worker(args):
    import numpy as np
    import torch
    os.environ['LOCAL_RANK'] = '0'; torch.cuda.set_device(0); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    root = Path(args.output); identity = read(root/'IDENTITY.json'); ident = signature(identity)
    if digest(MANIFEST) != identity['manifest_sha256'] or digest(root/'targets.json') != identity['target_sha256']:
        raise ValueError('Input/target identity changed')
    models, builder = load_models(identity)
    rows = read(MANIFEST)['rows']; rows = rows[:args.limit] if args.limit else rows
    assigned = rows[args.rank::args.world_size]; targets = read(root/'targets.json')['targets']
    for name in models: (root/name/'predictions').mkdir(parents=True, exist_ok=True)
    (root/'relations').mkdir(exist_ok=True)
    begin = time.time(); completed = 0; resumed = 0; parity_scene_token = None
    for row in assigned:
        if time.time()-begin > args.max_seconds: raise TimeoutError('Finite evaluation window exhausted')
        token = row['token']; record = root/'relations'/f'{token}.json'
        if record.exists():
            saved = read(record)
            if saved['identity_sha256'] != ident: raise ValueError('Foreign paired result')
            for name in models:
                p = root/name/'predictions'/f'{token}.npz'
                if digest(p) != read(p.with_suffix('.json'))['proposal_sha256']: raise ValueError('Changed prediction')
            completed += 1; resumed += 1; continue
        started = time.time(); current = make_agent_input(row)
        with torch.inference_mode():
            features = builder.compute_features(current)
            if set(features) != {'history_trajectory','high_command_one_hot','status_feature','last_hidden_state'}:
                raise ValueError('Unexpected current features')
            proposal_seed = scene_seed(token, 42)
            proposals = {}
            # Produce both actual sampled plans before loading the GT loss tensor.
            for name, model in models.items():
                seed_all(proposal_seed)
                proposals[name] = model.forward(dict(features))['pred_traj'].float().cpu().squeeze(0).numpy()
                if completed == 0:
                    seed_all(proposal_seed)
                    expected = model.compute_trajectory(current).poses
                    if isinstance(expected, torch.Tensor): expected = expected.numpy()
                    np.testing.assert_allclose(proposals[name], expected, atol=1e-6, rtol=0)
                    parity_scene_token = token
            gt = np.array(targets[token], dtype=np.float32)
            loss_seed = int.from_bytes(hashlib.sha256(f'il-loss:42:{token}'.encode()).digest()[:8], 'big') % 2**31
            losses = {name: original_il_loss(model, features, gt, identity['draws'], loss_seed)
                      for name, model in models.items()}
            if losses['IL']['times'] != losses['RL']['times'] or losses['IL']['noise_sha256'] != losses['RL']['noise_sha256']:
                raise ValueError('Paired noise/time diverged')
        metrics = {}
        for name, pred in proposals.items():
            if pred.shape != (8, 3) or not np.isfinite(pred).all(): raise ValueError('Invalid sampled trajectory')
            delta = pred-gt; xy = np.linalg.norm(delta[:, :2], axis=-1)
            yaw = np.arctan2(np.sin(delta[:,2]), np.cos(delta[:,2]))
            metrics[name] = dict(ADE_m=float(xy.mean()), FDE_m=float(xy[-1]),
                XY_MSE_m2=float(np.square(delta[:,:2]).mean()), heading_MAE_rad=float(np.abs(yaw).mean()),
                native_validation_L1_mixed_units=float(np.abs(delta).mean()), L2_by_time_m=xy.tolist())
            path = root/name/'predictions'/f'{token}.npz'
            tmp = path.with_suffix(f'.{os.getpid()}.tmp')
            with tmp.open('wb') as f: np.savez(f, trajectory=pred)
            tmp.replace(path)
            atomic_json(path.with_suffix('.json'), dict(token=token, log=row['log'], status='ok',
                proposal_sha256=digest(path), identity_sha256=ident, scene_seed=proposal_seed))
        atomic_json(record, dict(token=token, log=row['log'], identity_sha256=ident,
            losses=losses, metrics=metrics, seconds=time.time()-started, loss_seed=loss_seed))
        completed += 1
        if completed == 1 or completed % 25 == 0:
            atomic_json(root/f'progress_{args.rank}.json', dict(completed=completed, requested=len(assigned),
                seconds=time.time()-begin, peak_memory=torch.cuda.max_memory_allocated(), pid=os.getpid()))
    for name in models:
        atomic_json(root/name/f'shard_{args.rank}.json', dict(status='complete', completed=completed,
            requested=len(assigned), identity_sha256=ident))
    atomic_json(root/f'worker_{args.rank}.json', dict(status='COMPLETE', completed=completed,
        seconds=time.time()-begin, peak_memory=torch.cuda.max_memory_allocated(), optimizer_updates=0,
        parity_first_scene=parity_scene_token is not None, parity_scene_token=parity_scene_token,
        resumed_scenes=resumed, newly_computed_scenes=completed-resumed, pid=os.getpid()))


def launch(args):
    root = Path(args.output); identity = read(root/'IDENTITY.json')
    selected = [int(x) for x in args.gpus.split(',')]
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1')
    rows = read(MANIFEST)['rows']; rows = rows[:args.limit] if args.limit else rows
    metric = {r['token']:r for r in read(METRIC)}
    atomic_json(root/'requested_index.json', [metric[r['token']] for r in rows])
    policy = read('/mnt/project/server_dispatch_policy.json')
    if 'training-vla-zt' not in policy['allowed_hosts']: raise ValueError('Local host withdrawn')
    snapshot = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'], text=True)
    gpu_info = {int(row.split(',')[0]):row for row in snapshot.splitlines()}
    if any(float(gpu_info[g].split(',')[2])<16000 for g in selected): raise ValueError('Insufficient current GPU headroom')
    if len({gpu_info[g].split(',')[1].strip() for g in selected}) != len(selected): raise ValueError('Duplicate GPU')
    registration = dict(identity_sha256=signature(identity), gpus=selected, gpu_snapshot=snapshot,
        host='training-vla-zt', limit=args.limit, CPU_workers_per_model=args.cpu_workers,
        max_worker_seconds=args.max_seconds, source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        code_sha256=digest(__file__), start_unix=time.time(), policy_sha256=digest('/mnt/project/server_dispatch_policy.json'))
    atomic_json(root/'EXECUTION.json', registration)
    children = []; begin = time.time()
    for rank, gpu in enumerate(selected):
        cmd = [sys.executable,'-u','-m','tools.recogdrive_stage2.compare_il_rl','worker','--output',str(root),
            '--rank',str(rank),'--world-size',str(len(selected)),'--limit',str(args.limit),'--max-seconds',str(args.max_seconds)]
        log = (root/f'worker_{rank}.log').open('a')
        process = subprocess.Popen(cmd,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
        children.append((process,log,'GPU',cmd))
    for name in ('IL','RL'):
        cmd = [sys.executable,'-u','-m','tools.local_interaction_mask_v2.score_async','--devkit',DEVKIT,
            '--index',str(root/'requested_index.json'),'--predictions',str(root/name),
            '--output',str(root/(name+'_scores')),'--workers',str(args.cpu_workers),
            '--export-shards',str(len(selected)),'--timeout-seconds',str(int(args.max_seconds+600))]
        if not args.limit: cmd += ['--benchmark-navtest']
        log = (root/f'score_{name}.log').open('a')
        process = subprocess.Popen(cmd,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
        children.append((process,log,'CPU',cmd))
    atomic_json(root/'LAUNCH.json', dict(pid=os.getpid(), children=[dict(pid=p.pid,kind=k,argv=c) for p,l,k,c in children]))
    try:
        pending = set(range(len(children)))
        while pending:
            for i in list(pending):
                code = children[i][0].poll()
                if code is None: continue
                if code: raise RuntimeError(f'{children[i][2]} worker {i} exited {code}')
                pending.remove(i)
            if time.time()-begin > args.max_seconds+900: raise TimeoutError('Evaluation controller cap')
            time.sleep(2)
        atomic_json(root/'COMPLETE.json', dict(status='COMPLETE',scenes=len(rows),
            seconds=time.time()-begin, optimizer_updates=0, diagnostic=bool(args.limit)))
    finally:
        for p,log,kind,cmd in children:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            log.close()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('command', choices=['prepare','worker','launch'])
    p.add_argument('--output',required=True);p.add_argument('--draws',type=int,default=8)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--gpus',default='0')
    p.add_argument('--cpu-workers',type=int,default=4);p.add_argument('--max-seconds',type=float,default=7200)
    a = p.parse_args()
    if a.draws<1 or not 0<=a.rank<a.world_size: raise ValueError('Invalid sampling/shard')
    {'prepare':prepare,'worker':worker,'launch':launch}[a.command](a)


if __name__ == '__main__': main()
