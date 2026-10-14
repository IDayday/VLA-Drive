"""Shared current perception preparation; never instantiates an ego planner."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import time
import numpy as np
import torch
from torch import distributed as dist, nn
from torch.nn.parallel import DistributedDataParallel
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from starVLA.model.modules.vehicle_joint.initialization import initialization_seed
from starVLA.model.modules.structured_world.grid import GridSpec
from starVLA.model.modules.structured_world.geo_bev import SingleFrameGeoBEV
from starVLA.model.modules.structured_world.semantics import StructuredSemantics
from starVLA.model.modules.structured_world.losses import scene_means, scene_reduce, binary_loss
from tools.ddpolicy_vehicle.training_state import epoch_batches
from tools.foresight.student_state import capture_rng, restore_rng, learning_rate, validate_rank_batches
from tools.structured_world.recovery_boundary import exact
from tools.structured_world.training_assets import file_digest, verify_checkpoint_files


class GeometryCache:
    def __init__(self, root, *, debug=False):
        self.root = Path(root)
        self.identity = json.loads((self.root/'identity.json').read_text())
        self.complete = json.loads((self.root/'COMPLETE.json').read_text())
        if self.complete['identity'] != self.identity['identity'] or self.complete['scenes'] != self.complete['expected_scenes']:
            raise ValueError('Incomplete geometry label cache')
        if not debug and self.identity['population_kind'] != 'full_train_population':
            raise ValueError('Formal preparation requires the complete training population')
        if self.identity['source_current_identity']['split'] != 'train':
            raise ValueError('Geometry preparation cannot use development/test data')
        self.index = json.loads((self.root/'index.json').read_text())
        self.checked = set()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        token = self.index[index]['token']
        path = self.root/'labels'/(token+'.npz')
        if token not in self.checked:
            meta = json.loads((self.root/'records'/(token+'.json')).read_text())
            if meta['cache_identity'] != self.identity['identity'] or hashlib.sha256(path.read_bytes()).hexdigest() != meta['label_sha256']:
                raise ValueError('Corrupt/foreign geometry label cache')
            self.checked.add(token)
        with np.load(path, allow_pickle=False) as source:
            return {key: torch.from_numpy(source[key].copy()) for key in
                    ('geometry_rgb', 'geometry_pixel_valid', 'road_distance', 'road_valid', 'occupancy', 'occupancy_valid', 'depth',
                     'calibration_sensor2ego', 'calibration_intrinsics', 'calibration_post_rots', 'calibration_post_trans')}


class GeometryPreparation(nn.Module):
    def __init__(self, imagenet):
        super().__init__()
        with initialization_seed(42+4100):
            self.geometry = SingleFrameGeoBEV(imagenet_checkpoint=imagenet)
        with initialization_seed(42+4400):
            self.current_head = StructuredSemantics('none').current

    def forward(self, batch, generator, *, positive_weight, global_scenes):
        rgb = batch['geometry_rgb'].float()/255.
        images = (rgb-rgb.new_tensor([.485, .456, .406])[None, None, :, None, None])/rgb.new_tensor([.229, .224, .225])[None, None, :, None, None]
        current = self.geometry(images, {k: batch['calibration_'+k] for k in ('sensor2ego', 'intrinsics', 'post_rots', 'post_trans')},
                                batch['geometry_pixel_valid'])
        maps = self.current_head(current['B0'])
        grid = self.geometry.grid
        poses = []
        valid = batch['occupancy_valid'][:, 0]
        for row in range(len(valid)):
            cells = torch.nonzero(valid[row], as_tuple=False)
            chosen = cells[torch.randint(len(cells), (1024,), device=cells.device, generator=generator)] if len(cells) else torch.zeros(1024, 2, device=cells.device)
            poses.append(torch.stack((grid.xmin+(chosen[:, 1]+.5)*grid.dx, grid.ymin+(chosen[:, 0]+.5)*grid.dy), -1))
        xy = torch.stack(poses).detach()
        read, in_grid = grid.sample(maps, xy)
        road_gt, _ = grid.sample(batch['road_distance'][:, None], xy)
        road_mask, _ = grid.sample(batch['road_valid'][:, None].float(), xy, mode='nearest')
        occ_gt, _ = grid.sample(batch['occupancy'][:, :1].float(), xy, mode='nearest')
        occ_mask, _ = grid.sample(valid[:, None].float(), xy, mode='nearest')
        road, _ = scene_means(torch.nn.functional.smooth_l1_loss(read[..., 0].tanh(), road_gt[..., 0]/10., reduction='none'),
                              (road_mask[..., 0] > .5) & in_grid)
        occ, _ = binary_loss(read[..., 1], occ_gt[..., 0] == 1, (occ_mask[..., 0] > .5) & in_grid,
                             positive_weight=positive_weight)
        depth = []
        cameras = images.shape[1]
        for row in range(len(images)):
            depth.append(self.geometry.view.get_depth_loss(batch['depth'][row:row+1],
                       [v[row*cameras:(row+1)*cameras] for v in current['depth']]))
        losses = {'road': scene_reduce(road, global_scenes=global_scenes),
                  'occupancy': scene_reduce(occ, global_scenes=global_scenes),
                  'depth': scene_reduce(torch.stack(depth), global_scenes=global_scenes)}
        return {'loss': sum(losses.values()), 'losses': losses}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--imagenet', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--debug-updates', type=int, default=0)
    parser.add_argument('--stop-after', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save-every', type=int, default=1000)
    args = parser.parse_args()
    policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    if socket.gethostname().removesuffix('-worker-0') not in policy['task_authorizations']['structured_world_fgtr_round1']['allowed_hosts']:
        raise ValueError('This host is not authorized for the task')
    rank, world, local = int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local); dist.init_process_group('nccl')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    random.seed(42+rank); np.random.seed(42+rank); torch.manual_seed(42+rank); torch.cuda.manual_seed_all(42+rank)
    data = GeometryCache(args.cache, debug=bool(args.debug_updates))
    validate_rank_batches(len(data), 32, world, 4)
    if world != 8:
        raise ValueError('This registered geometry trainer uses one complete eight-GPU host')
    horizon = args.debug_updates or args.epochs*math.ceil(len(data)/32)
    if not 5 <= args.epochs <= 10 and not args.debug_updates:
        raise ValueError('Formal common geometry preparation permits five to ten epochs')
    warmup = max(1, min(500, horizon//10))
    paths = [Path(__file__), ROOT/'tools/structured_world/recovery_boundary.py', ROOT/'tools/structured_world/training_assets.py']
    for directory in ('starVLA/model/modules/structured_world', 'third_party/resworld/ported'):
        paths.extend(sorted((ROOT/directory).rglob('*.py')))
    code_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    contract = {'training_task': 'current_depth_road_occupancy_no_planner', 'dataset': data.identity['dataset'],
                'label_cache_identity': data.identity['identity'], 'population': data.identity['population_kind'],
                'horizon_updates': horizon, 'global_batch': 32, 'precision': 'FP32 TF32 disabled',
                'initialization_seed': 42, 'ImageNet_sha256': hashlib.sha256(args.imagenet.read_bytes()).hexdigest(),
                'geometry_grid': asdict(GridSpec()), 'code_hashes': code_hashes,
                'training_source_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'external_source_versions': json.loads((ROOT/'third_party/LOCK.json').read_text()),
                'external_source_addendum': json.loads((ROOT/'third_party/ADDENDUM_LOCK.json').read_text()),
                'epochs_planned': args.epochs, 'base_lr': 1e-4, 'minimum_lr': 1e-5,
                'scheduler_warmup_updates': warmup}
    identity = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        if (args.output/'identity.json').exists() and json.loads((args.output/'identity.json').read_text())['identity'] != identity:
            raise ValueError('Geometry preparation output already belongs to another contract')
        temporary = args.output/'identity.tmp'
        temporary.write_text(json.dumps({'identity': identity, **contract}, indent=2)+'\n')
        temporary.replace(args.output/'identity.json')
    dist.barrier()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(GeometryPreparation(args.imagenet)).float().cuda()
    model = DistributedDataParallel(model, device_ids=[local], broadcast_buffers=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .95), weight_decay=.001)
    generators = {'queries': torch.Generator(device='cuda').manual_seed(4200+rank)}
    completed, epoch, offset, exposure, seconds = 0, 0, 0, 0, 0.
    if args.resume:
        tag = (args.output/'latest').read_text().strip()
        complete = json.loads((args.output/tag/'COMPLETE.json').read_text())
        if Path(tag).name != tag or complete['identity'] != identity:
            raise ValueError('Incomplete/foreign geometry recovery point')
        verify_checkpoint_files(args.output/tag, complete, rank=rank, world=world)
        dist.barrier()
        state = torch.load(args.output/tag/f'rank{rank}.pt', map_location='cpu', weights_only=False)
        if state['identity'] != identity:
            raise ValueError('Foreign geometry checkpoint')
        shared = torch.load(args.output/tag/'model_optimizer.pt', map_location='cpu', weights_only=False)
        model.module.load_state_dict(shared['model'], strict=True); optimizer.load_state_dict(shared['optimizer'])
        restore_rng(state['rng'], generators)
        completed, epoch, offset, exposure, seconds = [state[k] for k in ('completed', 'epoch', 'offset', 'exposure', 'seconds')]
        exact(model.module.state_dict(), shared['model'], 'geometry_model_and_buffers')
        exact(optimizer.state_dict(), shared['optimizer'], 'geometry_Adam_moments_steps_and_LR')
        exact(capture_rng(generators), state['rng'], 'geometry_RNG')
        dist.barrier()
        if rank == 0:
            (args.output/'RESUME_BOUNDARY_VERIFIED.json').write_text(json.dumps({
                'passed': True, 'timing': 'before next forward/backward/update', 'world_size': world,
                'model_and_buffers': 'bitwise all ranks', 'Adam_moments_steps_and_LR': 'bitwise all ranks',
                'RNG': 'exact all ranks', 'completed': completed, 'epoch': epoch, 'offset': offset,
                'exposure': exposure, 'scheduler_horizon': horizon}, indent=2)+'\n')
    counts = data.complete['class_counts']['current']
    positive_weight = max(1., math.sqrt(counts['negative']/max(1, counts['positive'])))
    def save(status):
        tag = f'checkpoint_{completed:07d}'
        if shutil.disk_usage(args.output).free < 20*(1 << 30):
            raise RuntimeError('Below 20 GiB reserve; preserving existing checkpoints')
        folder = args.output/('writing_'+tag)
        if rank == 0: folder.mkdir(exist_ok=False)
        dist.barrier()
        state = {'identity': identity, 'rng': capture_rng(generators), 'completed': completed, 'epoch': epoch, 'offset': offset,
                 'exposure': exposure, 'seconds': seconds, 'scheduler_horizon': horizon}
        if rank == 0:
            temporary = folder/'model_optimizer.tmp'
            torch.save({'model': model.module.state_dict(), 'optimizer': optimizer.state_dict()}, temporary)
            temporary.replace(folder/'model_optimizer.pt')
        temporary = folder/f'rank{rank}.tmp'
        torch.save(state, temporary); temporary.replace(folder/f'rank{rank}.pt')
        dist.barrier()
        owned = [folder/f'rank{rank}.pt']+([folder/'model_optimizer.pt'] if rank == 0 else [])
        rank_hashes = [None]*world
        dist.all_gather_object(rank_hashes, {p.name: file_digest(p) for p in owned})
        hashes = {name: value for record in rank_hashes for name, value in record.items()}
        if rank == 0:
            temporary = folder/'COMPLETE.tmp'
            temporary.write_text(json.dumps({'identity': identity, 'ranks': world, 'updates': completed, 'status': status,
                                            'file_sha256': hashes})+'\n')
            temporary.replace(folder/'COMPLETE.json')
            final = args.output/tag
            if final.exists(): raise ValueError('Refusing to overwrite a geometry checkpoint')
            folder.rename(final)
            temporary = args.output/'latest.tmp'; temporary.write_text(tag+'\n'); temporary.replace(args.output/'latest')
            if status == 'COMPLETE':
                shared = {'identity': identity, **contract, 'geometry': model.module.geometry.state_dict(),
                          'current_head': model.module.current_head.state_dict(), 'actual_exposure': exposure,
                          'epochs': exposure/len(data), 'GPU_hours': seconds*world/3600.}
                temporary = args.output/'shared_geometry.tmp'; torch.save(shared, temporary); temporary.replace(args.output/'shared_geometry.pt')
        dist.barrier()
    if not completed:
        save('INITIAL')
    while completed < horizon:
        batches = epoch_batches(len(data), 32, 42, epoch)
        if offset >= len(batches):
            epoch += 1; offset = 0; continue
        started = time.monotonic()
        indices = batches[offset][rank::world]
        samples = [data[i] for i in indices]
        batch = {key: torch.stack([r[key] for r in samples]).cuda(non_blocking=True) for key in samples[0]}
        optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(completed, 1e-4, 1e-5, warmup, horizon)
        for group in optimizer.param_groups: group['lr'] = lr
        result = model(batch, generators['queries'], positive_weight=positive_weight, global_scenes=len(batches[offset]))
        result['loss'].backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.)
        before = next(model.module.current_head.parameters()).detach().flatten()[:512].clone()
        optimizer.step()
        changed = (before != next(model.module.current_head.parameters()).detach().flatten()[:512]).sum()
        dist.all_reduce(changed)
        if not changed:
            raise RuntimeError('FP32 optimizer did not update the sampled current-head parameters')
        torch.cuda.synchronize()
        cost = torch.tensor(time.monotonic()-started, device='cuda'); dist.all_reduce(cost, op=dist.ReduceOp.MAX)
        seconds += float(cost); completed += 1; exposure += len(batches[offset]); offset += 1
        logs = torch.tensor([float(v) for v in result['losses'].values()], device='cuda'); dist.all_reduce(logs); logs /= world
        if rank == 0:
            record = {'updates': completed, 'exposure': exposure, 'epoch': epoch, 'offset': offset,
                      'raw_losses': dict(zip(result['losses'], logs.tolist())), 'lr': lr,
                      'gradient_norm': float(norm), 'clipped': bool(norm > 1.), 'seconds': float(cost),
                      'FP32_changed_elements': int(changed), 'GPU_hours': seconds*world/3600.}
            with (args.output/'metrics.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
            if completed % 10 == 0: print(json.dumps(record), flush=True)
        if (args.output/'STOP_AFTER_CHECKPOINT').exists() or (args.stop_after and completed >= args.stop_after):
            save('PAUSED_FOR_RESTORE_TEST'); break
        if completed % args.save_every == 0 or completed == horizon:
            save('COMPLETE' if completed == horizon else 'RUNNING')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
