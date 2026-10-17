"""Bounded frozen learned-W spatial diagnostics; never a full-development PDMS result.

plan-cache/extract consume only immutable native checkpoints and observation-only
inputs. fit consumes their frozen W plus separately pinned current raw-GT labels.
Real dispatch requires a Lead-bound runtime/endpoint and independent review.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from starVLA.dataloader.st_world_schema import ModelInputs, identity_hash, require_hash
from starVLA.model.modules.st_world.sparse_readout import SparseFeatureReadout
from starVLA.model.modules.vehicle_joint.initialization import file_sha256, initialization_seed, tensor_hash
from tools.st_world_policy.base_binding import immutable_json
from tools.st_world_policy.cache_storage_budget import StorageBudget, json_write_bound

GRID = (18, 24)
VIEWS = ('CAM_F0', 'CAM_L0', 'CAM_R0')
LABEL_SCHEMA = 'current_raw_lidar_distorted_full_rgb_native384_patch_v1'
RECIPE = dict(seed=42, updates=300, scenes_per_update=2, patches_per_view=32,
              dim=128, layers=1, heads=4, outputs=2, lr=.001, weight_decay=.001,
              clip=1., depth_beta=.1, threshold_logit=0., allowed_times_s=[0.])
REPO = Path(__file__).resolve().parents[2]


def exact(value, names, name):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(name + ' missing/unknown fields')


def pinned(ref):
    exact(ref, ('path', 'sha256'), 'pinned reference')
    require_hash(ref['sha256'], 'artifact hash')
    p = Path(ref['path'])
    if not p.is_absolute() or file_sha256(p) != ref['sha256']:
        raise ValueError('Changed pinned artifact: ' + str(p))
    return json.loads(p.read_text())


def identity(value):
    if value.get('identity') != identity_hash({k: v for k, v in value.items() if k != 'identity'}):
        raise ValueError('Changed immutable identity')
    return value


def seal(value):
    return {**value, 'identity': identity_hash(value)}


def source_runtime(expected_source, expected_environment, device, *, configure_device=True):
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO).strip():
        raise ValueError('Commit clean probe runtime before dispatch')
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    if actual != expected_source or runtime_environment() != expected_environment:
        raise ValueError('Registered source/runtime changed')
    if device != 'cpu' and (not device.startswith('cuda:') or not torch.cuda.is_available()):
        raise ValueError('Explicit available CPU/CUDA device required')
    torch.set_num_threads(2)
    if configure_device and device.startswith('cuda:'):
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def runtime_environment():
    from tools.st_world_policy.training_config import environment_record
    return dict(base=environment_record(), scikit_learn=importlib.metadata.version('scikit-learn'))


def checked_rows(manifest, split, count):
    rows = manifest['scenes']
    if (manifest.get('schema') != 'st_fixed_scene_list_v1' or manifest.get('split') != split
            or len(rows) != count or len({r['token'] for r in rows}) != count):
        raise ValueError('Fixed scene population/role differs')
    for row in rows:
        token = row['token']
        if not isinstance(token, str) or not token or any(x in token for x in ('/', '\\', '..')) or not row['log']:
            raise ValueError('Unsafe scene/log identity')
    return rows


def validate_binding(binding):
    identity(binding)
    exact(binding, ('schema', 'identity', 'source_sha', 'environment', 'device', 'training_registration',
                    'checkpoint_tag', 'checkpoint_identity', 'arm', 'observations', 'hidden_dim',
                    'memory_tokens', 'max_seconds'), 'extraction binding')
    if binding['schema'] != 'st_spatial_w_binding_v1' or binding['arm'] not in ('S-P0', 'S-P1', 'S-P2', 'S-P3'):
        raise ValueError('Unknown spatial arm/binding')
    if (type(binding['hidden_dim']) is not int or binding['hidden_dim'] < 1
            or binding['memory_tokens'] != (288 if binding['arm'] == 'S-P3' else 144)
            or not isinstance(binding['max_seconds'], (int, float))
            or not math.isfinite(binding['max_seconds']) or not 0 < binding['max_seconds'] <= 7200):
        raise ValueError('Invalid bounded W shape/time')
    require_hash(binding['checkpoint_identity'], 'checkpoint identity')
    obs = binding['observations']
    exact(obs, ('current_root', 'manifest', 'identity', 'split'), 'observations')
    if obs['split'] not in ('train', 'dev'):
        raise ValueError('Only registered train/dev observations allowed')
    require_hash(obs['identity'], 'observation identity')
    checked_rows(pinned(obs['manifest']), obs['split'], 64 if obs['split'] == 'train' else 16)
    return binding


def cache_requirements(scenes, tokens, hidden):
    payload = tokens * hidden * 4 + tokens
    return dict(output_bytes=scenes * (payload + 8192), staging_bytes=payload + (4 << 20),
                metadata_bytes=(scenes + 10) * 65536)


def cache_plan(binding):
    """No label/teacher/metric access; validates real immutable input identities."""
    from starVLA.dataloader.st_world_observation import STObservationDataset
    from tools.st_world_policy.checkpoints import checkpoint_identity
    validate_binding(binding)
    source_runtime(binding['source_sha'], binding['environment'], binding['device'], configure_device=False)
    pinned(binding['training_registration'])
    cfg, base, checkpoint = checkpoint_identity(binding['training_registration']['path'], binding['checkpoint_tag'])
    if (checkpoint['sha256'] != binding['checkpoint_identity'] or checkpoint['completed'] != 5000
            or checkpoint['scope'] != 'FORMAL' or cfg['space_time']['cell'] != binding['arm']
            or cfg['data'].get('history_root')
            or binding['hidden_dim'] != base['parent_identity']['config']['framework']['qwenvl']['vl_hidden_dim']):
        raise ValueError('Checkpoint endpoint/arm/current-only identity differs')
    obs = binding['observations']
    data = STObservationDataset(obs['current_root'], obs['manifest']['path'])
    if data.identity['identity'] != obs['identity']:
        raise ValueError('Registered observation/image identity differs')
    signature = seal(dict(schema='st_frozen_spatial_w_v1', binding=binding, checkpoint=checkpoint,
                          observations=data.identity, rows=data.rows,
                          precision='FP32_native_master_TF32_off', targets_read=False,
                          memory_mask='all learned W slots valid; no label-derived mask'))
    return signature, data, cache_requirements(len(data), binding['memory_tokens'], binding['hidden_dim'])


@torch.no_grad()
def encode_world(model, observations, *, tokens, hidden):
    """Observation-only seam. No targets argument, no training forward or readout."""
    if not observations or any(type(x) is not ModelInputs or x.history is not None for x in observations):
        raise ValueError('Current-only ModelInputs required; labels are not an input')
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('W extraction requires a frozen eval model')
    tensors = dict(model.named_parameters()) | dict(model.named_buffers())
    before = {k: (v.data_ptr(), v._version) for k, v in tensors.items()}
    world = model.encode_inputs(observations)['W']
    if before != {k: (v.data_ptr(), v._version) for k, v in
                  (dict(model.named_parameters()) | dict(model.named_buffers())).items()}:
        raise ValueError('Frozen model parameter/buffer changed during extraction')
    if (world.shape != (len(observations), tokens, hidden) or world.dtype != torch.float32
            or not torch.isfinite(world).all()):
        raise ValueError('Canonical FP32 finite W shape differs')
    return world.detach().cpu().contiguous()


def read_world(root, signature, row):
    from safetensors.torch import load_file
    path = Path(root) / (row['token'] + '.safetensors')
    meta = json.loads(path.with_suffix('.json').read_text())
    if (meta.get('identity') != signature['identity'] or meta.get('token') != row['token']
            or meta.get('log') != row['log'] or meta.get('sha256') != file_sha256(path)):
        raise ValueError('W scene/hash/checkpoint identity differs')
    tensors = load_file(str(path))
    cfg = signature['binding']
    if set(tensors) != {'W', 'memory_valid'}:
        raise ValueError('Foreign W payload')
    world, valid = tensors['W'], tensors['memory_valid']
    if (world.shape != (cfg['memory_tokens'], cfg['hidden_dim']) or world.dtype != torch.float32
            or world.requires_grad or not torch.isfinite(world).all()
            or valid.shape != world.shape[:1] or valid.dtype != torch.bool or not valid.all()):
        raise ValueError('Invalid current learned W/memory mask')
    return world, valid


@contextmanager
def run_lock(path):
    with Path(path).open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def extract(binding, output, storage_policy):
    from safetensors.torch import save_file
    from tools.st_world_policy.checkpoints import load_world_checkpoint
    started = time.monotonic()
    signature, data, required = cache_plan(binding)
    root = Path(output).resolve()
    reservation = StorageBudget(Path(storage_policy).resolve()).reserve(root, signature['identity'], **required)
    root.mkdir(parents=True, exist_ok=True)
    with run_lock(root / 'RUN.lock'):
        with reservation.writing(json_write_bound(signature)):
            immutable_json(root / 'identity.json', signature)
        pending = []
        for index, row in enumerate(data.rows):
            if (root / (row['token'] + '.json')).exists():
                read_world(root, signature, row)
            else:
                pending.append(index)
        model = None
        if pending:
            source_runtime(binding['source_sha'], binding['environment'], binding['device'])
            model, checkpoint = load_world_checkpoint(binding['training_registration']['path'],
                    binding['checkpoint_tag'], device=binding['device'], strip=True)
            if checkpoint != signature['checkpoint']:
                raise ValueError('Checkpoint changed during load')
            model.eval().requires_grad_(False)
        completed = len(data) - len(pending)
        for index in pending:
            if time.monotonic() - started >= binding['max_seconds']:
                break
            row = data.rows[index]
            world = encode_world(model, [data[index]], tokens=binding['memory_tokens'], hidden=binding['hidden_dim'])[0]
            target = root / (row['token'] + '.safetensors')
            tmp = target.with_suffix('.tmp')
            with reservation.writing(world.numel() * 4 + len(world) + 8192):
                save_file({'W': world, 'memory_valid': torch.ones(len(world), dtype=torch.bool)}, str(tmp))
                os.replace(tmp, target)
            meta = dict(identity=signature['identity'], token=row['token'], log=row['log'], sha256=file_sha256(target))
            with reservation.writing(json_write_bound(meta)):
                immutable_json(target.with_suffix('.json'), meta)
            completed += 1
        result = dict(identity=signature['identity'], requested=len(data), completed=completed,
                      status='COMPLETE' if completed == len(data) else 'PARTIAL', scientific_result=False)
        if completed == len(data):
            with reservation.writing(json_write_bound(result)):
                immutable_json(root / 'COMPLETE.json', result)
        return result


def make_probe(hidden, seed=42):
    with initialization_seed(seed):
        return SparseFeatureReadout(hidden, 2, views=3, dim=128, layers=1, heads=4, allowed_times_s=(0.,)).float()


def queries(indices):
    """B,3,P row-major native-grid indices -> B,3P view/xy/t=0."""
    if indices.dtype != torch.long or indices.ndim != 3 or indices.shape[1] != 3 or not indices.shape[2]:
        raise ValueError('Expected B,3,P native integer patch indices')
    if ((indices < 0) | (indices >= 432)).any():
        raise ValueError('Patch outside native18x24 grid')
    view = torch.arange(3, device=indices.device)[None, :, None].expand_as(indices).flatten(1)
    x = (indices % 24 + .5) * (2. / 24) - 1.
    y = (indices // 24 + .5) * (2. / 18) - 1.
    xy = torch.stack((x, y), -1).flatten(1, 2)
    return view, xy, torch.zeros_like(view, dtype=torch.float32)


class PatchStream:
    """The same label-independent scene/patch exposure for every arm."""
    def __init__(self, scenes, seed=42):
        if type(scenes) is not int or scenes < 1:
            raise ValueError('Positive scene population required')
        self.scenes, self.completed = scenes, 0
        self.generators = {k: torch.Generator().manual_seed(seed + i) for i, k in enumerate(('scene', 'patch'))}

    def next(self):
        scene = torch.randint(self.scenes, (2,), generator=self.generators['scene'])
        patch = torch.stack([torch.randperm(432, generator=self.generators['patch'])[:32]
                             for _ in range(6)]).reshape(2, 3, 32)
        self.completed += 1
        return scene, patch

    def state_dict(self):
        return dict(scenes=self.scenes, completed=self.completed,
                    generators={k: g.get_state() for k, g in self.generators.items()})

    def load_state_dict(self, state):
        if state['scenes'] != self.scenes or set(state['generators']) != set(self.generators):
            raise ValueError('Sample stream inventory changed')
        self.completed = state['completed']
        for key, generator in self.generators.items():
            generator.set_state(state['generators'][key].cpu())


def capture_rng(device):
    # Same inspected formula as joint_local_scene_v3.train_mechanism.rng_state;
    # local CPU-conditional wrapper avoids importing that unrelated trainer.
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


def restore_rng(state, device):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'].cpu())
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'].cpu(), device)


def validate_labels(labels):
    if set(labels) != {'depth', 'depth_valid', 'small_vehicle', 'small_valid'}:
        raise ValueError('Unexpected raw label schema')
    if (labels['depth'].ndim not in (3, 4)
            or any(x.shape != labels['depth'].shape or x.shape[-3:] != (3, 18, 24) for x in labels.values())):
        raise ValueError('Labels must retain native view,height,width')
    for key in ('depth_valid', 'small_valid'):
        if labels[key].dtype != torch.bool:
            raise ValueError('Label validity must be Boolean')
    if any(labels[k].dtype != torch.float32 for k in ('depth', 'small_vehicle')):
        raise ValueError('Raw target tensors must be FP32')
    depth = labels['depth'][labels['depth_valid']]
    small = labels['small_vehicle'][labels['small_valid']]
    if (not torch.isfinite(depth).all() or ((depth < 1) | (depth > 80)).any()
            or not torch.isfinite(small).all() or ((small != 0) & (small != 1)).any()):
        raise ValueError('Invalid valid raw-GT values')


def train_statistics(labels):
    validate_labels(labels)
    depth = labels['depth'][labels['depth_valid']].double().log()
    small = labels['small_vehicle'][labels['small_valid']].double()
    positive, negative = int(small.sum()), int((small == 0).sum())
    if not depth.numel() or not positive or not negative:
        raise ValueError('Fixed train population lacks valid depth or both small classes')
    return dict(log_mean=float(depth.mean()), log_std=max(float(depth.std(unbiased=False)), 1e-6),
                pos_weight=negative / positive, depth_valid=depth.numel(), positive=positive, negative=negative)


def sampled_labels(labels, scene, patch, device):
    return {k: v[scene].flatten(2).gather(2, patch).flatten(1).to(device) for k, v in labels.items()}


def task_loss(prediction, target, stats):
    if prediction.shape != (*target['depth'].shape, 2) or not torch.isfinite(prediction).all():
        raise ValueError('Invalid probe prediction')
    zero = prediction.sum() * 0.
    losses, counts = {}, {}
    for task, maskkey in (('depth', 'depth_valid'), ('small', 'small_valid')):
        mask = target[maskkey]
        if mask.dtype != torch.bool or mask.shape != prediction.shape[:-1]:
            raise ValueError('Task validity shape/type differs')
        counts[task] = int(mask.sum())
        if not counts[task]:
            losses[task] = zero
        elif task == 'depth':
            if not torch.isfinite(target['depth'][mask]).all() or (target['depth'][mask] <= 0).any():
                raise ValueError('Invalid valid depth target')
            expected = (target['depth'][mask].log() - stats['log_mean']) / stats['log_std']
            losses[task] = F.smooth_l1_loss(prediction[..., 0][mask], expected, beta=.1, reduction='sum') / counts[task]
        else:
            values = target['small_vehicle'][mask]
            if not torch.isfinite(values).all() or ((values != 0) & (values != 1)).any():
                raise ValueError('Invalid valid binary target')
            losses[task] = F.binary_cross_entropy_with_logits(prediction[..., 1][mask], target['small_vehicle'][mask],
                    pos_weight=prediction.new_tensor(stats['pos_weight']), reduction='sum') / counts[task]
    return sum(losses.values()), losses, counts


def update_probe(model, optimizer, world, memory_valid, labels, stats, stream):
    scene, patch = stream.next()
    device = next(model.parameters()).device
    w = world[scene].detach().to(device)
    valid = memory_valid[scene].to(device)
    pred = model(w, *queries(patch.to(device)), memory_valid=valid)
    loss, losses, counts = task_loss(pred, sampled_labels(labels, scene, patch, device), stats)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if sum(counts.values()):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    return dict(update=stream.completed, scene_indices=scene.tolist(),
                patch_sha256=tensor_hash(patch), queried_patches=int(patch.numel()), valid=counts,
                losses={k: float(v.detach()) for k, v in losses.items()}, optimizer_step=bool(sum(counts.values())))


def checkpoint_state(model, optimizer, stream, run_identity, exposure):
    return dict(schema='st_spatial_probe_state_v1', identity=run_identity, model=model.state_dict(),
                optimizer=optimizer.state_dict(), stream=stream.state_dict(), exposure=exposure,
                rng=capture_rng(next(model.parameters()).device))


def restore_state(state, model, optimizer, stream, run_identity):
    if state.get('schema') != 'st_spatial_probe_state_v1' or state.get('identity') != run_identity:
        raise ValueError('Probe checkpoint identity differs')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    stream.load_state_dict(state['stream'])
    if len(state['exposure']) != stream.completed:
        raise ValueError('Checkpoint sample cursor/exposure differs')
    restore_rng(state['rng'], next(model.parameters()).device)
    return state['exposure']


def binary_metrics(logits, truth):
    from sklearn.metrics import average_precision_score, roc_auc_score
    logits, truth = np.asarray(logits), np.asarray(truth, dtype=bool)
    positive, negative = int(truth.sum()), int((~truth).sum())
    result = dict(positive=positive, negative=negative, valid=len(truth), threshold_logit=0.,
                  PR_AUC=None, ROC_AUC=None, balanced_accuracy=None)
    if positive and negative:
        pred = logits > 0.
        result.update(PR_AUC=float(average_precision_score(truth, logits)), ROC_AUC=float(roc_auc_score(truth, logits)),
                      balanced_accuracy=float((pred[truth].mean() + (~pred[~truth]).mean()) / 2))
    return result


def depth_metrics(prediction, truth):
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    if not len(truth):
        return dict(valid=0, MAE_m=None, RMSE_m=None, AbsRel=None)
    error = prediction - truth
    return dict(valid=len(truth), MAE_m=float(np.abs(error).mean()), RMSE_m=float(np.sqrt((error ** 2).mean())),
                AbsRel=float((np.abs(error) / truth).mean()))


@torch.no_grad()
def evaluate(model, world, memory_valid, labels, stats, rows, *, deadline=None):
    model.eval()
    predictions = []
    device = next(model.parameters()).device
    grid = torch.arange(432, device=device).reshape(1, 1, 432).expand(1, 3, 432)
    for i in range(len(world)):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('Evaluation wall budget reached; resume the saved final probe')
        predictions.append(model(world[i:i+1].to(device), *queries(grid),
                    memory_valid=memory_valid[i:i+1].to(device)).cpu().reshape(3, 18, 24, 2))
    pred = torch.stack(predictions)
    depth = (pred[..., 0].double() * stats['log_std'] + stats['log_mean']).exp()
    if not torch.isfinite(pred).all() or not torch.isfinite(depth).all():
        raise ValueError('Nonfinite evaluation; no scene may be dropped')
    def metrics(selection):
        dm = labels['depth_valid'][selection]; sm = labels['small_valid'][selection]
        d = depth[selection][dm].numpy(); y = labels['depth'][selection][dm].numpy()
        result = dict(depth=depth_metrics(d, y), small=binary_metrics(pred[..., 1][selection][sm].numpy(),
                      labels['small_vehicle'][selection][sm].numpy()), patches=int(dm.numel()),
                      depth_unknown=int((~dm).sum()), small_unknown=int((~sm).sum()))
        result['depth_ranges_m'] = {name: depth_metrics(d[(y >= lo) & (y < hi)], y[(y >= lo) & (y < hi)])
                for name, lo, hi in (('1_to_10', 1, 10), ('10_to_30', 10, 30), ('30_to_80_inclusive', 30, 81))}
        return result
    return dict(overall=metrics(slice(None)), views={name: metrics((slice(None), i)) for i, name in enumerate(VIEWS)},
                scenes=[dict(token=r['token'], log=r['log'], **metrics(i)) for i, r in enumerate(rows)],
                logs={log: metrics([i for i, r in enumerate(rows) if r['log'] == log]) for log in sorted({r['log'] for r in rows})})


def load_protocol(ref):
    p = identity(pinned(ref))
    exact(p, ('schema', 'identity', 'recipe', 'populations', 'labels', 'label_schema', 'interpretation',
              'common_hidden_dim'), 'probe protocol')
    if p['schema'] != 'st_learned_w_probe_protocol_v1' or p['recipe'] != RECIPE or p['label_schema'] != LABEL_SCHEMA:
        raise ValueError('Unregistered spatial probe recipe/schema')
    if type(p['common_hidden_dim']) is not int or p['common_hidden_dim'] < 1:
        raise ValueError('A common hidden dimension must be registered across all arms')
    if set(p['populations']) != {'train', 'dev'}:
        raise ValueError('Only train/dev protocol populations')
    rows = {s: checked_rows(pinned(p['populations'][s]), s, n) for s, n in (('train', 64), ('dev', 16))}
    if {r['log'] for r in rows['train']} & {r['log'] for r in rows['dev']}:
        raise ValueError('Train/dev log leakage')
    labels = pinned(p['labels'])
    if labels['label_schema'] != LABEL_SCHEMA:
        raise ValueError('Raw label manifest schema differs')
    return p, rows, labels


def load_fit_data(reg):
    protocol, rows, label_manifest = load_protocol(reg['protocol'])
    data, cache_signatures = {}, {}
    for split in ('train', 'dev'):
        ref = reg['caches'][split]
        exact(ref, ('root', 'identity'), 'W cache reference')
        root = Path(ref['root']); sig = identity(json.loads((root / 'identity.json').read_text()))
        validate_binding(sig['binding'])
        if (sig['identity'] != ref['identity'] or sig['schema'] != 'st_frozen_spatial_w_v1'
                or sig['rows'] != rows[split] or sig['binding']['arm'] != reg['arm']
                or sig['binding']['observations']['manifest'] != protocol['populations'][split]
                or sig['observations']['split'] != split or sig['checkpoint']['completed'] != 5000
                or sig['binding']['checkpoint_identity'] != sig['checkpoint']['sha256']
                or sig['binding']['hidden_dim'] != protocol['common_hidden_dim']
                or sig['targets_read'] is not False):
            raise ValueError('Frozen cache population/arm/checkpoint differs')
        complete = json.loads((root / 'COMPLETE.json').read_text())
        if (complete['identity'] != sig['identity'] or complete['status'] != 'COMPLETE'
                or complete['completed'] != len(rows[split]) or complete['requested'] != len(rows[split])):
            raise ValueError('Incomplete frozen W population')
        index = {r['token']: r for r in label_manifest['roles'][split]}
        worlds, masks, labels = [], [], []
        for row in rows[split]:
            source = index[row['token']]
            if source['log'] != row['log']:
                raise ValueError('Label scene/log differs')
            label_path = Path(protocol['labels']['path']).parent / 'labels' / split / (row['token'] + '.npz')
            if file_sha256(label_path) != source['label_sha256']:
                raise ValueError('Raw label hash differs')
            with np.load(label_path, allow_pickle=False) as arrays:
                y = {k: torch.from_numpy(arrays[k].copy()) for k in arrays.files}
            validate_labels(y)
            w, m = read_world(root, sig, row); worlds.append(w); masks.append(m); labels.append(y)
        data[split] = (torch.stack(worlds), torch.stack(masks), {k: torch.stack([y[k] for y in labels]) for k in labels[0]})
        cache_signatures[split] = sig
    a, b = cache_signatures.values()
    if a['checkpoint'] != b['checkpoint'] or a['binding']['hidden_dim'] != b['binding']['hidden_dim']:
        raise ValueError('Train/dev W checkpoint/hidden dimension differs')
    return protocol, rows, data, cache_signatures


def fit(reg, output, *, resume=False):
    identity(reg)
    exact(reg, ('schema', 'identity', 'protocol', 'caches', 'arm', 'source_sha', 'environment', 'device',
                'max_seconds', 'max_output_bytes'), 'fit registration')
    if (reg['schema'] != 'st_spatial_probe_fit_v1' or set(reg['caches']) != {'train', 'dev'}
            or not isinstance(reg['max_seconds'], (int, float)) or not math.isfinite(reg['max_seconds'])
            or not 0 < reg['max_seconds'] <= 7200 or type(reg['max_output_bytes']) is not int
            or not 0 < reg['max_output_bytes'] <= 64 << 20):
        raise ValueError('Bounded probe registration required')
    source_runtime(reg['source_sha'], reg['environment'], reg['device'])
    started = time.monotonic()
    protocol, rows, data, signatures = load_fit_data(reg)
    stats = train_statistics(data['train'][2])
    hidden = data['train'][0].shape[-1]
    model = make_probe(hidden).to(reg['device'])
    tensor_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    # Two durable slots plus one staged serialization; reject before updates.
    if tensor_bytes * 12 + (4 << 20) > reg['max_output_bytes']:
        raise ValueError('Probe output/staging exceeds registered byte budget')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    stream = PatchStream(len(rows['train']))
    manifest = seal(dict(schema='st_spatial_probe_fit_manifest_v1', registration=reg, statistics=stats,
                         initialization={k: tensor_hash(v) for k, v in model.state_dict().items()},
                         parameter_count=sum(p.numel() for p in model.parameters()),
                         scientific_result=False, final_update=300, protocol_identity=protocol['identity']))
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    allowed_files = {'RUN.lock', 'MANIFEST.json', 'probe_a.pt', 'probe_b.pt', 'probe.tmp',
                     'STATE.json', 'STATE.tmp', 'RESULT.json'}
    if any(not x.is_file() or x.is_symlink() or x.name not in allowed_files for x in root.iterdir()):
        raise ValueError('Probe directory contains foreign/unbudgeted artifacts')
    if sum(x.stat().st_size for x in root.iterdir()) > reg['max_output_bytes']:
        raise ValueError('Existing probe directory exceeds byte budget')
    with run_lock(root / 'RUN.lock'):
        immutable_json(root / 'MANIFEST.json', manifest)
        receipt_path = root / 'STATE.json'
        if receipt_path.exists() != resume:
            raise ValueError('Use explicit resume only with an existing probe checkpoint')
        exposure = []
        active_slot = None
        if resume:
            receipt = json.loads(receipt_path.read_text())
            active_slot = receipt['file']
            if active_slot not in ('probe_a.pt', 'probe_b.pt'):
                raise ValueError('Unknown probe checkpoint slot')
            checkpoint = root / active_slot
            if receipt['identity'] != manifest['identity'] or file_sha256(checkpoint) != receipt['sha256']:
                raise ValueError('Probe checkpoint receipt/hash changed')
            exposure = restore_state(torch.load(checkpoint, map_location=reg['device'], weights_only=False),
                                     model, optimizer, stream, manifest['identity'])
            if receipt['completed'] != stream.completed or not 0 <= stream.completed <= 300:
                raise ValueError('Probe committed endpoint/cursor differs')
        def save():
            nonlocal active_slot
            tmp = root / 'probe.tmp'
            state = checkpoint_state(model, optimizer, stream, manifest['identity'], exposure)
            with tmp.open('wb') as output_stream:
                torch.save(state, output_stream)
                output_stream.flush(); os.fsync(output_stream.fileno())
            if tmp.stat().st_size * 3 + (4 << 20) > reg['max_output_bytes']:
                raise ValueError('Serialized probe exceeds byte budget')
            new_slot = 'probe_b.pt' if active_slot == 'probe_a.pt' else 'probe_a.pt'
            checkpoint = root / new_slot
            os.replace(tmp, checkpoint)
            receipt = dict(identity=manifest['identity'], file=new_slot,
                           sha256=file_sha256(checkpoint), completed=stream.completed)
            # Pointer publishes only after new slot is durable; prior slot survives interruption.
            receipt_tmp = root / 'STATE.tmp'
            with receipt_tmp.open('w') as output_stream:
                json.dump(receipt, output_stream, sort_keys=True)
                output_stream.flush(); os.fsync(output_stream.fileno())
            os.replace(receipt_tmp, receipt_path)
            directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            active_slot = new_slot
        frozen_hashes = {s: tensor_hash(data[s][0]) for s in data}
        model.train()
        while stream.completed < 300:
            if time.monotonic() - started >= reg['max_seconds']:
                break
            exposure.append(update_probe(model, optimizer, *data['train'], stats, stream))
            if stream.completed % 25 == 0:
                save()
        save()
        if frozen_hashes != {s: tensor_hash(data[s][0]) for s in data}:
            raise ValueError('Frozen W changed during probe training')
        result = dict(identity=manifest['identity'], completed=stream.completed,
                      status='COMPLETE' if stream.completed == 300 else 'PARTIAL', scientific_result=False)
        if stream.completed == 300:
            result.update(exposure=exposure, statistics=stats,
                          metrics={s: evaluate(model, *data[s], stats, rows[s], deadline=started+reg['max_seconds'])
                                   for s in ('train', 'dev')},
                          interpretation=protocol['interpretation'], frozen_W_unchanged=True)
            immutable_json(root / 'RESULT.json', result)
        return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('plan-cache', 'extract'):
        q = sub.add_parser(name, help='Validate pinned native inputs' if name == 'plan-cache' else 'Extract budgeted frozen W')
        q.add_argument('--binding', required=True)
        q.add_argument('--output', required=True)
        if name == 'extract':
            q.add_argument('--storage-policy', required=True)
    q = sub.add_parser('fit', help='Train only a fresh bounded spatial readout from frozen W')
    q.add_argument('--registration', required=True); q.add_argument('--output', required=True)
    q.add_argument('--resume', action='store_true')
    return p


def main():
    a = parser().parse_args()
    try:
        if a.command == 'fit':
            result = fit(json.loads(Path(a.registration).read_text()), a.output, resume=a.resume)
        else:
            binding = json.loads(Path(a.binding).read_text())
            if a.command == 'extract':
                result = extract(binding, a.output, a.storage_policy)
            else:
                signature, _, required = cache_plan(binding)
                result = dict(status='PLANNED_ONLY', cache_identity=signature['identity'], root=str(Path(a.output).resolve()),
                              **required, payload_total_bytes=required['output_bytes'], no_cache_written=True)
        print(json.dumps({k: v for k, v in result.items() if k not in ('metrics', 'exposure')}, allow_nan=False))
        return 0 if result['status'] in ('COMPLETE', 'PLANNED_ONLY') else 2
    except Exception as exc:
        print(json.dumps(dict(status='BLOCKED_SPATIAL_PROBE', error=repr(exc), scientific_result=False)))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
