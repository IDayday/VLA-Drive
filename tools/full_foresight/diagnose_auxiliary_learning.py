"""Read-only, fixed-population diagnostics of learned auxiliary tasks, not PDMS."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from starVLA.dataloader.full_foresight_dataset import FullForesightDataset
from starVLA.model.modules.trajectory_mae.model import TrajectoryMAE
from starVLA.model.modules.vehicle_joint.initialization import file_sha256, identity_hash
from tools.foresight.checkpoints import checkpoint_identity, load_student, scene_noise
from tools.foresight.teacher_runtime import TeacherDataset, inputs
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.run_meter import metered_run
from tools.full_foresight.evaluate_auxiliary import spatial_statistics

HORIZONS = (0, 1, 2, 4)


def selected_indices(index, per_log):
    """Data-only hash selection; retain every log, no model/target error selection."""
    groups = {}
    for i, row in enumerate(index):
        groups.setdefault(row['log'], []).append(i)
    key = lambda i: hashlib.sha256(('auxiliary-diagnostic-v1:' + index[i]['token']).encode()).hexdigest()
    return sorted(i for ids in groups.values() for i in sorted(ids, key=key)[:per_log])


def label_images(ds, i):
    features, masks = [], []
    for row in ds.dino_scenes[i]['images']:
        pairs = [ds.read_image(j) for j in row]
        features.append(torch.stack([p[0] for p in pairs]))
        masks.append(torch.stack([p[1] for p in pairs]))
    return torch.stack(features).float(), torch.stack(masks)


def feature_error(prediction, target, valid):
    active = valid.unsqueeze(-3).expand_as(target)
    if prediction.shape != target.shape or valid.dtype != torch.bool:
        raise ValueError('Invalid feature comparison')
    if not torch.isfinite(prediction[active]).all() or not torch.isfinite(target[active]).all():
        raise ValueError('Nonfinite valid feature')
    # Mask BEFORE subtraction: invalid NaNs must not affect valid metrics.
    delta = torch.where(active, prediction.float(), 0.) - torch.where(active, target.float(), 0.)
    return {'squared_error': float(delta.square().sum()), 'elements': int(active.sum())}


def trajectory_error(prediction, target, valid):
    if valid.dtype != torch.bool or valid.shape != target.shape[:-1] or prediction.shape != target.shape:
        raise ValueError('Invalid trajectory comparison')
    if not torch.isfinite(prediction[valid]).all() or not torch.isfinite(target[valid]).all():
        raise ValueError('Nonfinite valid trajectory')
    ids = torch.where(valid)[0]
    if not len(ids):
        return {'valid_points': 0, 'ADE': None, 'endpoint_FDE': None}
    d = (prediction[valid].float() - target[valid].float()).norm(dim=-1)
    return {'valid_points': len(ids), 'ADE': float(d.mean()),
            'endpoint_FDE': float(d[-1]) if bool(valid[-1]) else None}


class AcrossSceneVariance:
    """Variance across scenes at EACH fixed feature coordinate, not across channels."""
    def __init__(self):
        self.sum = self.square = self.count = None

    def add(self, value, valid):
        active = valid.unsqueeze(-3).expand_as(value)
        if not torch.isfinite(value[active]).all():
            raise ValueError('Nonfinite variance input')
        clean = torch.where(active, value.double(), 0.).cpu()
        if self.sum is None:
            self.sum = torch.zeros_like(clean)
            self.square = torch.zeros_like(clean)
            self.count = torch.zeros_like(clean)
        self.sum += clean
        self.square += clean.square()
        self.count += active.cpu()

    def result(self):
        if self.sum is None:
            return None
        eligible = self.count > 1
        variance = (self.square / self.count.clamp_min(1) -
                    (self.sum / self.count.clamp_min(1)).square()).clamp_min(0)
        return float(variance[eligible].mean()) if eligible.any() else None


def dataset(root, cache, index, interactions, candidate):
    di = json.loads((Path(cache)/'identity.json').read_text())
    ii = json.loads((Path(interactions)/'identity.json').read_text())
    return FullForesightDataset(root, candidate=candidate, current=True, future=True,
        dino_root=cache, dino_index=index, expected_dino=di['identity'], allow_partial=False,
        interaction_root=interactions, expected_interaction=ii['identity'])


def summarize(rows, requested, variances):
    result = {'requested': requested, 'completed': len(rows),
              'failed': sum(r['failure'] is not None for r in rows),
              'scope': 'fixed log-stratified development subset; no planning score or training updates',
              'visual': {}, 'interaction': {}, 'offline_ego_probes': {}}
    for h in HORIZONS:
        items = [r['visual'][str(h)] for r in rows if 'visual' in r]
        n = sum(x['elements'] for x in items)
        if not n:
            continue
        sums = {key: sum(x[key] for x in items) for key in
                ('squared_error', 'template_squared_error', 'copy_current_squared_error')}
        record = {key.replace('squared_error', 'mse'): value/n for key, value in sums.items()}
        record.update(elements=n, valid_scenes=sum(x['elements'] > 0 for x in items),
            prediction_across_scene_variance=variances[str(h)]['prediction'].result(),
            target_across_scene_variance=variances[str(h)]['target'].result())
        for name in ('template', 'copy_current'):
            den = sums[name+'_squared_error']
            record['relative_improvement_over_'+name] = 1-sums['squared_error']/den if den > 0 else None
        result['visual'][str(h)] = record
    items = [r['interaction'] for r in rows if 'interaction' in r and r['interaction']['valid']]
    result['interaction'] = {'valid_scenes': len(items),
        'normalized_mse': sum(r['mse'] for r in items)/len(items) if items else None,
        'train_mean_normalized_mse': sum(r['template_mse'] for r in items)/len(items) if items else None}
    for group in ('all', 'with_peer', 'without_peer'):
        subset = [r for r in rows if 'ego_probes' in r and
                  (group == 'all' or r['interaction']['valid'] == (group == 'with_peer'))]
        out = {'scenes': len(subset)}
        for key in ('student_latent_decoded', 'teacher_latent_decoded', 'teacher_current_only', 'action', 'stationary'):
            for metric in ('ADE', 'endpoint_FDE'):
                values = [r['ego_probes'][key][metric] for r in subset if r['ego_probes'][key][metric] is not None]
                out[key+'_'+metric] = sum(values)/len(values) if values else None
        result['offline_ego_probes'][group] = out
    result['complete'] = len(rows) == requested and result['failed'] == 0
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    for key in ('training-run', 'checkpoint-tag', 'dev-data', 'train-data', 'dino-root', 'dino-index',
                'dev-interactions', 'train-interactions', 'teacher-root', 'teacher-data',
                'frozen-teacher', 'output', 'campaign-root', 'run-id'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--per-log', type=int, default=8)
    p.add_argument('--template-scenes', type=int, default=64)
    p.add_argument('--max-seconds', type=float, default=3600)
    p.add_argument('--memory-fraction', type=float, default=.45)
    a = p.parse_args()
    if min(a.per_log, a.template_scenes, a.max_seconds) <= 0 or not 0 < a.memory_fraction <= 1:
        raise ValueError('Invalid diagnostic bound')
    if subprocess.check_output(['git', 'status', '--porcelain']).strip():
        raise ValueError('Lock diagnostic source before use')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    with metered_run(a.campaign_root, a.run_id, 1, {'kind': 'read_only_auxiliary_diagnostic',
            'real_optimizer_updates': 0, 'max_seconds': a.max_seconds}) as (meter, _, save):
        train_id, checkpoint = checkpoint_identity(a.training_run, a.checkpoint_tag)
        candidate = train_id['candidate']['name']
        dev = dataset(a.dev_data, a.dino_root, a.dino_index, a.dev_interactions, candidate)
        train = dataset(a.train_data, a.dino_root, a.dino_index, a.train_interactions, candidate)
        if dev.identity['split'] != 'dev' or train.identity['split'] != 'train':
            raise ValueError('No test-set calibration or diagnostics')
        if {r['log'] for r in dev.index} & {r['log'] for r in train.index}:
            raise ValueError('Train/dev log leakage')
        selected = selected_indices(dev.index, a.per_log)
        template_ids = sorted(range(len(train)), key=lambda i: hashlib.sha256(
            ('auxiliary-template-v1:'+train.index[i]['token']).encode()).hexdigest())[:a.template_scenes]
        out = Path(a.output)
        out.mkdir(parents=True, exist_ok=False)
        frozen = json.loads(Path(a.frozen_teacher).read_text())
        teacher_id = json.loads((Path(a.teacher_root)/'identity.json').read_text())
        teacher_path = Path(a.teacher_root)/frozen['checkpoint']
        if file_sha256(teacher_path) != frozen['checkpoint_sha256'] or teacher_id['identity'] != frozen['teacher_run_identity']:
            raise ValueError('Frozen teacher changed')
        if dev.interaction_identity['frozen_teacher_identity'] != frozen['identity']:
            raise ValueError('Wrong cached teacher')
        identity = {'source': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'checkpoint': checkpoint, 'data': dev.identity['identity'], 'dino': dev.dino_identity['identity'],
            'teacher': frozen, 'dev_scenes': [dev.index[i] for i in selected],
            'template_train_scenes': [train.index[i] for i in template_ids],
            'selection': f'{a.per_log} per log by fixed SHA256 of scene token; no label/model error ranking',
            'precision': 'FP32 masters, FP32 compute, TF32 disabled',
            'memory_fraction': a.memory_fraction, 'max_seconds': a.max_seconds,
            'real_optimizer_updates': 0}
        atomic_json(out/'identity.json', identity)
        # Training-only, horizon-specific fixed spatial templates.
        sums = counts = None
        zsum = None
        zn = 0
        for i in template_ids:
            z, valid = label_images(train, i)
            active = valid[:, :, None].expand_as(z)
            if not torch.isfinite(z[active]).all():
                raise ValueError('Nonfinite training template feature')
            clean = torch.where(active, z, 0.).double()
            if sums is None:
                sums, counts = torch.zeros_like(clean), torch.zeros_like(clean)
            sums += clean
            counts += active
            label = torch.load(Path(a.train_interactions)/'targets'/(train.index[i]['token']+'.pt'), weights_only=True)
            if label['identity'] != train.interaction_identity['identity']:
                raise ValueError('Foreign template interaction label')
            if label['interaction_target_valid']:
                normalized = F.layer_norm(label['latent'].float(), (512,), eps=1e-5)
                zsum = normalized if zsum is None else zsum + normalized
                zn += 1
        if zn == 0 or (counts == 0).any():
            raise ValueError('Insufficient training reference coverage')
        template, ztemplate = (sums/counts).float().cuda(), (zsum/zn).cuda()
        atomic_json(out/'reference.json', {'training_scenes': len(template_ids), 'valid_interaction_scenes': zn,
            'template_counts_min': float(counts.min()), 'uses_validation_labels_to_fit': False})
        td = TeacherDataset(a.teacher_data, 'dev')
        teacher_indices = {r['token']: i for i, r in enumerate(td.index)}
        if set(teacher_indices) != {r['token'] for r in dev.index}:
            raise ValueError('Teacher/student development population mismatch')
        teacher = TrajectoryMAE(**teacher_id['model'])
        saved = torch.load(teacher_path, map_location='cpu', weights_only=False)
        if saved['identity'] != teacher_id['identity']:
            raise ValueError('Foreign teacher checkpoint')
        teacher.load_state_dict(saved['model'], strict=True)
        del saved
        teacher.requires_grad_(False).eval().cuda()
        model = load_student(a.training_run, a.checkpoint_tag, train_id, strip=False)
        variances = {str(h): {k: AcrossSceneVariance() for k in ('prediction','target')} for h in HORIZONS}
        rows = []
        path = out/'scenes.jsonl'
        for i in selected:
            if time.time()-meter['start_unix'] > a.max_seconds:
                meter['status'] = 'PAUSED'
                break
            row = {**dev.index[i], 'failure': None}
            try:
                observation, targets = dev[i]
                with torch.inference_mode():
                    encoded = model.encode_current([observation])
                    world = encoded['W']
                    current = targets['current_dino'].cuda()
                    row['visual'] = {}
                    for hi, h in enumerate(HORIZONS):
                        target = current if h == 0 else targets['future_dino'][hi-1].cuda()
                        valid = targets['current_dino_valid'].cuda() if h == 0 else targets['future_dino_valid'][hi-1].cuda()
                        pred = model.dino_head(world, torch.tensor([float(h)], device='cuda'), target.shape[-2:])[0]
                        views = spatial_statistics(pred, target, current, valid)
                        record = {key: sum(v[key] for v in views) for key in views[0]}
                        ref = feature_error(template[hi], target, valid)
                        record['template_squared_error'] = ref['squared_error']
                        row['visual'][str(h)] = record
                        variances[str(h)]['prediction'].add(pred, valid)
                        variances[str(h)]['target'].add(target, valid)
                        # Small native-grid diagnostic maps; never label them RGB reconstructions.
                        if len(rows) < 4:
                            active = valid[:, None].expand_as(target)
                            delta = torch.where(active, pred, 0.) - torch.where(active, target, 0.)
                            atomic_json(out/f"feature_map_{row['token']}_h{h}.json", {
                                'patch_mse': delta.square().mean(1).cpu().tolist(),
                                'cosine': F.cosine_similarity(pred, target, dim=1).cpu().tolist(),
                                'grid_hw': list(target.shape[-2:]), 'views': ['CAM_F0','CAM_L0','CAM_R0']})
                    predz = model.interaction_head(world)[0].float()
                    targetz = targets['interaction_latent'].cuda().float()
                    normalized = F.layer_norm(predz, (512,), eps=1e-5)
                    normalized_target = F.layer_norm(targetz, (512,), eps=1e-5)
                    validz = bool(targets['interaction_valid'])
                    row['interaction'] = {'valid': validz,
                        'mse': float((normalized-normalized_target).square().mean()) if validz else None,
                        'template_mse': float((ztemplate-normalized_target).square().mean()) if validz else None}
                    batch = td.batch([teacher_indices[row['token']]], 'cuda')
                    target_actor = torch.zeros(1, device='cuda', dtype=torch.long)
                    empty = torch.zeros_like(batch['point_valid'])
                    current_only = teacher(inputs(batch, target_actor, empty))['xy'][0]
                    anchor = batch['current'][0, 0, :2]
                    decoded = teacher.reconstruct(predz)*teacher.xy_scale + anchor
                    oracle_decoded = teacher.reconstruct(targetz)*teacher.xy_scale + anchor
                    # Original action head uses SAME encode_current output, no extra Qwen pass.
                    noise = scene_noise(row['token'], 42, 'cuda')
                    action = model.action_model.predict_action(encoded['action_queries'], initial_noise=noise)
                    from starVLA.dataloader.foresight_dataset import decode_ego
                    action = decode_ego(action.float())[0, :, :2]
                    label, point_valid = batch['future'][0,0], batch['point_valid'][0,0]
                    row['ego_probes'] = {k: trajectory_error(x, label, point_valid) for k, x in (
                        ('student_latent_decoded', decoded), ('teacher_latent_decoded', oracle_decoded),
                        ('teacher_current_only', current_only), ('action', action),
                        ('stationary', anchor[None].expand(8,-1)))}
                    row['teacher_latent_decoded_is_privileged'] = True
                    row['student_latent_decoded_is_offline_probe_not_policy'] = True
            except Exception as error:
                row = {**dev.index[i], 'failure': repr(error)}
            rows.append(row)
            with path.open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            meter['inference_scenes'] = len(rows)
            save()
        summary = summarize(rows, len(selected), variances)
        atomic_json(out/'summary.json', summary)
        if summary['failed']:
            raise RuntimeError('Failed rows retained; diagnosis not claimed complete')


if __name__ == '__main__':
    main()
