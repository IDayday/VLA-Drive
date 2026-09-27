"""Finite true-data graph isolation; cached CURRENT features, separate future labels."""
import argparse
import hashlib
import json
import math
import pickle
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

from starVLA.model.modules.joint_world.flow import JointTrajectoryFlow, actor_mask, training_loss_sums
from starVLA.model.modules.joint_world.targets import joint_targets
from starVLA.model.modules.structured_world.contracts import WorldTargets
from starVLA.model.modules.structured_world.metrics import match_geometry, prediction_filters
from tools.joint_world.extract_conditions import CACHE_FIELDS
from tools.structured_world_v1p1.budget import start, record
from tools.structured_world_v1p1.reaudit_metrics import write_csv
from tools.structured_world.runtime import seed_all


def current_batch(data):
    return {k: torch.cat([r['cache'][k] for r in data]).cuda() for k in ['actor_features', 'context', 'current_xy', 'existence']}


def load_samples(cache, targets, data_root, steps, records=None, manifest_override=None):
    manifest = manifest_override if manifest_override is not None else json.loads((Path(cache) / 'manifest.json').read_text())
    if not manifest['identity']['frozen_upstream'] or manifest['identity']['targets_loaded']:
        raise ValueError('Only frozen current-only caches are allowed')
    samples = []; fingerprint = hashlib.sha256()
    for row in (manifest['records'] if records is None else records):
        token = row['token']; file = Path(cache) / (token + '.pt'); blob = file.read_bytes()
        if hashlib.sha256(blob).hexdigest() != row['sha256']: raise ValueError('Condition cache checksum mismatch')
        c = torch.load(file, map_location='cpu', weights_only=True)
        if set(c) != CACHE_FIELDS or c['identity_sha256'] != row.get('identity_sha256', manifest['identity_sha256']) or c['token'] != token:
            raise ValueError('Condition cache contract mismatch')
        target_file = Path(targets) / 'targets' / (token + '.pt')
        t = WorldTargets(**torch.load(target_file, map_location='cpu', weights_only=True))
        if t.overflow: raise ValueError('Use full target cache')
        meta_path = Path(data_root) / 'meta/train' / (token + '.pkl')
        with meta_path.open('rb') as f: meta = pickle.load(f)
        poses = np.asarray(meta['glo_status']['global_poses'], dtype=np.float64)
        if len(poses) < 4 + steps: raise ValueError('Missing ego horizon')
        origin = poses[3]; yaw = origin[2]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        ego = torch.tensor((poses[4:4+steps, :2] - origin[:2]) @ rot, dtype=torch.float32)[None]
        pred = {'boxes': c['current_boxes'], 'logits': c['current_logits'],
                'future_xy': c['current_boxes'].new_zeros(1, c['current_boxes'].shape[1], steps, 2)}
        xy, valid, _ = joint_targets(pred, [t], ego)
        samples.append(dict(cache=c, xy=xy, valid=valid, target=t, pred=pred))
        for path in [target_file, meta_path]: fingerprint.update(token.encode() + hashlib.sha256(path.read_bytes()).digest())
    return samples, manifest, fingerprint.hexdigest()


@torch.no_grad()
def evaluate(model, samples, output, step):
    rows = []; objects = []; model.eval()
    banks = output / ('predictions_' + str(step)); banks.mkdir(exist_ok=False)
    for sample in samples:
        c = current_batch([sample]); token = sample['cache']['token']
        generator = torch.Generator(device='cuda').manual_seed(int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], 'little'))
        noise = torch.randn(sample['xy'].shape, device='cuda', generator=generator)
        xy, _ = model.sample(noise, **c, sampling_steps=10); xy = xy.cpu()[0]
        np.savez(banks / (token + '.npz'), future_xy=xy.numpy(), current_boxes=sample['cache']['current_boxes'][0].numpy(), current_logits=sample['cache']['current_logits'][0].numpy())
        ego_error = (xy[0] - sample['xy'][0, 0]).norm(dim=-1)
        # Evaluation association is independent geometric matching, not the training matcher.
        target = sample['target']; pred = {k: v[0] for k, v in sample['pred'].items()}
        _, support, obj = prediction_filters(pred, target)
        slots = torch.where(support & obj)[0]; gt = torch.where(target.current_supervision_mask)[0]
        r, g = match_geometry(pred['boxes'][slots, :2], target.current_boxes[gt, :2])
        r, g = slots[r], gt[g]
        valid = target.future_valid_mask[g]; error = (xy[r+1] - target.future_xy_in_ego_t0[g]).norm(dim=-1)
        stationary = (pred['boxes'][r, None, :2] - target.future_xy_in_ego_t0[g]).norm(dim=-1)
        final = valid[:, -1]
        rows.append({'token': token, 'status': 'ok', 'ego_ADE': float(ego_error.mean()), 'ego_FDE': float(ego_error[-1]),
                     'gt': len(gt), 'current_matches': len(r), 'current_predictions': len(slots),
                     'agent_error_sum': float(error[valid].sum()), 'agent_points': int(valid.sum()),
                     'agent_final_error_sum': float(error[:, -1][final].sum()), 'agent_final_count': int(final.sum()),
                     'stationary_error_sum': float(stationary[valid].sum()),
                     'valid_gt_points': int(target.future_valid_mask[gt].sum())})
        dynamic = valid[:, -1] & ((target.future_xy_in_ego_t0[g, -1] - target.current_boxes[g, :2]).norm(dim=-1) > 2.)
        static = valid[:, -1] & ~dynamic
        for label, select in [('dynamic', dynamic), ('static', static)]:
            mask = valid & select[:, None]
            rows[-1][label + '_error_sum'] = float(error[mask].sum())
            rows[-1][label + '_stationary_error_sum'] = float(stationary[mask].sum())
            rows[-1][label + '_points'] = int(mask.sum())
        assigned = {int(col): int(row) for row, col in zip(r, g)}
        for gi in gt.tolist():
            slot = assigned.get(gi); vm = target.future_valid_mask[gi]
            distance = None if slot is None else (xy[slot+1] - target.future_xy_in_ego_t0[gi]).norm(dim=-1)
            objects.append({'token':token, 'track_id':target.track_ids[gi], 'gt_class':int(target.current_classes[gi]),
                            'matched_slot':slot, 'detected':slot is not None, 'future_points':int(vm.sum()),
                            'ADE':None if slot is None or not vm.any() else float(distance[vm].mean()),
                            'FDE':None if slot is None or not vm[-1] else float(distance[-1])})

    points = sum(r['agent_points'] for r in rows); finals = sum(r['agent_final_count'] for r in rows)
    summary = {'step': step, 'scenes': len(rows), 'failed': 0, 'conditioning': 'all_futures_hidden; current image only',
               'ego_ADE': float(np.mean([r['ego_ADE'] for r in rows])), 'ego_FDE': float(np.mean([r['ego_FDE'] for r in rows])),
               'agent_ADE': sum(r['agent_error_sum'] for r in rows) / points if points else None,
               'agent_FDE': sum(r['agent_final_error_sum'] for r in rows) / finals if finals else None,
               'stationary_ADE': sum(r['stationary_error_sum'] for r in rows) / points if points else None,
               'motion_point_coverage': points / max(sum(r['valid_gt_points'] for r in rows), 1),
               'limitation': 'Joint graph ego output, not original DiT planning output; no PDMS evaluation.'}
    for group in ['dynamic', 'static']:
        count = sum(r[group + '_points'] for r in rows)
        summary[group] = {'points': count, 'ADE': sum(r[group + '_error_sum'] for r in rows) / count if count else None,
                          'stationary_ADE': sum(r[group + '_stationary_error_sum'] for r in rows) / count if count else None}
    write_csv(output / f'objects_{step}.csv', objects)
    write_csv(output / f'eval_{step}.csv', rows); (output / f'eval_{step}.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True); model.train()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['cache', 'targets', 'data-root', 'config', 'output', 'ledger', 'run-id']: p.add_argument('--'+key, required=True)
    p.add_argument('--steps', type=int, default=1000); p.add_argument('--batch', type=int, default=8)
    p.add_argument('--all-hidden-probability', type=float, default=.5); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume'); p.add_argument('--stop-after', type=int)
    a = p.parse_args()
    if not 1 <= a.steps <= 1000: raise ValueError('Isolation phase at most1000 updates')
    out = Path(a.output); out.mkdir(parents=True, exist_ok=bool(a.resume)); cfg = json.loads(Path(a.config).read_text())
    seed_all(a.seed); samples, manifest, target_hash = load_samples(a.cache, a.targets, a.data_root, 8)
    if len(samples) > 64: raise ValueError('This in-memory isolation entry is capped at64 scenes')
    identity = {'arguments': vars(a), 'code_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'cache_identity': manifest['identity_sha256'], 'labels_fingerprint': target_hash, 'config': cfg,
                'base_world_seen_training64': True, 'pdms_training': False}
    saved = torch.load(a.resume, map_location='cpu', weights_only=False) if a.resume else None
    if saved is not None:
        for key in ['code_sha', 'cache_identity', 'labels_fingerprint', 'config']:
            if saved['identity'][key] != identity[key]: raise ValueError('Resume identity changed: ' + key)
        for key in ['steps', 'batch', 'all_hidden_probability', 'seed', 'output', 'run_id']:
            if saved['identity']['arguments'][key] != vars(a)[key]: raise ValueError('Resume setting changed: ' + key)
        ledger = json.loads(Path(a.ledger).read_text())
        previous = next(r for r in ledger['runs'] if r['id'] == a.run_id)
        if previous['optimizer_steps'] != saved['step'] or previous['status'] == 'running':
            raise ValueError('Resume requires terminal, exactly accounted checkpoint')
        if saved['step'] >= a.steps: raise ValueError('Phase already completed')
    start(a.ledger, a.run_id, a.steps, identity, resume=bool(a.resume)); step = 0
    completed = 0
    try:
        model = JointTrajectoryFlow(samples[0]['cache']['context'].shape[-1], **{k: v for k, v in cfg.items() if k in ['dim','heads','layers','scale_m','trajectory_mode','agent_scale_m']}).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
        order = list(range(len(samples))); random.shuffle(order); position = 0; seen = set(); presentations = 0
        if saved is None:
            evaluate(model, samples, out, 0)
        else:
            model.load_state_dict(saved['model'], strict=True); opt.load_state_dict(saved['optimizer'])
            step = completed = saved['step']; order = saved['order']; position = saved['position']
            seen = set(saved['seen']); presentations = saved['presentations']
            random.setstate(saved['rng']['python']); np.random.set_state(saved['rng']['numpy'])
            torch.set_rng_state(saved['rng']['torch']); torch.cuda.set_rng_state_all(saved['rng']['cuda'])
        for step in range(step+1, a.steps+1):
            if not record(a.ledger, a.run_id, step-1): raise RuntimeError('Budget reached')
            batch = []
            for _ in range(a.batch):
                if position == len(order): random.shuffle(order); position = 0
                batch.append(samples[order[position]]); position += 1
            current = current_batch(batch); xy = torch.cat([s['xy'] for s in batch]).cuda(); valid = torch.cat([s['valid'] for s in batch]).cuda()
            hidden = actor_mask(len(batch), xy.shape[1], 'cuda', all_hidden_probability=a.all_hidden_probability)
            opt.zero_grad(set_to_none=True)
            sums, counts = training_loss_sums(model, xy, valid, hidden, torch.randn_like(xy), torch.rand(len(batch), device='cuda'), **current)
            loss = sum(sums[k] / max(counts[k], 1) for k in sums)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite graph loss')
            loss.backward(); norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)); opt.step(); completed = step
            presentations += len(batch); seen.update(s['cache']['token'] for s in batch)
            row = {'step':step,'loss':float(loss.detach()),'gradient_before_clip':norm, 'unique_scenes_seen':len(seen),
                   'sample_presentations':presentations,'effective_epochs':presentations/len(samples),'global_batch':a.batch}
            with (out/'train.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            record(a.ledger, a.run_id, step)
            if step % 50 == 0: print(json.dumps(row), flush=True)
            if step in {500, a.steps, a.stop_after}:
                evaluate(model, samples, out, step)
                torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'identity':identity,
                            'order':order,'position':position,'presentations':presentations,'seen':sorted(seen),
                            'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}},out/f'checkpoint_{step}.pt')
            if a.stop_after and step >= a.stop_after: break
        (out/'manifest.json').write_text(json.dumps(identity,indent=2)); record(a.ledger,a.run_id,completed,'complete' if completed == a.steps else 'paused')
    except BaseException:
        record(a.ledger,a.run_id,completed,'failed');raise


if __name__ == '__main__': main()
