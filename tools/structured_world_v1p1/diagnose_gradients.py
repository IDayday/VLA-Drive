"""No-update task gradients and assignment diagnostics on fixed training scenes."""
import argparse
import hashlib
import io
import json
import subprocess
from pathlib import Path

import torch

from tools.structured_world.runtime import load_baseline, load_dataset, load_world_batch, seed_all
from tools.structured_world_v1p1.budget import start, record
from tools.structured_world_v1p1.reaudit_metrics import write_csv
from starVLA.model.modules.structured_world.policy import StructuredWorldPolicy
from starVLA.model.modules.structured_world.rehab import reference_loss_sums
from starVLA.model.modules.structured_world.metrics import match_geometry


def norm(parameters):
    return sum(float(p.grad.float().square().sum()) for p in parameters if p.grad is not None) ** .5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['trained', 'manifest', 'targets', 'output', 'ledger', 'run-id']:
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--limit', type=int, default=16)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    blob = Path(args.trained).read_bytes()
    saved = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=False)
    cfg = saved['identity']['config']
    base = saved['identity']['arguments']
    identity = dict(arguments=vars(args), checkpoint_sha256=hashlib.sha256(blob).hexdigest(),
                    trained_step=saved['step'], code_sha=subprocess.check_output(
                        ['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest())
    start(args.ledger, args.run_id, 0, identity)
    try:
        seed_all(42)
        agent = load_baseline(base['checkpoint'], base['vlm'])
        agent.model.requires_grad_(False)
        policy = StructuredWorldPolicy(agent.model, cfg).cuda().eval()
        for name in ['reader', 'heads']:
            getattr(policy, name).load_state_dict(saved[name], strict=True)
        parameters = [p for p in policy.parameters() if p.requires_grad]
        assert not any(p.requires_grad for p in policy.baseline.parameters())
        ds = load_dataset(agent, args.manifest, base['data_root'], args.limit)
        rows, assignments = [], []
        target_hash = hashlib.sha256()
        for index in range(len(ds)):
            if not record(args.ledger, args.run_id, 0):
                raise RuntimeError('GPU hour cap')
            raw = ds[index]
            example = {k: raw[k] for k in ['image', 'lang', 'state', 'token']}
            # Labels are opened after inference and cannot influence model inputs.
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = policy.encode_conditions([example], world_only=True)[1]
            _, targets = load_world_batch([example], args.targets)
            target = targets[0]
            assert not target.overflow and bool(target.annotation_valid_mask)
            token = example['token']
            target_hash.update(token.encode() + hashlib.sha256(
                (Path(args.targets) / 'targets' / (token + '.pt')).read_bytes()).digest())
            sums, counts, matches = reference_loss_sums(prediction, targets)
            vectors = {}
            for task in sums:
                grad = torch.autograd.grad(sums[task] / max(counts[task], 1.),
                                           policy.reader.queries, retain_graph=True,
                                           allow_unused=True)[0]
                vectors[task] = torch.zeros_like(policy.reader.queries).flatten() if grad is None else grad.float().flatten()
            row = {'token': token, 'gt_count': int(target.current_supervision_mask.sum())}
            for task, vector in vectors.items():
                row[task + '_query_grad_norm'] = float(vector.norm())
                row[task + '_loss'] = float(sums[task].detach()) / max(counts[task], 1.)
            for a, b in [('cls', 'box'), ('cls', 'motion'), ('box', 'motion')]:
                denominator = float(vectors[a].norm() * vectors[b].norm())
                row[a + '_' + b + '_cosine'] = float(vectors[a].dot(vectors[b])) / denominator if denominator else None
            policy.zero_grad(set_to_none=True)
            sum(cfg.get('lambda_' + task, 1.) * sums[task] / max(counts[task], 1.) for task in sums).backward()
            row['grad_norm_before_clip_measured'] = norm(parameters)
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            row['grad_norm_after_clip_measured'] = norm(parameters)
            boxes = prediction['boxes'][0].detach().float()
            r, c = matches[0]
            gt = torch.where(target.current_supervision_mask)[0]
            gr, gc = match_geometry(boxes[:, :2], target.current_boxes[gt, :2])
            row['training_assignment_within_2m'] = int(((boxes[r, :2] - target.current_boxes[c, :2]).norm(dim=-1) < 2).sum())
            row['geometry_maximum_within_2m'] = len(gr)
            row['training_assignment_mean_distance_m'] = float((boxes[r, :2] - target.current_boxes[c, :2]).norm(dim=-1).mean()) if len(r) else None
            row['slot_residual_std_xy_m'] = float((boxes[:, :2] - prediction['references'][0]).std(dim=0).mean())
            for slot, target_index in zip(r.tolist(), c.tolist()):
                assignments.append({'token': token, 'track_id': target.track_ids[target_index], 'slot': slot,
                                    'centre_error_m': float((boxes[slot, :2] - target.current_boxes[target_index, :2]).norm())})
            rows.append(row)
            print(json.dumps(row), flush=True)
        write_csv(out / 'scenes.csv', rows)
        write_csv(out / 'assignments.csv', assignments)
        identity.update(target_fingerprint=target_hash.hexdigest(), optimizer_updates=0,
                        gradient_normalization='per-scene task denominator; diagnostic, not accumulated batch training',
                        scenes=len(rows), failed=0)
        (out / 'manifest.json').write_text(json.dumps(identity, indent=2))
        record(args.ledger, args.run_id, 0, 'complete')
    except BaseException:
        record(args.ledger, args.run_id, 0, 'failed')
        raise


if __name__ == '__main__':
    main()
