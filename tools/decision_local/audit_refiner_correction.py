"""Measure correction demand, learned residuals and fixed-head relation dependence.

Uses completed checkpoints only. GT is a target for descriptive errors and is
an input only for explicitly privileged E probes. Relation removal is an
inference dependence diagnostic, not an independently trained control.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.decision_local.common import atomic_json, check_host, digest, read, source_sha
from tools.decision_local.train_small_model import build_model, load_predictor
from starVLA.dataloader.decision_local import FrozenConsequenceData, inputs_queries_targets, oracle_relations
from starVLA.dataloader.foresight_dataset import decode_ego
from starVLA.model.modules.decision_local.consequence_contract import OracleRelations, predicted_relations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('features', 'labels', 'checkpoint', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--predictor', type=Path)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--max-seconds', type=float, required=True)
    args = parser.parse_args()
    resources = check_host()
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.output.exists():
        raise FileExistsError('Preserve completed or partial correction diagnostics')
    data = FrozenConsequenceData(args.features, args.labels, reader='mmap_tensors')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    contract = state['identity']
    kind = contract['kind']
    if kind.startswith('P_') or contract['anchor_identity'] != data.feature_identity['anchor_identity']:
        raise ValueError('A matched refiner checkpoint is required')
    if contract['query_config'] != data.label_identity['query_config']:
        raise ValueError('Different query contract')
    if state['epoch'] != 24 or state['offset'] != 0 or not (args.checkpoint.parent/'COMPLETE.json').exists():
        raise ValueError('This diagnostic uses common completed24epoch heads only')
    model = build_model(kind, contract['seed'], contract['dim']).cuda().eval()
    model.load_state_dict(state['model'], strict=True)
    predictor = None
    if kind.startswith('R_'):
        if args.predictor is None or digest(args.predictor) != contract['predictor_sha256']:
            raise ValueError('Use the same frozen predictor as refiner training')
        predictor, _ = load_predictor(args.predictor, 'P_DIFF' if kind == 'R_FUTURE_DIFF' else 'P_ABS')
    elif args.predictor is not None:
        raise ValueError('E probes do not read an additional predictor')
    rows = []
    with torch.no_grad():
        for start in range(0, len(data.index), args.batch):
            if time.monotonic()-started >= args.max_seconds:
                raise TimeoutError('Finite diagnostic bound reached')
            ids = list(range(start, min(start+args.batch, len(data.index))))
            batch = data.batch(ids, 'cuda', evaluation=True)
            inputs, queries, targets = inputs_queries_targets(batch)
            if kind.startswith('E_') and kind != 'E_NONE':
                relations = oracle_relations(batch, kind)
                encoded = model.oracle_probe(inputs, queries, relations)
                removed = model.oracle_probe(inputs, queries,
                    OracleRelations(torch.zeros_like(relations.values), relations.common_valid, relations.source))
            else:
                relations = predicted_relations(predictor(inputs, queries)) if predictor is not None else None
                if kind == 'R_CURRENT_PRED':
                    relations = torch.cat((relations[..., :3], relations[..., 1:3]), -1)
                encoded = model.camera_only(inputs, queries, relations)
                removed = model.camera_only(inputs, queries, torch.zeros_like(relations) if relations is not None else None)
            final, ablated, truth = [decode_ego(x).cpu().numpy() for x in (encoded, removed, targets.ego)]
            q0 = inputs.q0.cpu().numpy()
            mask = targets.future_valid.cpu().numpy().astype(bool)
            for b, i in enumerate(ids):
                valid = mask[b]
                if not valid.any():
                    rows.append({**data.index[i], 'valid_future_points': 0})
                    continue
                demand = truth[b, :, :2]-q0[b, :, :2]
                residual = final[b, :, :2]-q0[b, :, :2]
                denom = np.linalg.norm(demand[valid])*np.linalg.norm(residual[valid])
                e0 = np.linalg.norm(demand, axis=-1)
                ef = np.linalg.norm(final[b, :, :2]-truth[b, :, :2], axis=-1)
                magnitude = np.linalg.norm(residual, axis=-1)
                dependence = np.linalg.norm(final[b, :, :2]-ablated[b, :, :2], axis=-1)
                heading_delta = final[b, :, 2]-q0[b, :, 2]
                rows.append({**data.index[i], 'valid_future_points': int(valid.sum()),
                    'q0_ADE_m': float(e0[valid].mean()), 'q_final_ADE_m': float(ef[valid].mean()),
                    'q0_FDE_m': float(e0[np.flatnonzero(valid)[-1]]),
                    'q_final_FDE_m': float(ef[np.flatnonzero(valid)[-1]]),
                    'ADE_improvement_m': float((e0-ef)[valid].mean()),
                    'residual_mean_m': float(magnitude[valid].mean()),
                    'relation_zeroing_trajectory_change_m': float(dependence[valid].mean()),
                    'residual_GT_direction_cosine': float((demand[valid]*residual[valid]).sum()/denom) if denom > 1e-12 else None,
                    'residual_heading_mean_rad': float(np.abs(np.arctan2(np.sin(heading_delta), np.cos(heading_delta)))[valid].mean()),
                    'per_time_q0_error_m': e0.tolist(), 'per_time_final_error_m': ef.tolist(),
                    'per_time_residual_m': magnitude.tolist(), 'per_time_valid': valid.tolist()})
    groups = {}
    for name, predicate in [('all_scenes', lambda x: True),
                            ('q0_ADE_le_0p5m', lambda x: x.get('q0_ADE_m', float('inf')) <= .5),
                            ('q0_ADE_gt_0p5m', lambda x: x.get('q0_ADE_m', -1) > .5)]:
        selected = [x for x in rows if predicate(x)]
        supported = [x for x in selected if x['valid_future_points']]
        values = {}
        if supported:
            for key in ('q0_ADE_m', 'q_final_ADE_m', 'q0_FDE_m', 'q_final_FDE_m', 'ADE_improvement_m',
                        'residual_mean_m', 'relation_zeroing_trajectory_change_m',
                        'residual_GT_direction_cosine', 'residual_heading_mean_rad'):
                array = [x[key] for x in supported if x[key] is not None]
                values[key] = {'mean': float(np.mean(array)), 'median': float(np.median(array)),
                               'p90': float(np.quantile(array, .9)), 'scenes': len(array)} if array else None
        groups[name] = {'scenes': len(selected), 'supported_scenes': len(supported), 'metrics': values}
    args.output.mkdir(parents=True)
    with (args.output/'scenes.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    atomic_json(args.output/'summary.json', {'source_sha': source_sha(ROOT), 'tool_sha256': digest(__file__),
        'checkpoint_sha256': digest(args.checkpoint), 'kind': kind, 'head_seed': contract['seed'],
        'features': data.feature_identity['identity'], 'labels': data.label_identity['identity'],
        'split': data.feature_identity['split'], 'scenes': len(rows), 'groups': groups,
        'GPU_hours': (time.monotonic()-started)/3600, 'resources': resources,
        'precision': 'FP32 heads, TF32off; fixed anchor42; descriptive batch32, not canonical deployment latency',
        'limits': ['Zeroing relation values retains the same support and queries; it is input-dependence evidence, not a trained E_NONE control',
                   'ADE and correction size are descriptive; canonical safety/progress and net failures remain separate',
                   'Training diagnostic4096 is in-sample for the frozen S0 and the small head; no hidden reselection of scenes']})


if __name__ == '__main__':
    main()
