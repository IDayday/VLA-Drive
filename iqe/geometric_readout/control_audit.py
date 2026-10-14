"""Read-only matched validation predictions from this project's original S11.

This is a control measurement, never a new scorer initializer or teacher. Only
the already selected original S11 checkpoint and fixed stage_val are accepted.
"""
import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch

from iqe.contracts import require, TrajectoryContract
from iqe.evaluation.retention import module_hash
from iqe.internal_baseline.scorer import initialization_rng
from iqe.io import atomic_bytes, atomic_json, file_hash, lock, read_json
from iqe.specialist_study.binding import scenes_by_role
from iqe.specialist_study.data import ReusedContext
from iqe.specialist_study.context_reader import VerifiedSpatialReader
from iqe.specialist_study.models import StudyScorer
from iqe.specialist_study.scorer_training import predict
from iqe.specialist_study.metrics import metrics, select
from iqe.specialist_study.training import runtime
from .full import load_config, resource_check


def run(config, device):
    resource_check(config)
    device = runtime(device)
    source = config.source()
    folder = source.root / 'scorers/S11'
    result = read_json(folder / 'result.json')
    checkpoint = folder / 'best.pt'
    require(result['status'] == 'COMPLETE'
            and file_hash(checkpoint) == result['best_checkpoint_hash'], 'invalid own S11 control')
    reference = [(s, read_json(folder / f'validation_{s:06d}.json'))
                 for s in (3000, 4000, 5000, 6000)]
    step, validation = max(reference, key=lambda x: (x[1]['selection_key'], -x[0]))
    identity = dict(checkpoint=str(checkpoint), checkpoint_hash=result['best_checkpoint_hash'],
                    step=step, validation_hash=file_hash(folder / f'validation_{step:06d}.json'),
                    bank_hash=read_json(source.root / 'banks/deploy/stage_val/BANK.json')['hash'],
                    role='stage_val', external_scorer=False, new_optimizer_steps=0,
                    teacher_or_initialization_use=False, code_hash=file_hash(__file__))
    output = config.root / 'qualification/global_S11_control'
    with lock(output / 'RUN.lock'):
        done = output / 'COMPLETE.json'
        if done.exists():
            receipt = read_json(done)
            require(receipt['identity'] == identity
                    and file_hash(output / 'predictions.npz') == receipt['prediction_hash'], 'stale control')
            return receipt
        scene = scenes_by_role(source)['scorer_fit'][0]
        x = ReusedContext(source, 1).read(scene)
        v = VerifiedSpatialReader(source, 1).read(scene)
        contract = TrajectoryContract(**read_json(source.root / 'OFFLINE_BASE_CONTRACT.json')['trajectory'])
        with initialization_rng(source.seed):
            model = StudyScorer(x['world'].shape[-1], x['action'].shape[-1], contract,
                                spatial=True, visual_dim=v['visual'].shape[-1], views=v['visual'].shape[0])
        require(module_hash(model) == read_json(folder / 'SCORER_INITIALIZATION.json')['initial_state_hash'],
                'control architecture/initializer changed')
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        require(module_hash(model) == validation['scorer_state_hash'], 'selected control weights mismatch')
        meta, bank, values = predict(source, model.to(device).eval().requires_grad_(False), 'stage_val', device)
        chosen = select(values, bank['valid'], expert_ids=meta['provenance']['sources'])
        statistics = metrics(bank['scores'], bank['components'], chosen, meta['source_logs'],
                             seed=source.bootstrap_seed, repetitions=source.bootstrap_repetitions)
        require(meta['hash'] == identity['bank_hash'] and meta['pool_hash'] == validation['pool_hash'],
                'control population changed')
        require(abs(statistics['selected_01'] - validation['selected_01']) < 1e-10
                and statistics['high_to_zero'] == validation['high_to_zero'], 'control reproduction differs')
        buffer = io.BytesIO()
        np.savez(buffer, values=values)
        atomic_bytes(output / 'predictions.npz', buffer.getvalue(), immutable=True)
        receipt = dict(identity=identity, bank_hash=meta['hash'], statistics=statistics,
                       prediction_hash=file_hash(output / 'predictions.npz'), status='COMPLETE')
        atomic_json(done, receipt, immutable=True)
        return receipt


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    print(json.dumps(run(load_config(a.config), a.device), allow_nan=False))


if __name__ == '__main__':
    main()
