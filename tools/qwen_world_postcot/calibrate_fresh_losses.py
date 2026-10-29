"""One32-scene TRAIN calibration at generic initialization, no optimizer/score."""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import yaml

from starVLA.model.modules.qwen_world_postcot.model import QwenWorldPostCoT
from starVLA.model.modules.qwen_world_postcot.flow import shift_sigma
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
from starVLA.rl.qwen_world_postcot.data import WorldSceneData
from starVLA.rl.qwen_world_postcot.contracts import require_dispatch, file_sha, fingerprint
from starVLA.rl.qwen_world_postcot.checkpoint import atomic_json
from starVLA.rl.qwen_world_postcot.coordinate_objective import CoordinateObjective, tolerance_weights
from starVLA.rl.qwen_world_postcot.loss_designs import LossDesign
from starVLA.rl.qwen_world_postcot.fresh_loss import student_initialization_digest
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from tools.qwen_world_postcot.calibrate_loss_designs import residual_time_weights
from tools.qwen_world_postcot.train_il import validate, verify_generic_qwen_identity, QWEN


def main():
    p = argparse.ArgumentParser(__doc__); p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); require_dispatch(admit_long_job=True)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); cfg = validate(yaml.safe_load(Path(a.config).read_text()))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8'); configure_numerics(); torch.set_num_threads(4)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    provenance = verify_generic_qwen_identity(); data = WorldSceneData(cfg['training_data'])
    processor = QwenObservationProcessor(QWEN)
    model = QwenWorldPostCoT(QWEN, config=cfg['model'])
    init_hash = student_initialization_digest(model)
    model.to('cuda:0').eval(); model.configure_stage(True)
    time_stats = residual_time_weights(data)
    tokens = [data.tokens[data.index(i)] for i in range(32)]
    rng = torch.Generator().manual_seed(42000000)
    noise = torch.randn(32, 8, 3, generator=rng); sigma = shift_sigma(torch.rand(32, generator=rng))
    params = (model.actor.action_output.weight, model.actor.action_output.bias,
              model.actor.output_norm.weight, model.actor.output_norm.bias)
    objectives = dict(CONTROL=LossDesign('CONTROL'), PHYSICAL=CoordinateObjective(tolerance_weights()),
        TIME_RESIDUAL=LossDesign('TIME_RESIDUAL', horizon_weights=time_stats['weights']),
        DELTA_UNIT=LossDesign('DELTA', delta_coefficient=1.), HUBER=LossDesign('HUBER'))
    valid_all = torch.stack([(torch.tensor(data.records[t]['raw_future_dt_seconds']).cumsum(0)
        - torch.arange(1, 9)*.5).abs() < .06 for t in tokens])
    denominator = int(valid_all.sum())*3
    gradients = {name: None for name in objectives}; losses = dict.fromkeys(objectives, 0.)
    for offset in range(0, 32, 2):
        selected = tokens[offset:offset+2]; obs = [data.observation(t) for t in selected]
        ego = torch.tensor([o['proprio'] for o in obs], device='cuda:0')
        target = ActionCodec.norm_odo(torch.stack([data.target(t) for t in selected])).cuda()
        valid = valid_all[offset:offset+2].cuda()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            memory = model.encode_condition(processor(obs), ego)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            context = model.actor.prepare_context(memory)
            s = sigma[offset:offset+2].cuda(); eps = noise[offset:offset+2].cuda()
            prediction = model.actor((1-s[:, None, None])*target+s[:, None, None]*eps, s, context)
        for index, (name, obj) in enumerate(objectives.items()):
            _, terms = obj(prediction, target, valid); loss = terms['numerator']/denominator
            gs = torch.autograd.grad(loss, params, retain_graph=index < len(objectives)-1)
            vector = torch.cat([g.detach().float().cpu().flatten() for g in gs])
            gradients[name] = vector if gradients[name] is None else gradients[name]+vector
            losses[name] += float(loss)
        del context, memory, prediction
    baseline = gradients['CONTROL']; delta = gradients['DELTA_UNIT']-baseline
    coefficient = float(.1*baseline.norm()/delta.norm())
    raw = {k: gradients[k] for k in ('PHYSICAL', 'TIME_RESIDUAL', 'HUBER')}
    raw['DELTA'] = baseline+coefficient*delta
    scales = {name: float(baseline.norm()/g.norm()) for name, g in raw.items()}
    if (not math.isfinite(coefficient) or coefficient <= 0
            or any(not math.isfinite(v) or not .25 <= v <= 4 for v in scales.values())):
        raise ValueError('Fresh calibration outside preregistered positive .25..4 output-gradient scale bounds')
    arms = dict(PHYSICAL=dict(kind='PHYSICAL', scale=scales['PHYSICAL'], weights=tolerance_weights()),
        TIME_RESIDUAL=dict(kind='LOSS_DESIGN', loss_kind='TIME_RESIDUAL', scale=scales['TIME_RESIDUAL'], horizon_weights=time_stats['weights']),
        DELTA=dict(kind='LOSS_DESIGN', loss_kind='DELTA', scale=scales['DELTA'], delta_coefficient=coefficient),
        HUBER=dict(kind='LOSS_DESIGN', loss_kind='HUBER', scale=scales['HUBER']))
    row = dict(schema='world_fresh_loss_calibration_v1', status='PASS', arms=arms, split='TRAIN', scenes=32,
        token_sha256=fingerprint(tokens), initialization_student_sha256=init_hash,
        qwen_identity_file_sha256=provenance['identity_file_sha256'], time_statistics=time_stats,
        raw_loss_values=losses, baseline_output_gradient_norm=float(baseline.norm()),
        gradient_scope=['action_output.weight', 'action_output.bias', 'output_norm.weight', 'output_norm.bias'],
        scales=scales, delta_target_fraction=.1, delta_coefficient=coefficient,
        gradient_cosines={k: float(torch.dot(baseline, g)/(baseline.norm()*g.norm())) for k, g in raw.items()},
        model_forward_calls=16, optimizer_updates=0, scorer_calls=0, calibration_seed=42000000,
        wall_seconds=time.monotonic()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        config_sha256=file_sha(a.config), source_sha256=file_sha(__file__),
        limitation='Output-layer gradients only, not full-model gradient matching; all four auxiliary losses remain unchanged in formal IL.')
    row['sha256'] = fingerprint(row); atomic_json(out/'CALIBRATION.json', row)
    print(json.dumps(dict(status='PASS', scales=scales, initialization_student_sha256=init_hash, wall_seconds=row['wall_seconds'])), flush=True)


if __name__ == '__main__': main()
