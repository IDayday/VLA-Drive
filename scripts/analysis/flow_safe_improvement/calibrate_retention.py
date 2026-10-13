"""TRAIN-only common retention scale after a bounded full-head FM pilot.

This observer measures native BF16 leaf gradients in FP64 reductions. It does
not call them actual globally reduced ZeRO optimizer gradients; those are saved
separately by the eight-GPU qualification.
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from scripts.analysis.flow_safe_improvement.train_fixed_bank import load_sealed
from starVLA.rl.flow_grpo.loading import weight_path
from starVLA.rl.flow_grpo.model import make_reference
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import PopulationContract, file_sha, fingerprint, seed_for
from starVLA.rl.safe_flow_improvement.policy_adapter import original_fm_loss
from starVLA.rl.safe_flow_improvement.reference_retention import fm_draws, original_replay_loss, safe_path_retention
from starVLA.rl.safe_flow_improvement.sampling import load_cached_policy


def dot(left, right):
    return sum(float((a.detach().float()*b.detach().float()).sum(dtype=torch.float64)) for a, b in zip(left, right))


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('population', 'cohorts', 'targets', 'reference-scores', 'behavior', 'behavior-sha256', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--config', default='configs/flow_grpo/frozen_action_head_batched_single8_epoch1.yaml')
    p.add_argument('--source-config', default='/mnt/project/DriveDreamer-Policy-flow-grpo/artifacts/action-only-source/checkpoint_code/action-only-checkpoints-v1/configs/frozen-visual-step100000-config.yaml')
    a = p.parse_args()
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    configure_numerics()
    pop, cohorts = [json.loads(Path(x).read_text()) for x in (a.population, a.cohorts)]
    targets = json.loads((Path(a.targets)/'manifest.json').read_text())
    tokens = pop['calibration_tokens'][:16]
    if tokens != targets['tokens'] or not targets['qualification_only']:
        raise ValueError('Use the exact sixteen-scene TRAIN qualification bank')
    contract = PopulationContract(*[frozenset(pop[k]) for k in ('train_tokens','dev_tokens','navtest_tokens')], pop['token_logs'])
    contract.require_training(tokens, 'retention calibration')
    safe = cohorts['reference_joint_safe'][:16]
    contract.require_training(safe, 'fixed reference-safe retention calibration')
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    registration = dict(population_identity=pop['identity_sha256'], target_identity=targets['sha256'],
                        source_sha256=file_sha(__file__), behavior_sha256=a.behavior_sha256,
                        coefficient_candidates=[.1, 1., 10.], target_gradient_ratio=.25,
                        rule='closest_median_log_retention_to_improve_plus_0.1_replay_gradient_norm_ratio_to_0.25',
                        scenes=16, native_gradient_precision='BF16_leaves_FP64_observer_reductions',
                        qualification_window_not_method_selection=True, dev_or_navtest_used=False)
    atomic_json(out/'preregistration.json', registration)
    policy, store, cfg, sft, _ = load_cached_policy(a.config, a.source_config, behavior=a.behavior, behavior_sha256=a.behavior_sha256)
    reference = make_reference(policy)
    with torch.serialization.safe_globals([set]):
        state = torch.load(weight_path(cfg['sft_checkpoint']), map_location='cpu', mmap=True, weights_only=True)
    state = state.get('module', state)
    reference.action_model.load_state_dict({k.removeprefix('action_model.'):v for k,v in state.items() if k.startswith('action_model.')}, strict=True)
    del state
    parameters = [p for p in policy.parameters() if p.requires_grad]
    if len(parameters) != 359:
        raise ValueError('Calibrate the complete action head, never a readout')
    rows = []
    for index, (token, safe_token) in enumerate(zip(tokens, safe)):
        row = load_sealed(a.targets, token, targets['sha256'])
        key = seed_for('retention_calibration', 42, token)
        choice = int(torch.multinomial(row['target_probabilities'],1,generator=torch.Generator().manual_seed(key)))
        value, safe_value = store.get(token), store.get(safe_token)
        raw = value['raw'].cuda()
        target = row['actions'][choice:choice+1].cuda().float()
        y, noise, time = fm_draws(policy.action_model, target, key)
        improve = original_fm_loss(policy, raw.repeat(8,1,1), y, noise, time)
        g_improve = torch.autograd.grad(improve, parameters)
        replay = original_replay_loss(policy, raw, value['action'][None].cuda().float(), seed_for('replay_calibration',42,token))
        g_replay = torch.autograd.grad(replay, parameters)
        fixed = load_sealed(a.reference_scores, safe_token)
        retention = safe_path_retention(policy, reference, safe_value['raw'].cuda(), fixed['sample']['normalized_actions'][index%4:index%4+1].cuda(), seed_for('safe_calibration',42,safe_token))
        g_retention = torch.autograd.grad(retention, parameters)
        if any(not torch.isfinite(g).all() for group in (g_improve,g_replay,g_retention) for g in group):
            raise FloatingPointError('Invalid calibration gradient')
        ii, gg, rr = dot(g_improve,g_improve), dot(g_replay,g_replay), dot(g_retention,g_retention)
        ig, ir, gr = dot(g_improve,g_replay), dot(g_improve,g_retention), dot(g_replay,g_retention)
        base = math.sqrt(max(0., ii+.01*gg+.2*ig))
        rows.append(dict(token=token, reference_token=safe_token, improve_norm=math.sqrt(ii), replay_norm=math.sqrt(gg),
                         retention_norm=math.sqrt(rr), base_joint_norm=base,
                         retention_to_base=math.sqrt(rr)/base if base > 0 else None,
                         improve_replay_cosine=ig/math.sqrt(ii*gg) if ii*gg>0 else None,
                         improve_retention_cosine=ir/math.sqrt(ii*rr) if ii*rr>0 else None,
                         replay_retention_cosine=gr/math.sqrt(gg*rr) if gg*rr>0 else None))
        del g_improve,g_replay,g_retention,improve,replay,retention
        atomic_json(out/'per_scene.json',rows)
    ratios = [r['retention_to_base'] for r in rows if r['retention_to_base'] is not None and r['retention_to_base']>0]
    if len(ratios) < 8:
        raise ValueError('Too few nonzero post-update retention gradients to calibrate')
    scores = {c:float(np.median(np.abs(np.log(np.array(ratios)*c/.25)))) for c in (.1,1.,10.)}
    selected = min(scores,key=lambda c:(scores[c],c))
    result = dict(registration, selected_coefficient=selected, selection_distances=scores,
                  median_unweighted_gradient_ratio=float(np.median(ratios)), status='LOCKED_COMMON_RETENTION_SCALE')
    result['sha256']=fingerprint(result)
    atomic_json(out/'registration.json',result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
