"""Training-only real-head gradients/cost; does not claim an official PDMS gain."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from starVLA.dataloader.foresight_dataset import ForesightTrainingDataset
from starVLA.model.modules.foresight.trajectory_objective import TrajectoryObjectiveConfig, trajectory_loss_terms
from tools.foresight.checkpoints import checkpoint_identity, load_student
from tools.ddpolicy_vehicle.prepare_data import atomic_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--training-run', required=True); p.add_argument('--tag', default='milestone_100000')
    p.add_argument('--data', required=True); p.add_argument('--labels', required=True)
    p.add_argument('--objective', required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=4)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    if not 1 <= a.samples <= 8: raise ValueError('Small training-only calibration')
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    atomic_json(a.output/'progress.json', dict(stage='LOADING', started_unix=started))
    training, checkpoint = checkpoint_identity(a.training_run, a.tag)
    cfg = TrajectoryObjectiveConfig.from_mapping(json.loads(Path(a.objective).read_text()))
    model = load_student(a.training_run, a.tag, training, precision='bf16', strip=True)
    model.action_model.train()
    data = ForesightTrainingDataset(a.data, ego_relabel_root=a.labels)
    if data.identity['split'] != 'train': raise ValueError('Calibration cannot use dev/Navtest')
    chosen = np.random.default_rng(20261006).choice(len(data), size=a.samples, replace=False)
    records = []
    for at, index in enumerate(chosen):
        observation, target = data[int(index)]
        with torch.no_grad():
            encoded = model.encode_current([observation]); condition = model.build_planner_condition(encoded)
        condition = condition.detach().requires_grad_()
        ego = target['ego'][None].cuda().float(); y = ego.repeat(8, 1, 1)
        gen = torch.Generator(device='cuda').manual_seed(99000 + int(index))
        noise = torch.randn(y.shape, generator=gen, device='cuda')
        u = torch.rand(8, generator=gen, device='cuda')
        head_cfg = model.action_model.config
        times = (head_cfg.noise_s-u.pow(1./head_cfg.noise_beta_alpha))/head_cfg.noise_s
        selected = (condition, model.action_model.action_decoder.layer2.weight)
        torch.manual_seed(77000 + int(index))
        with model.amp(): fm = model.action_model(condition.repeat(8,1,1), y, noise=noise, times=times)
        base_grad = torch.autograd.grad(fm, selected)
        torch.manual_seed(77000 + int(index))
        with model.amp(): weighted = model.action_model(condition.repeat(8,1,1), y, noise=noise, times=times,
            channel_weights=cfg.fm_channel_weights)
        weighted_grad = torch.autograd.grad(weighted, selected)
        torch.cuda.synchronize(); step_start = time.time(); torch.cuda.reset_peak_memory_stats()
        with model.amp(): rollout = model.action_model.rollout_action(condition,
            initial_noise=noise[:1], checkpoint_steps=True, inference_mode=True)
        terms, diagnostics = trajectory_loss_terms(rollout, ego, cfg)
        values = {}; gradients = {}
        for task, term in terms.items():
            grads = torch.autograd.grad(term.mean(), selected, retain_graph=True)
            if not all(torch.isfinite(g).all() for g in grads): raise FloatingPointError('Nonfinite '+task+' gradient')
            gradients[task] = tuple(g.detach().float() for g in grads)
            values[task] = float(term.mean().detach())
        combined = [sum(gradients[task][i] * getattr(cfg,task+'_weight') for task in gradients)
                    for i in range(len(selected))]
        def norms(grads): return dict(condition=float(grads[0].float().norm()), decoder=float(grads[1].float().norm()))
        bnorm, wnorm, rnorm = norms(base_grad), norms(weighted_grad), norms(combined)
        ratios = {key: rnorm[key]/max(wnorm[key],1e-12) for key in rnorm}
        record = dict(token=observation['token'], index=int(index), fm=float(fm.detach()), weighted_fm=float(weighted.detach()),
            rollout_raw=values, physical_diagnostics={k:float(v.mean().detach()) for k,v in diagnostics.items()},
            fm_gradient=bnorm, weighted_fm_gradient=wnorm, rollout_combined_gradient=rnorm,
            active_update_rollout_to_weighted_fm_gradient_ratio=ratios,
            rollout_term_gradient={k:norms(v) for k,v in gradients.items()},
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), seconds=time.time()-step_start)
        records.append(record); atomic_json(a.output/'records.json',records)
        atomic_json(a.output/'progress.json',dict(stage='CALIBRATING',completed=at+1,expected=a.samples,seconds=time.time()-started))
        del rollout,terms,gradients,combined,base_grad,weighted_grad,condition,encoded
        torch.cuda.empty_cache()
    result=dict(status='REAL_GRADIENT_CALIBRATION_COMPLETE', checkpoint=checkpoint, data=data.identity['identity'],
        labels=data.ego_relabel_identity['identity'], objective=asdict(cfg), samples=records,
        scope='uniform training scenes; real mature BF16 action head; detached current condition to measure shared-condition and decoder gradients; not optimizer/ZeRO verification or official PDMS',
        claim='finite gradients and magnitude evidence only; training/held-out safety validation still required',seconds=time.time()-started)
    atomic_json(a.output/'result.json',result); atomic_json(a.output/'progress.json',dict(stage='COMPLETE',seconds=time.time()-started))


if __name__ == '__main__': main()
