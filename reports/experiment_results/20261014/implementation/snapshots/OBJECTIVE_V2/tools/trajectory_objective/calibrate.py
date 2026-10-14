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
    p.add_argument('--compare-objective')
    p.add_argument('--generic-initialization', action='store_true')
    p.add_argument('--samples', type=int, default=4)
    p.add_argument('--braking-samples', type=int, default=0)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    if not 1 <= a.samples <= 8: raise ValueError('Small training-only calibration')
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    atomic_json(a.output/'progress.json', dict(stage='LOADING', started_unix=started))
    training, checkpoint = checkpoint_identity(a.training_run, a.tag)
    cfg = TrajectoryObjectiveConfig.from_mapping(json.loads(Path(a.objective).read_text()))
    comparison = TrajectoryObjectiveConfig.from_mapping(json.loads(Path(a.compare_objective).read_text())) if a.compare_objective else None
    if a.generic_initialization:
        from omegaconf import OmegaConf
        from starVLA.model.framework import build_framework
        config=OmegaConf.create(training['config']);config.framework.qwenvl.device_map='cpu'
        model=build_framework(config);model.strip_auxiliary_heads();model.cuda().bfloat16().eval()
    else:
        model = load_student(a.training_run, a.tag, training, precision='bf16', strip=True)
    model.action_model.train()
    data = ForesightTrainingDataset(a.data, ego_relabel_root=a.labels)
    if data.identity['split'] != 'train': raise ValueError('Calibration cannot use dev/Navtest')
    chosen = list(np.random.default_rng(20261006).choice(len(data), size=a.samples, replace=False))
    if a.braking_samples:
        if not 0 < a.braking_samples <= 4:raise ValueError('At most four additional braking scenes')
        overlay=np.load(Path(a.labels)/'labels.npz')
        xy=overlay['trajectories'][...,:2]
        velocity=np.diff(np.concatenate((np.zeros((len(xy),1,2)),xy),axis=1),axis=1)/.5
        speed=np.linalg.norm(velocity,axis=-1)
        mask=overlay['accepted'] & ((speed[:,:2].mean(-1)-speed[:,-2:].mean(-1))>cfg.fm_braking_speed_drop_mps)
        tokens=set(overlay['tokens'][mask].tolist())
        candidates=[i for i,row in enumerate(data.index) if row['token'] in tokens and i not in chosen]
        chosen += list(np.random.default_rng(20261008).choice(candidates,size=a.braking_samples,replace=False))
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
        reference=None
        if comparison is not None:
            torch.manual_seed(77000+int(index))
            with model.amp(): ref=model.action_model(condition.repeat(8,1,1), y, noise=noise, times=times,
                channel_weights=comparison.fm_channel_weights,endpoint_options=dict(braking_scene_weight=comparison.fm_braking_scene_weight,
                    braking_speed_drop_mps=comparison.fm_braking_speed_drop_mps))
            ref_loss=ref['fm']*comparison.fm_total_weight+ref['endpoint_velocity']*comparison.fm_endpoint_velocity_weight+ref['endpoint_acceleration']*comparison.fm_endpoint_acceleration_weight
            ref_grad=torch.autograd.grad(ref_loss,selected)
            reference=dict(loss=float(ref_loss.detach()),gradient=dict(condition=float(ref_grad[0].float().norm()),decoder=float(ref_grad[1].float().norm())))
            del ref,ref_loss,ref_grad
        torch.manual_seed(77000 + int(index))
        with model.amp(): endpoint = model.action_model(condition.repeat(8,1,1), y, noise=noise, times=times,
            channel_weights=cfg.fm_channel_weights, endpoint_options=dict(braking_scene_weight=cfg.fm_braking_scene_weight,
                braking_speed_drop_mps=cfg.fm_braking_speed_drop_mps))
        # Train-only reference fraction is recorded; optimizer training computes
        # the exact global-batch denominator, including original-GT fallbacks.
        normalizer=1.+cfg.fm_braking_scene_weight*.15 if cfg.fm_normalize_scene_weights else 1.
        components=dict(fm=endpoint['fm']*cfg.fm_total_weight/normalizer,
            endpoint_velocity=endpoint['endpoint_velocity']*cfg.fm_endpoint_velocity_weight/normalizer,
            endpoint_acceleration=endpoint['endpoint_acceleration']*cfg.fm_endpoint_acceleration_weight/normalizer)
        weighted=sum(components.values())
        component_grads={k:torch.autograd.grad(v,selected,retain_graph=True) for k,v in components.items()}
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
        record = dict(token=observation['token'], index=int(index), sampling_group='uniform' if at<a.samples else 'braking',
            fm=float(fm.detach()), weighted_fm=float(weighted.detach()),
            fm_components={k:float(v.detach()) for k,v in components.items()},
            fm_channel_raw=endpoint['channel_mse'].tolist(),
            fm_component_gradients={k:norms(v) for k,v in component_grads.items()},
            calibration_scene_weight_normalizer=normalizer,
            rollout_raw=values, physical_diagnostics={k:float(v.mean().detach()) for k,v in diagnostics.items()},
            fm_gradient=bnorm, weighted_fm_gradient=wnorm, rollout_combined_gradient=rnorm,
            active_update_rollout_to_weighted_fm_gradient_ratio=ratios,
            rollout_term_gradient={k:norms(v) for k,v in gradients.items()},
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), seconds=time.time()-step_start)
        record['objective_to_original_fm_gradient_ratio']={k:wnorm[k]/max(bnorm[k],1e-12) for k in bnorm}
        record['comparison_objective']=reference
        record['weighted_fm_vs_original_gradient_cosine']={key:float(torch.nn.functional.cosine_similarity(base_grad[i].float().flatten(),weighted_grad[i].float().flatten(),dim=0)) for i,key in enumerate(['condition','decoder'])}
        records.append(record); atomic_json(a.output/'records.json',records)
        atomic_json(a.output/'progress.json',dict(stage='CALIBRATING',completed=at+1,expected=len(chosen),seconds=time.time()-started))
        del rollout,terms,gradients,combined,base_grad,weighted_grad,condition,encoded,component_grads,components,endpoint
        torch.cuda.empty_cache()
    result=dict(status='REAL_GRADIENT_CALIBRATION_COMPLETE', checkpoint=checkpoint, generic_initialization=a.generic_initialization, data=data.identity['identity'],
        labels=data.ego_relabel_identity['identity'], objective=asdict(cfg), comparison_objective=asdict(comparison) if comparison else None, samples=records,
        scope='uniform training scenes; real mature BF16 action head; detached current condition to measure shared-condition and decoder gradients; not optimizer/ZeRO verification or official PDMS',
        claim='finite gradients and magnitude evidence only; training/held-out safety validation still required',seconds=time.time()-started)
    atomic_json(a.output/'result.json',result); atomic_json(a.output/'progress.json',dict(stage='COMPLETE',seconds=time.time()-started))


if __name__ == '__main__': main()
