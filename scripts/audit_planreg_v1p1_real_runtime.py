#!/usr/bin/env python3
"""Bounded real InternVL/trainval update audit, never a PDMS benchmark claim."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from navsim.agents.EpisodeDrive.precision import audit_precision, parameter_update_statistics
from navsim.agents.EpisodeDrive.shared_planreg_initialization import capture_shared_trainable_state
from navsim.agents.EpisodeDrive.gradient_diagnostics import isolated_same_batch_audit
from navsim.agents.EpisodeDrive.layers.world_model.frozen_task_diagnostics import frozen_predictor_controls
from navsim.planning.training.dataset import load_feature_target_from_pickle
from planreg_audit_runtime import collate_samples, to_device_non_paths
from export_planreg_student_checkpoint import export_student_checkpoint, sha256_file


def agent_config(variant, shared):
    with initialize_config_dir(version_base=None, config_dir=str(
            ROOT / 'navsim/planning/script/config/common/agent')):
        cfg = compose(config_name='episode_drive_planreg_wm_v1p1_' + variant)
    cfg.initialization.shared_trainable_init_path = str(shared)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shared-init', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--prior-resolved-config', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--skip-pair', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.steps < 3:
        raise ValueError('Use >=3 actual updates, including post-zero-B A updates')
    args.output.mkdir(parents=True)
    report = {'kind': 'new_real_model_trainval_numerical_smoke', 'status': 'RUNNING',
              'not_formal_training': True, 'not_navtest_evaluation': True,
              'shared_init_sha256': sha256_file(args.shared_init)}
    def save():
        temporary=args.output/'runtime.json.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(args.output/'runtime.json')
    save()
    torch.manual_seed(0)
    torch.set_num_threads(4)
    cfg = agent_config('base', args.shared_init)
    OmegaConf.save(cfg, args.output/'architecture.yaml', resolve=True)
    agent = instantiate(cfg)
    agent.initialize()
    agent.to('cuda:0')
    state, metadata = capture_shared_trainable_state(agent)
    report['initial_state'] = metadata
    report['agent_checkpoint_loaded'] = agent._agent_checkpoint_loaded
    report['vision_class'] = str(type(agent.backbone.model.vision_model))
    report['llm_class'] = str(type(agent.backbone.model.language_model))
    if not args.skip_pair:
        vqa_cfg = agent_config('vqa', args.shared_init)
        vqa = instantiate(vqa_cfg)
        vqa.initialize()
        other, other_meta = capture_shared_trainable_state(vqa)
        assert state.keys() == other.keys()
        assert all(torch.equal(v, other[k]) for k,v in state.items())
        report['paired_trainable_initial_state_bitwise_equal'] = True
        report['vqa_initial_state'] = other_meta
        assert not vqa._agent_checkpoint_loaded
        del vqa, other
        gc.collect()
        torch.cuda.empty_cache()
    del state
    prior = OmegaConf.load(args.prior_resolved_config)
    train_logs, val_logs = set(prior.train_logs), set(prior.val_logs)
    assert not train_logs.intersection(val_logs)
    samples, provenance = [], []
    # Small deterministic train-only sample selection; no Navtest labels or
    # regret-ranked token list is read by this script.
    for log in sorted(train_logs):
        directory = args.cache_root/log
        if not directory.is_dir():
            continue
        for target_path in sorted(directory.glob('*/trajectory_target_planreg_wm_v1.gz')):
            feature_path = target_path.parent/'internvl_feature.gz'
            if not feature_path.is_file():
                continue
            features = load_feature_target_from_pickle(feature_path)
            targets = load_feature_target_from_pickle(target_path)
            if not bool(targets['future_valid_mask'].all()):
                continue
            samples.append((features, targets))
            provenance.append({'log': log, 'token': targets['token'],
                               'feature_sha256': sha256_file(feature_path),
                               'target_sha256': sha256_file(target_path)})
            break
        if len(samples) >= args.batch_size:
            break
    if len(samples) != args.batch_size:
        raise RuntimeError('Insufficient eligible train-only scenes')
    features, targets = collate_samples(samples)
    features, targets = to_device_non_paths(features, torch.device('cuda:0')), to_device_non_paths(targets, torch.device('cuda:0'))
    report.update(samples=provenance, train_val_overlap=0,
                  future_valid_mask=targets['future_valid_mask'].tolist(),
                  micro_batch=args.batch_size, optimizer_reference_global_batch=64,
                  note='Repeated small train batch for update proof, not convergence or throughput')
    total_steps = 43578
    agent.configure_total_optimizer_steps(total_steps)
    optimizers, schedulers = agent.get_optimizers(total_optimizer_steps=total_steps)
    optimizer, scheduler = optimizers[0], schedulers[0]['scheduler']
    report['optimizer_groups'] = agent._planreg_optimizer_group_summary
    before = {k:p.detach().clone() for k,p in agent.named_parameters() if p.requires_grad}
    report['steps'] = []
    agent.train()
    for step in range(args.steps):
        agent.set_optimizer_step(step)
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            predictions = agent(features)
            losses = agent.compute_loss(features, targets, predictions)
        assert torch.isfinite(losses['loss'])
        losses['loss'].backward()
        for name,p in agent.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise FloatingPointError('Nonfinite grad: ' + name)
        lora_blocks = []
        for index, block in enumerate(agent.backbone.model.vision_model.encoder.layers):
            norm = sum(float(p.grad.float().square().sum()) for n,p in block.named_parameters()
                       if 'lora_' in n and p.grad is not None) ** .5
            assert norm > 0, (index, norm)
            lora_blocks.append(norm)
        assert all(p.grad is None for p in agent.backbone.model.language_model.parameters())
        assert agent.backbone.planning_register_adapter.planning_registers.grad.norm() > 0
        assert any(p.grad is not None and p.grad.norm() > 0 for p in agent.action_head.q_former.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in agent.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        scheduler.step()
        agent.update_ema_after_optimizer_step(step+1, total_steps)
        torch.cuda.synchronize()
        report['steps'].append({'step':step, 'loss':float(losses['loss'].detach()),
                                'wm_loss':float(losses['wm_loss'].detach()),
                                'wm_weight':float(losses['wm_weight_current']),
                                'grad_norm_pre_clip':float(grad_norm),
                                'lora_block_grad_norms':lora_blocks,
                                'seconds':time.perf_counter()-start})
        report['precision'] = audit_precision(agent, optimizer, True)
        save()
        del predictions, losses
    report['actual_parameter_updates'] = parameter_update_statistics(before, agent)
    del before
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        report['same_batch_gradients'] = isolated_same_batch_audit(agent, features, targets)
    assert all(p.grad is None for p in agent.parameters())
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        current, future, valid = agent._encode_ema_register_targets(features, targets, args.batch_size)
        controls = copy.deepcopy(agent.future_register_predictor).eval().requires_grad_(False)
        report['frozen_controls'] = frozen_predictor_controls(
            controls, current, targets['trajectory'], future, features['status_feature'][:,4:6].norm(dim=-1))
        del controls
    report['ema_master_dtypes'] = sorted({str(v.dtype) for v in agent.ema_register_target.master.tensors().values()})
    report['ema_master_key_count'] = len(agent.ema_register_target.master.names)
    report['semantic_diagnostics'] = {k:float(v) for k,v in agent.get_planreg_register_diagnostics().items()}
    report['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
    save()
    # Explicitly labelled smoke artifact, never presented as a formal result.
    checkpoint = args.output/'smoke_training.ckpt'
    torch.save({'state_dict': {'agent.'+k:v for k,v in agent.state_dict().items()},
                'optimizer_states':[optimizer.state_dict()], 'lr_schedulers':[scheduler.state_dict()],
                'global_step':args.steps}, checkpoint)
    student_path = args.output/'smoke_student.ckpt'
    report['export'] = export_student_checkpoint(checkpoint, student_path, resolved_config=cfg)
    agent.eval()
    with torch.no_grad():
        reference = agent(features)['trajectory'].detach().cpu()
    del optimizer, scheduler, optimizers, schedulers, agent
    gc.collect()
    torch.cuda.empty_cache()
    deployment = copy.deepcopy(cfg)
    deployment.initialization = None
    deployment.checkpoint_path = str(student_path)
    OmegaConf.update(deployment, 'vlm_config.exact_student_checkpoint', True, force_add=True)
    deployment.vlm_config.gradient_checkpointing = False
    deployment.world_model.enabled = False
    deployment.ema.enabled = False
    student = instantiate(deployment)
    assert student.future_register_predictor is None and student.ema_register_target is None
    student.initialize()
    student.to('cuda:0').eval()
    current_only = {k:v for k,v in features.items() if not k.startswith('future')}
    with torch.no_grad():
        result = student(current_only)['trajectory'].cpu()
    assert torch.isfinite(result).all() and tuple(result.shape) == (args.batch_size,8,3)
    report['student_current_only_max_abs_diff'] = float((result-reference).abs().max())
    assert torch.equal(result, reference)
    report['status'] = 'PASS'
    save()
    print(json.dumps({k:v for k,v in report.items() if k != 'actual_parameter_updates'}, indent=2))


if __name__ == '__main__':
    main()
