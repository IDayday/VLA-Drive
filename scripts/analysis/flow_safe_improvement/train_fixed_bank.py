"""Complete action-model extraction with the audited eight-GPU ZeRO-2 engine.

Uses existing parameter groups, precision correction, optimizer finite gates,
atomic full checkpoints/resume, and original inference export. No old training
controller is invoked. Candidate/behavior identities remain fixed throughout.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from accelerate.utils import set_seed

from starVLA.rl.flow_grpo.action_head_policy import prefix_parameters
from starVLA.rl.flow_grpo.checkpoint import save_boundary, resume_boundary, export_checkpoint
from starVLA.rl.flow_grpo.config import resolve_config
from starVLA.rl.flow_grpo.contracts import optimizer_groups, parameter_manifest
from starVLA.rl.flow_grpo.data import keys_for_positions
from starVLA.rl.flow_grpo.distributed import initialize_control_group, rank0_call
from starVLA.rl.flow_grpo.loading import weight_path
from starVLA.rl.flow_grpo.model import make_reference
from starVLA.rl.flow_grpo.monitor import GradientMonitor, CheckedDeepSpeedBackward, ParameterProbe, save_full_optimizer_gradients, optimizer_norm_metrics
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.stability import lr_multiplier
from starVLA.rl.flow_grpo.trainer import make_accelerator
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.flow_grpo.zero2_precision import install_fp32_partitions
from starVLA.rl.safe_flow_improvement.contracts import PopulationContract, SFT_SHA256, file_sha, fingerprint, seed_for
from starVLA.rl.safe_flow_improvement.sampling import load_cached_policy
from starVLA.rl.safe_flow_improvement.training_actor import ExtractionActor


class Cursor:
    def __init__(self, identity, rank, world):
        self.identity, self.rank, self.world, self.consumed = identity, rank, world, 0

    def state_dict(self):
        return dict(identity=self.identity, rank=self.rank, world=self.world, consumed=self.consumed)

    def load_state_dict(self, value):
        if any(value[k] != getattr(self, k) for k in ('identity', 'rank', 'world')):
            raise ValueError('Resume changed bank/window/rank')
        self.consumed = value['consumed']


def prefix_digest(policy):
    h = hashlib.sha256()
    for name, value in prefix_parameters(policy):
        if value.requires_grad or value.grad is not None:
            raise ValueError('Frozen prefix became trainable')
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load_sealed(root, token, identity=None):
    path = Path(root)/'scenes'/token[:2]/(token+'.pt')
    seal = json.loads(path.with_suffix('.json').read_text())
    if seal['payload_sha256'] != file_sha(path) or (identity is not None and seal['identity_sha256'] != identity):
        raise ValueError('Immutable target/reference changed')
    return torch.load(path, map_location='cpu', weights_only=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--population', required=True)
    p.add_argument('--cohorts', required=True)
    p.add_argument('--targets', required=True)
    p.add_argument('--reference-scores', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--training-seed', type=int, default=42)
    p.add_argument('--retention-coefficient', type=float, required=True)
    p.add_argument('--retention-registration', help='Required for formal research, calibrated on RL_TRAIN only')
    p.add_argument('--initial-behavior')
    p.add_argument('--initial-behavior-sha256')
    p.add_argument('--resume')
    p.add_argument('--stop-after-updates', type=int)
    p.add_argument('--qualification', action='store_true')
    p.add_argument('--save-every', type=int, default=512)
    p.add_argument('--config', default='configs/flow_grpo/frozen_action_head_batched_single8_epoch1.yaml')
    p.add_argument('--source-config', default='/mnt/project/DriveDreamer-Policy-flow-grpo/artifacts/action-only-source/checkpoint_code/action-only-checkpoints-v1/configs/frozen-visual-step100000-config.yaml')
    a = p.parse_args()
    if a.epochs <= 0 or a.retention_coefficient < 0 or (not a.qualification and not a.retention_registration):
        raise ValueError('Formal learning requires a fixed common retention calibration')
    if a.retention_registration:
        retention_spec = json.loads(Path(a.retention_registration).read_text())
        if retention_spec['selected_coefficient'] != a.retention_coefficient:
            raise ValueError('Retention differs from common training-only calibration')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    configure_numerics()
    pop, cohorts, targets = [json.loads(Path(x).read_text()) for x in (a.population, a.cohorts, str(Path(a.targets)/'manifest.json'))]
    complete = json.loads((Path(a.targets)/'COMPLETE.json').read_text())
    if complete['identity_sha256'] != targets['sha256'] or targets['population_identity'] != pop['identity_sha256'] or cohorts['population_identity'] != pop['identity_sha256']:
        raise ValueError('Incomplete/mismatched target population')
    population = PopulationContract(*[frozenset(pop[k]) for k in ('train_tokens', 'dev_tokens', 'navtest_tokens')], pop['token_logs'])
    tokens, safe_tokens = targets['tokens'], cohorts['reference_joint_safe']
    population.require_training(tokens, 'positive extraction / NFT')
    population.require_training(safe_tokens, 'reference-safe retention')
    if len(tokens) % 16 or (not a.qualification and len(tokens) != 4096):
        raise ValueError('Use the registered global scene batch 16 with complete scene epochs')
    total = len(tokens)*a.epochs//16
    if total < 2:
        raise ValueError('Learning window too small')
    mode = 'nft' if targets['objective'] == 'nft' else 'positive_fm'
    if targets.get('qualification_only', False) and not a.qualification:
        raise ValueError('Qualification targets cannot enter formal method learning')
    initialization = a.initial_behavior_sha256 or SFT_SHA256
    if targets['behavior_sha256'] != initialization:
        raise ValueError('Fixed bank behavior and optimizer initialization disagree')
    # Preserve the original trainer's initialization order. Qwen's logger
    # creates PartialState while importing the model; the DeepSpeed plugin
    # must be configured before that import, not after model construction.
    engine_cfg, _ = resolve_config(a.config, overrides=['checkpoint_contract.source_config='+a.source_config])
    accelerator = make_accelerator(engine_cfg)
    initialize_control_group(1800)
    if accelerator.num_processes != 8 or engine_cfg['runtime']['deepspeed_stage'] != 2:
        raise ValueError('This new learning profile requires the registered single-host eight-GPU ZeRO-2 configuration')
    set_seed(a.training_seed, device_specific=True)
    policy, store, cfg, sft, loaded_identity = load_cached_policy(a.config, a.source_config,
                                                               behavior=a.initial_behavior, behavior_sha256=a.initial_behavior_sha256)
    assert loaded_identity == initialization
    # All reference velocity targets remain the original SFT even in later
    # rounds. The current behavior is a separate frozen dependency for NFT.
    reference = make_reference(policy)
    if initialization != SFT_SHA256:
        with torch.serialization.safe_globals([set]):
            original = torch.load(weight_path(cfg['sft_checkpoint']), mmap=True, weights_only=True, map_location='cpu')
        original = original.get('module', original)
        reference.action_model.load_state_dict({k.removeprefix('action_model.'): v for k, v in original.items() if k.startswith('action_model.')}, strict=True)
        del original
    old_behavior = make_reference(policy) if mode == 'nft' else None
    frozen_before = prefix_digest(policy)
    cfg['reward_version'] = 'navsim_v1.1_independent_reference_candidate'
    cfg['algorithm'] = {'kind': mode, 'logprob_reduction': 'NOT_APPLICABLE_ODE_EXTRACTION', 'fixed_bank': targets['sha256']}
    cfg['runtime'].update(output_dir=str(Path(a.output).resolve()), max_updates=total, accumulation_steps=2,
                          seed=a.training_seed, scene_microbatch=1, activation_checkpointing=False)
    cfg['optimizer']['schedule'] = dict(type='warmup_cosine', total_updates=total,
                                       warmup_updates=max(1, min(total-1, math.ceil(total*.03))), min_lr_ratio=.1)
    cfg['safe_flow'] = {'target_manifest': targets, 'epochs': a.epochs, 'retention_coefficient': a.retention_coefficient,
                        'reference_replay_population': 'all_RL_TRAIN', 'retention_population_sha256': fingerprint(safe_tokens),
                        'initial_behavior': initialization, 'qualification_not_method_result': a.qualification,
                        'retention_registration_sha256': file_sha(a.retention_registration) if a.retention_registration else None}
    from starVLA.rl.safe_flow_improvement import training_actor, reference_retention, nft_adapter, weighted_flow_objective
    provenance = {'schema': 'flow_safe_extraction_run_v1', 'sft_sha256': SFT_SHA256, 'behavior_sha256': initialization,
                  'feature_identity': store.identity_sha, 'target_manifest_sha256': targets['sha256'],
                  'target_completion_sha256': file_sha(Path(a.targets)/'COMPLETE.json'), 'cohort_sha256': cohorts['sha256'],
                  'population_identity': pop['identity_sha256'], 'protocol_sha256': targets['protocol_sha256'],
                  'source_files': {Path(x).name: file_sha(x) for x in (__file__, training_actor.__file__, reference_retention.__file__, nft_adapter.__file__, weighted_flow_objective.__file__)}}
    provenance['sha256'] = fingerprint(provenance)
    out = Path(a.output)
    def prepare():
        if out.exists() and not a.resume:
            raise FileExistsError('New learning must not overwrite an existing run')
        out.mkdir(parents=True, exist_ok=True)
        if (out/'run_identity.json').exists() and json.loads((out/'run_identity.json').read_text()) != provenance:
            raise ValueError('Resume changed learning identity')
        atomic_json(out/'run_identity.json', provenance)
        atomic_json(out/'registered_config.json', cfg)
        snapshot = out/'source_snapshot'
        snapshot.mkdir(exist_ok=True)
        for source in (__file__, training_actor.__file__, reference_retention.__file__, nft_adapter.__file__, weighted_flow_objective.__file__):
            destination = snapshot/Path(source).name
            if destination.exists() and destination.read_bytes() != Path(source).read_bytes():
                raise ValueError('Saved learning source differs')
            if not destination.exists():
                destination.write_bytes(Path(source).read_bytes())
    rank0_call(prepare, accelerator.device)
    groups = optimizer_groups(policy, sft, cfg['optimizer']['initial_lr_multiplier'])
    manifest = parameter_manifest(policy, groups)
    if manifest['trainable_numel'] != 819503620:
        raise ValueError('Must train the complete original action_model, including qwen_proj')
    if abs(max(g['lr'] for g in groups)-1e-6) > 1e-12:
        raise ValueError('Registered peak learning rate is 1e-6')
    rank0_call(lambda: atomic_json(out/'sft_parameter_manifest.json', manifest), accelerator.device)
    opt = sft.trainer.optimizer
    optimizer = torch.optim.AdamW(groups, betas=tuple(opt.betas), eps=opt.eps, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_multiplier(step, cfg['optimizer']['schedule']))
    monitor = GradientMonitor(policy)
    actor = ExtractionActor(policy)
    actor, optimizer, scheduler = accelerator.prepare(actor, optimizer, scheduler)
    precision = install_fp32_partitions(actor)
    rank0_call(lambda: atomic_json(out/'zero2_precision.json', precision), accelerator.device)
    def gradient_probe(engine):
        if a.qualification and engine.global_steps in (0, total-1):
            save_full_optimizer_gradients(engine, out/'optimizer_gradients', save_tensors=False)
    accelerator.deepspeed_engine_wrapped = CheckedDeepSpeedBackward(actor, monitor, accelerator.device, boundary_probe=gradient_probe)
    actor.eval()
    policy = accelerator.unwrap_model(actor).policy
    reference.to(device=accelerator.device, dtype=next(policy.action_model.parameters()).dtype).eval()
    if old_behavior is not None:
        old_behavior.to(device=accelerator.device, dtype=next(policy.action_model.parameters()).dtype).eval()
    cursor = Cursor(provenance['sha256'], accelerator.process_index, accelerator.num_processes)
    update = 0
    if a.resume:
        update, _, pending = resume_boundary(Path(a.resume), accelerator, actor, cfg, provenance, [cursor])
        if pending is not None or cursor.consumed != update*2:
            raise ValueError('Unexpected extraction data/update boundary')
    parameter_probe = ParameterProbe(policy)
    stop = min(total, a.stop_after_updates) if a.stop_after_updates is not None else total
    epoch_checkpoints = {len(tokens)*epoch//16 for epoch in (1, 2, 4, 8, 16) if epoch <= a.epochs}
    if a.qualification:
        epoch_checkpoints = {total//2, total}
    started = time.monotonic()
    last_checkpoint = None
    while update < stop:
        t0 = time.monotonic(); losses = []
        for micro in range(2):
            position = cursor.consumed*8 + accelerator.process_index
            token, _ = next(keys_for_positions(tokens, a.training_seed, [position]))
            row = load_sealed(a.targets, token, targets['sha256'])
            generation_key = seed_for('extraction_scene', a.training_seed, token, position)
            choice = int(torch.multinomial(row['target_probabilities'], 1, generator=torch.Generator().manual_seed(seed_for('candidate_target_draw', generation_key, token))))
            target = row['actions'][choice:choice+1].to(accelerator.device).float()
            replay_token, _ = next(keys_for_positions(pop['train_tokens'], a.training_seed+104729, [position]))
            safe_token, _ = next(keys_for_positions(safe_tokens, a.training_seed+130363, [position]))
            safe_reference = load_sealed(a.reference_scores, safe_token)
            safe_choice = seed_for('reference_target_draw', generation_key, safe_token) % 4
            current_value, replay_value, safe_value = [store.get(x) for x in (token, replay_token, safe_token)]
            with accelerator.accumulate(actor):
                with accelerator.autocast():
                    result = actor(raw=current_value['raw'].to(accelerator.device), target=target,
                                   replay_raw=replay_value['raw'].to(accelerator.device), replay_target=replay_value['action'][None].to(accelerator.device).float(),
                                   retention_raw=safe_value['raw'].to(accelerator.device), retention_target=safe_reference['sample']['normalized_actions'][safe_choice:safe_choice+1].to(accelerator.device).float(),
                                   reference=reference, seed=generation_key, retention_coefficient=a.retention_coefficient, mode=mode,
                                   old_behavior=old_behavior, advantage=row['advantages'][choice:choice+1].to(accelerator.device) if mode == 'nft' else None)
                accelerator.backward(result['loss'])
                optimizer.step(); optimizer.zero_grad()
            losses.append({k: float(result[k].detach()) for k in ('loss', 'improve', 'replay', 'reference_velocity')})
            cursor.consumed += 1
            del result
        update += 1; scheduler.step()
        if actor.global_steps != update or scheduler.state_dict()['last_epoch'] != update:
            raise ValueError('Optimizer/scheduler did not advance exactly once per 16 scenes')
        metrics = {k: float(np.mean([x[k] for x in losses])) for k in losses[0]}
        metrics.update(monitor.consume()); metrics.update(parameter_probe.consume())
        metrics.update(optimizer_norm_metrics(actor, 2, clip=1.))
        metrics.update(update=update, scene_exposures_global=update*16, epoch=update*16/len(tokens), rank=accelerator.process_index,
                       seconds=time.monotonic()-t0, cumulative_wall_seconds=time.monotonic()-started,
                       velocity_forward_calls_per_scene=5 if mode == 'nft' else 4,
                       velocity_backward_calls_per_scene=3, FM_noise_repeats=8,
                       peak_gpu_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
        with (out/f'training_rank{accelerator.process_index}.jsonl').open('a') as stream:
            stream.write(json.dumps(metrics, allow_nan=False)+'\n')
        if accelerator.is_main_process:
            print(json.dumps(metrics), flush=True)
        if update in epoch_checkpoints or update % a.save_every == 0 or update == stop:
            save_cfg = copy.deepcopy(cfg)
            save_cfg['runtime']['output_dir'] = str(out/('milestones' if update in epoch_checkpoints else 'rolling'))
            last_checkpoint = save_boundary(accelerator, actor, save_cfg, sft, manifest, provenance, update, 0, [cursor])
            rank0_call(lambda: atomic_json(out/'latest_checkpoint.json', {'path': str(last_checkpoint), 'update': update, 'epoch': update*16/len(tokens), 'milestone': update in epoch_checkpoints}), accelerator.device)
            if update in epoch_checkpoints or update == stop:
                rank0_call(lambda: export_checkpoint(last_checkpoint, out/f'export_update{update:06d}'), accelerator.device)
    frozen_after = prefix_digest(policy)
    if frozen_before != frozen_after:
        raise ValueError('Frozen encoder prefix changed')
    if any(p.grad is not None or p.requires_grad for p in reference.parameters()):
        raise ValueError('Permanent SFT reference was trained')
    report = {'status': 'COMPLETE_FIXED_BUDGET' if update == total else 'STOPPED_AT_SAVED_BOUNDARY',
              'identity_sha256': provenance['sha256'], 'rank': accelerator.process_index, 'updates': update,
              'total_registered_updates': total, 'global_scene_exposures': update*16,
              'frozen_prefix_sha256_before': frozen_before, 'frozen_prefix_sha256_after': frozen_after,
              'all_encoder_parameters_frozen': True, 'reference_frozen': True,
              'qualification_only': a.qualification, 'fresh_ODE_evaluation_required': True,
              'checkpoint': str(last_checkpoint), 'wall_seconds_this_invocation': time.monotonic()-started}
    atomic_json(out/f'result_rank{accelerator.process_index}.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
