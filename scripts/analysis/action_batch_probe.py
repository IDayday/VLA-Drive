"""Isolated real-head throughput/memory probe, never a production release.

Two predeclared official-reward chains, no optimizer updates or resampling.
Keep original BF16 storage / FP32 velocity arithmetic. Gradients here use native
PyTorch BF16 leaf accumulation, NOT the production ZeRO FP32 partition pathway.
Batch layout changes therefore need a separate native optimizer qualification.
"""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import time

import torch

from starVLA.rl.flow_grpo.action_head_policy import (
    FrozenFeatureStore, feature_identity, freeze_for_action_head, install_frozen_features,
)
from starVLA.rl.flow_grpo.config import resolve_config
from starVLA.rl.flow_grpo.data import to_device
from starVLA.rl.flow_grpo.loading import load_policy, file_sha
from starVLA.rl.flow_grpo.math import transition, reduce_dimensions, clipped_surrogate, conditional_kl
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.rollout import evaluate_transitions, velocity
from starVLA.rl.flow_grpo.transactions import atomic_json


def flat_transitions(policy, chain, checkpoint):
    """Use the actual trainer implementation, not a parallel probe algorithm."""
    return evaluate_transitions(policy, chain.observation, chain, checkpoint,
                                layout="flat_saved_chain")


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', required=True); p.add_argument('--bank', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--head-storage', choices=['bf16', 'fp32'], default='bf16')
    p.add_argument('--preserve-bf16-time-input', action='store_true',
                   help='FP32 leaf-gradient oracle retaining the source timestep input quantization')
    p.add_argument('--mode', choices=['serial_on', 'serial_off', 'candidate16_off', 'flat160_off', 'flat160_on'], required=True)
    a = p.parse_args(); cfg, sft = resolve_config(a.config); configure_numerics()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    report = dict(status='RUNNING', mode=a.mode, scope=__doc__, device=torch.cuda.get_device_name(),
                  torch_version=torch.__version__, script_sha256=file_sha(__file__),
                  checkpoint_sha256=cfg['checkpoint_contract']['sha256'],
                  head_storage=a.head_storage,
                  preserve_bf16_time_input=a.preserve_bf16_time_input, scenes=[])
    from starVLA.rl.flow_grpo.batch_profile import kernel_identity
    report.update(kernels=kernel_identity(), device_type='cuda')
    atomic_json(out/'report.json', report)
    # Match the trainer/precompute identity timing: model construction adds
    # resolved defaults to the SFT OmegaConf object.
    store = FrozenFeatureStore(cfg['frozen_feature_cache']['root'], feature_identity(cfg, sft))
    policy = load_policy(cfg, sft).cuda().bfloat16().eval()
    freeze_for_action_head(policy)
    # Preserve source BF16 values exactly, but remove BF16 leaf accumulation.
    if a.head_storage == 'fp32': policy.action_model.float()
    if a.preserve_bf16_time_input:
        if a.head_storage != 'fp32': raise ValueError('oracle requires FP32 leaves')
        # The source TimestepEncoder explicitly casts the sinusoid to parameter
        # dtype before its FP32-autocast MLP. Hold those VALUES fixed while moving
        # trainable leaves to FP32, so this isolates backward accumulation instead
        # of also changing the timestep embedding input. Diagnostic only.
        policy.action_model.model.timestep_encoder.time_proj.register_forward_hook(
            lambda module, inputs, output: output.to(torch.bfloat16))
    install_frozen_features(policy, store)
    trainable = {n: p for n, p in policy.named_parameters() if p.requires_grad}
    report['observed_head_dtypes'] = sorted({str(p.dtype) for p in trainable.values()})
    report['trainable_tensors'] = len(trainable)
    report['trainable_numel'] = sum(p.numel() for p in trainable.values())
    checkpoint = a.mode.endswith('_on')
    # Fixed positions 1 and 2, selected for their previously recorded nonzero
    # official reward advantages, never selected by success of this comparison.
    for rank in [1, 2]:
        chain = to_device(torch.load(Path(a.bank)/f'rollout_rank{rank}_v0.pt',
                                    map_location='cpu', weights_only=False)[0], 'cuda')
        row = dict(rank=rank, tokens=list(chain.observation.tokens),
                   official_nonzero_advantage=bool(chain.advantages.abs().max()>0), repeats=[])
        assert chain.chain.shape[1:3] == (16, 11)
        assert row['official_nonzero_advantage']
        if a.mode == 'candidate16_off':
            chain = replace(chain, spec=replace(chain.spec, candidate_chunk_size=16))
        for repeat in range(2):
            policy.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            try:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    stats = (flat_transitions(policy, chain, checkpoint) if a.mode.startswith('flat')
                             else evaluate_transitions(policy, chain.observation, chain, checkpoint))
                    lp = reduce_dimensions(stats['elementwise_logprob'], chain.dimension_mask, chain.spec.reduction)
                    pg, ratio = clipped_surrogate(lp, chain.old_logprob, chain.advantages[:, :, None], .02)
                    kl = reduce_dimensions(conditional_kl(stats['mean'], stats['std'], chain.reference_mean,
                                           chain.reference_std, chain.spec.temporal_noise_correlation),
                                           chain.dimension_mask, chain.spec.reduction)
                    # RL + original reference term, without replay/optimizer:
                    # this is a kernel/memory pretest, not a native training step.
                    loss = ((pg + .04*kl)*chain.transition_mask).sum()/chain.transition_mask.sum()
                torch.cuda.synchronize(); forward_seconds = time.monotonic()-start
                loss.backward(); torch.cuda.synchronize()
                measurement = dict(forward_seconds=forward_seconds, total_seconds=time.monotonic()-start,
                                   peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                                   peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                                   loss=float(loss), ratio_min=float(ratio.min()), ratio_max=float(ratio.max()),
                                   gradient_tensors=sum(p.grad is not None for p in trainable.values()),
                                   gradient_finite=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in trainable.values()))
                row['repeats'].append(measurement)
                if repeat == 0:
                    torch.save({k: v.detach().cpu() for k, v in stats.items()}, out/f'stats_rank{rank}.pt')
                    torch.save({n: p.grad.detach().cpu() for n, p in trainable.items()}, out/f'grads_rank{rank}.pt')
                del stats, lp, pg, kl, loss, ratio
                print(json.dumps(dict(mode=a.mode, rank=rank, repeat=repeat, **measurement)), flush=True)
            except torch.cuda.OutOfMemoryError as exc:
                row['oom'] = dict(error=str(exc), seconds=time.monotonic()-start,
                                 peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
                report['scenes'].append(row); report['status']='OOM'
                atomic_json(out/'report.json', report)
                raise
        report['scenes'].append(row); atomic_json(out/'report.json', report)
    report['status']='MEASURED_NOT_QUALIFIED'
    atomic_json(out/'report.json', report)


if __name__ == '__main__': main()
