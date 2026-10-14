"""CPU comparison against an unmodified frozen native head; never load a VLM/GPU."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead


def tiny_config():
    return OmegaConf.create({'framework': {'qwenvl': {'vl_hidden_dim': 24}, 'action_model': {
        'hidden_size': 32, 'action_dim': 4, 'action_horizon': 8, 'num_inference_timesteps': 10,
        'noise_beta_alpha': 1.5, 'noise_beta_beta': 1., 'noise_s': .999,
        'num_timestep_buckets': 1000, 'add_pos_embed': True, 'max_seq_len': 16,
        'DiTConfig': {'input_embedding_dim': 32, 'num_layers': 2,
                      'num_attention_heads': 4, 'attention_head_dim': 8},
        'diffusion_model_cfg': {'cross_attention_dim': 32, 'dropout': .2,
            'final_dropout': True, 'interleave_self_attention': True,
            'norm_type': 'ada_norm', 'output_dim': 32, 'positional_embeddings': None}}}})


def verify(reference_source):
    path = Path(reference_source) / 'starVLA/model/modules/action_model/GR00T_ActionHeader.py'
    spec = importlib.util.spec_from_file_location('frozen_native_action_header', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    torch.manual_seed(42); original = module.FlowmatchingActionHead(tiny_config())
    torch.manual_seed(42); changed = FlowmatchingActionHead(tiny_config())
    assert original.state_dict().keys() == changed.state_dict().keys()
    assert all(torch.equal(v, changed.state_dict()[k]) for k, v in original.state_dict().items())
    condition = torch.randn(2, 8, 24); noise = torch.randn(2, 8, 4)
    target = torch.randn_like(noise); times = torch.tensor([.2, .8])
    old_loss, new_loss = [], []
    for model, losses in ((original, old_loss), (changed, new_loss)):
        model.train(); torch.manual_seed(71)
        loss = model(condition, target, noise=noise, times=times)
        loss.backward(); losses.append(loss.detach())
    assert torch.equal(old_loss[0], new_loss[0])
    gradient_errors = []
    for (name, a), (other, b) in zip(original.named_parameters(), changed.named_parameters()):
        assert name == other and (a.grad is None) == (b.grad is None)
        if a.grad is not None: gradient_errors.append((a.grad - b.grad).abs().max().item())
    assert max(gradient_errors) == 0
    original.eval(); changed.eval()
    before = original.predict_action(condition, initial_noise=noise)
    after = changed.predict_action(condition, initial_noise=noise)
    assert torch.equal(before, after)
    differentiable = changed.rollout_action(condition, initial_noise=noise, checkpoint_steps=True, inference_mode=True)
    assert torch.equal(after, differentiable)
    return {
        'passed': True, 'device': 'cpu', 'dtype': 'float32', 'torch': torch.__version__,
        'scope': 'two scenes, small real DiT, 10 native Euler steps; not full-checkpoint GPU qualification',
        'reference_source': str(Path(reference_source).resolve()),
        'reference_action_header_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'state_dict_keys_and_initial_parameters_exact': True,
        'default_fm_loss_max_error': (old_loss[0] - new_loss[0]).abs().item(),
        'default_fm_parameter_gradient_max_error': max(gradient_errors),
        'native_prediction_max_error': (before - after).abs().max().item(),
        'differentiable_rollout_native_prediction_max_error': (after - differentiable).abs().max().item(),
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--reference-source', required=True)
    p.add_argument('--output', type=Path)
    a = p.parse_args(); torch.set_num_threads(2)
    result = verify(a.reference_source)
    print(json.dumps(result, indent=2))
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__': main()
