"""Load the existing action module; upstream weights remain immutable on disk.

No replacement model is implemented. The original FlowmatchingActionHead class,
SFT configuration and exact state dict are used. Frozen encoders need not occupy
GPU memory when their original-execution observation cache is available.
"""
from pathlib import Path
from types import SimpleNamespace
import pickle

import torch
from torch import nn

from starVLA.rl.flow_grpo.config import resolve_config
from starVLA.rl.flow_grpo.loading import weight_path
from starVLA.rl.flow_grpo.data import KeyedDataset
from .contracts import SFT_SHA256, file_sha, validate_parameters
from .noise_policy import KernelSpec, NoisePolicy, legacy_std_range

DEFAULT_CONFIG = 'configs/flow_grpo/frozen_action_head_batched_single8_epoch1.yaml'
DEFAULT_SOURCE = '/mnt/project/DriveDreamer-Policy-flow-grpo/artifacts/action-only-source/checkpoint_code/action-only-checkpoints-v1/configs/frozen-visual-step100000-config.yaml'


class ActionClosure(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.action_model = head


def load_action_closure(*, config=DEFAULT_CONFIG, source=DEFAULT_SOURCE, dtype=torch.bfloat16,
                        device='cuda', verify_hash=True):
    from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead
    cfg, sft = resolve_config(config, overrides=['checkpoint_contract.source_config='+str(source)])
    path = weight_path(cfg['sft_checkpoint'])
    if verify_hash and file_sha(path) != SFT_SHA256:
        raise ValueError('SFT weight hash changed')
    # Exactly the derived configuration in QwenOFT.__init__, not a new DiT.
    action = sft.framework.action_model
    action.DiTConfig = dict(num_layers=action.diffusion_model_cfg.num_layers,
                            input_embedding_dim=action.hidden_size, attention_head_dim=64,
                            num_attention_heads=action.hidden_size//64)
    head = FlowmatchingActionHead(sft)
    with torch.serialization.safe_globals([set]):
        state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    state = state.get('module', state)
    selected = {k.removeprefix('action_model.'): v for k,v in state.items() if k.startswith('action_model.')}
    head.load_state_dict(selected, strict=True)
    policy = ActionClosure(head).to(device=device, dtype=dtype).eval()
    validate_parameters(policy)
    return policy, cfg, sft


def make_noise_policy(variant, bounds, **kwargs):
    policy, cfg, sft = load_action_closure(**kwargs)
    # C_SDE ignores learned bounds; label its actual unchanged transition range.
    # This is metadata only: distribution() still calls the original SDE kernel.
    if variant=='C_SDE':bounds=legacy_std_range()[:2]
    # Same deterministic initial MLP on every rank, independent of data/RNG
    # consumed when constructing the existing action module.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        actor = NoisePolicy(policy, KernelSpec(variant, *bounds))
    actor.to(device=next(policy.parameters()).device, dtype=next(policy.parameters()).dtype).eval()
    validate_parameters(policy, actor.noise_head)
    return actor, cfg, sft


class ReplayLabels:
    """Original GT preprocessing on demand, without image I/O or label caching."""
    def __init__(self, sft):
        self.keyed = KeyedDataset(sft)

    def get(self, token):
        ds = self.keyed.dataset
        if token not in self.keyed.indices:
            raise ValueError('GT replay token outside the source population')
        with (Path(ds.base_dir)/(token+'.pkl')).open('rb') as stream:
            raw = pickle.load(stream)
        # This existing dataset switch skips image loading. No fabricated Qwen
        # tensor is sent to the actor: only the unmodified GT action is returned.
        row = ds._get_sample(raw, token, cached_features={'qwen': None})
        target = torch.as_tensor(row['action'], dtype=torch.float32)
        if target.shape != (8,4) or not torch.isfinite(target).all():
            raise ValueError('Invalid original SFT replay target')
        return target


def attach_checkpoint_processor(policy, sft_checkpoint):
    """Metadata-only adapter for the existing atomic checkpoint writer."""
    import shutil
    root = Path(sft_checkpoint)/'processor'
    if not root.is_dir():
        raise FileNotFoundError(root)

    class ProcessorCopy:
        def save_pretrained(self, destination):
            shutil.copytree(root, destination)

    policy.qwen_vl_interface = SimpleNamespace(processor=ProcessorCopy())
