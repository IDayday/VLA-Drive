"""Explicit action-head RL contract, authorized separately from full-SFT RL.

The cache boundary is BEFORE action_model.qwen_proj. Only the frozen encoder's
eight normalized action token states are reused; current/reference projections
and DiT computations keep their own parameters. The original SFT forward is used.
"""
import fcntl
import json
import os
from pathlib import Path
import types
import uuid
from collections import OrderedDict

import numpy as np
import torch
from .contracts import digest
from .loading import file_sha
from .transactions import atomic_json


def prefix_parameters(policy):
    head = {id(p) for p in policy.action_model.parameters()}
    return [(n, p) for n, p in policy.named_parameters() if id(p) not in head]


def freeze_for_action_head(policy):
    """Intersect original SFT trainables with the actual action module objects."""
    head = {id(p) for p in policy.action_model.parameters()}
    outside = {id(p) for name, child in policy.named_children()
               if child is not policy.action_model for p in child.parameters()}
    outside.update(id(p) for p in policy.parameters(recurse=False))
    if head & outside:
        raise ValueError("action head shares parameter objects with frozen encoder")
    if not any(p.requires_grad for p in policy.action_model.parameters()):
        raise ValueError("source action head has no trainable parameters")
    frozen = []
    for name, p in policy.named_parameters():
        if id(p) not in head:
            p.requires_grad_(False)
            frozen.append(name)
    return tuple(frozen)


def assert_frozen_prefix(policy):
    if policy.training or any(p.requires_grad or p.grad is not None for _, p in prefix_parameters(policy)):
        raise ValueError("feature caching requires an eval, fully frozen encoder")


def feature_identity(cfg, sft):
    from omegaconf import OmegaConf
    from .reproducibility import processor_identity, dependency_versions, numerical_profile
    sources = ['starVLA/model/framework/QwenOFT.py', 'starVLA/model/framework/baseline_qwen.py',
               'starVLA/model/modules/vlm/QWen3.py', 'starVLA/dataloader/navsim_dataset.py', __file__]
    sources += [str(p) for p in sorted(Path('starVLA/model/modules/vlm/qwen3_vl').glob('*.py'))]
    # Parameter update topology/optimizer do not change a frozen prefix output.
    numeric = numerical_profile(cfg, sft)
    numeric = {k: numeric[k] for k in ('model_dtype','qwen_autocast','action_history_projector_autocast',
               'attention_backend','deterministic','cudnn_deterministic','cudnn_benchmark',
               'matmul_tf32','cudnn_tf32','bf16_reduced_precision_reduction','float32_matmul_precision',
               'flash_deterministic','cublas_workspace')}
    return {'schema': 1, 'boundary': 'normalized_action_tokens_before_action_model.qwen_proj',
            'checkpoint_sha256': cfg['checkpoint_contract']['sha256'],
            'processor': processor_identity(cfg['paths']['base_vlm']),
            'data_manifest_identity': cfg['paths']['asset_manifest_identity'],
            'data_root': str(Path(cfg['paths']['data_root']).resolve()),
            'sft_config': digest(OmegaConf.to_container(sft, resolve=True)),
            'dependencies': dependency_versions(), 'numerics': numeric,
            'sources': {Path(p).name if Path(p).is_absolute() else p: file_sha(p) for p in sources}}


class FrozenFeatureStore:
    """Per-token atomic records; concurrent precompute/consumer never mix payloads.

    Identity is locked before any entry. Entry seals bind the tensor AND replay
    labels, but only raw observation token states are returned to policy encoding.
    COMPLETE is optional for read-through operation, mandatory for readonly mode.
    """
    def __init__(self, root, identity, *, readonly=False):
        self.root = Path(root)
        self.identity = identity
        self.identity_sha = digest(identity)
        self.readonly = readonly
        self.memory = OrderedDict()
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / 'identity.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.root / 'identity.json'
            if path.exists():
                if json.loads(path.read_text()) != identity:
                    raise ValueError('frozen feature cache identity conflict')
            elif readonly:
                raise ValueError('readonly feature cache is not initialized')
            else:
                atomic_json(path, identity)
        if readonly:
            marker = json.loads((self.root / 'COMPLETE').read_text())
            if marker['identity_sha256'] != self.identity_sha or marker['manifest_sha256'] != file_sha(self.root / 'manifest.json'):
                raise ValueError('feature cache publication changed')

    def paths(self, token):
        if not token or any(c not in '0123456789abcdef' for c in token):
            raise ValueError('invalid NAVSIM token')
        base = self.root / token[:2] / token
        return base.with_suffix('.pt'), base.with_suffix('.json')

    def get(self, token):
        if token in self.memory:
            self.memory.move_to_end(token)
            return self.memory[token]
        path, seal = self.paths(token)
        if not seal.exists():
            if self.readonly: raise FileNotFoundError(seal)
            return None
        info = json.loads(seal.read_text())
        if (info['identity_sha256'] != self.identity_sha or info['token'] != token
                or info['sha256'] != file_sha(path)):
            raise ValueError('frozen feature record identity/content mismatch: ' + token)
        value = torch.load(path, weights_only=True, map_location='cpu')
        if (value['token'] != token or value['raw'].ndim != 3 or value['raw'].shape[0] != 1
                or value['raw'].requires_grad or not torch.isfinite(value['raw']).all()
                or not torch.isfinite(value['state']).all()
                or ('action' in value and not torch.isfinite(value['action']).all())):
            raise ValueError('invalid frozen feature tensor: ' + token)
        self.memory[token] = value
        if len(self.memory) > 64: self.memory.popitem(last=False)
        return value

    def put(self, token, raw, example):
        if self.readonly: raise ValueError('cannot write readonly feature cache')
        path, seal = self.paths(token)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            old = self.get(token)
            if old is not None:
                if not torch.equal(old['raw'], raw.detach().cpu()):
                    raise ValueError('different frozen features for the same token')
                return old
            raw = raw.detach().cpu().contiguous()
            if not torch.isfinite(raw).all(): raise ValueError('nonfinite frozen feature')
            # Only replay labels are optional; no labels enter prefix encoding.
            value = {'token': token, 'raw': raw, 'lang': example['lang'],
                     'state': torch.as_tensor(np.asarray(example['state'])).clone()}
            if 'action' in example:
                value['action'] = torch.as_tensor(np.asarray(example['action'])).clone()
            attempt = path.with_suffix('.attempt-' + uuid.uuid4().hex)
            torch.save(value, attempt)
            if path.exists():
                path.rename(path.with_suffix('.interrupted-' + uuid.uuid4().hex))
            os.replace(attempt, path)
            atomic_json(seal, {'token': token, 'identity_sha256': self.identity_sha,
                               'sha256': file_sha(path), 'has_replay': 'action' in value})
            return self.get(token)

    def finalize(self, tokens):
        if len(tokens) != len(set(tokens)): raise ValueError('duplicate cache tokens')
        entries = []
        for token in sorted(tokens):
            self.memory.pop(token, None)
            value = self.get(token)
            if value is None or 'action' not in value:
                raise ValueError('missing complete feature/replay entry: ' + token)
            _, seal = self.paths(token)
            entries.append(json.loads(seal.read_text()))
        manifest = {'identity_sha256': self.identity_sha, 'entries': entries}
        path = self.root / 'manifest.json'
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError('published feature token/content conflict')
        if not path.exists(): atomic_json(path, manifest)
        marker = {'identity_sha256': self.identity_sha, 'count': len(tokens), 'manifest_sha256': file_sha(path)}
        complete = self.root / 'COMPLETE'
        if complete.exists() and json.loads(complete.read_text()) != marker:
            raise ValueError('feature completion conflict')
        if not complete.exists(): atomic_json(complete, marker)
        return marker


def install_frozen_features(policy, store):
    """Replace only frozen prefix encoding; original projection/SFT stay intact."""
    assert_frozen_prefix(policy)
    cfg = policy.config
    if (cfg.datasets.video_data.load_2d_data or cfg.datasets.gs_data.load_3d_data
            or cfg.datasets.reward_data.load_reward_data or cfg.get('w_depth', 0)):
        raise ValueError('action feature cache cannot serve auxiliary branches')
    original = policy.encode_policy_features
    def cached(self, examples, feature_output=None):
        assert_frozen_prefix(self)
        convention = feature_output or self.config.framework.qwenvl.get('sft_feature_output', 'normalized')
        if convention != 'normalized': raise ValueError('cached feature convention mismatch')
        from starVLA.model.framework.baseline_qwen import extract_baseline_action_conditions
        values = []
        for example in examples:
            row = store.get(example['token'])
            if row is None:
                if not example.get('image'): raise ValueError('cache miss without current images')
                dtype = next(self.qwen_vl_interface.parameters()).dtype
                device = next(self.qwen_vl_interface.parameters()).device
                with torch.no_grad(), torch.autocast(device.type, dtype=dtype,
                                                    enabled=dtype != torch.float32):
                    hidden, positions = original([example], feature_output=convention)
                    raw = extract_baseline_action_conditions(hidden, positions['action'])
                row = store.put(example['token'], raw, example)
            if (row['lang'] != example['lang'] or not np.array_equal(row['state'].numpy(), example['state'])):
                raise ValueError('cached observation state/instruction differs')
            values.append(row['raw'])
        device = next(self.action_model.parameters()).device
        raw = torch.cat(values, 0).to(device)
        positions = torch.arange(raw.shape[1], device=device)[None].expand(raw.shape[0], -1)
        return raw, {'action': positions}
    policy.encode_policy_features = types.MethodType(cached, policy)
    policy._flow_feature_identity = store.identity_sha


class CachedKeyedDataset:
    def __init__(self, original, store, encoder):
        self.original, self.store, self.encoder = original, store, encoder

    def __getitem__(self, key):
        token, seed = key
        row = self.store.get(token)
        if row is None:
            sample = self.original[key]
            self.encoder([sample])
            row = self.store.get(token)
        if row is None or 'action' not in row:
            raise ValueError('feature cache record lacks original replay supervision')
        return {'image': (), 'lang': row['lang'], 'state': row['state'].numpy().copy(),
                'token': token, 'action': row['action'].numpy().copy(), '_flow_sample_seed': int(seed)}
