"""Generate pre-qwen_proj observations using the original VLAAgent execution.

Canonical prefix batch is one scene (the original deployment unit). Qualification
then covers actual action batches B=1/2 and candidate chunks 1/16. Nothing after
the trainable projection, and no replay labels, is saved in this cache.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.rl.constrained_flow_noise.condition_cache import ConditionStore
from starVLA.rl.constrained_flow_noise.contracts import CONDITION_BOUNDARY, SFT_SHA256, file_sha, fingerprint, require_dispatch, seed_for
from starVLA.rl.flow_grpo.data import KeyedDataset
from starVLA.rl.flow_grpo.observation import prepare_policy_observation
from starVLA.rl.flow_grpo.reproducibility import configure_numerics, dependency_versions, processor_identity
from starVLA.rl.flow_grpo.transactions import atomic_json

SFT_EXPORT = '/mnt/project/DriveDreamer-Policy-flow-grpo/artifacts/action-only-checkpoints-v1/frozen_visual'
VLM = '/mnt/project/DriveDreamer-Policy/models/Qwen3-VL-2B-WorldAction'
DATA_ROOT = '/mnt/project/DriveDreamer-Policy-paired/runs/paired_full_assets_v1/dataset'
TRAIN_LIST = '/mnt/project/DriveDreamer-Policy-paired/runs/paired_full_assets_v1/navtrain_tokens.json'


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--population', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--export', default=SFT_EXPORT)
    p.add_argument('--qualification-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--finalize', action='store_true')
    p.add_argument('--priority-tokens', help='JSON list of TRAIN tokens to encode first; does not change population')
    p.add_argument('--progress-prefix', default='', help='Distinct administrative receipt names for an additional producer; feature identity unchanged')
    a = p.parse_args()
    host = require_dispatch(); configure_numerics()
    pop = json.loads(Path(a.population).read_text())
    tokens = pop['train_tokens']
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    if any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in a.progress_prefix):
        raise ValueError('Invalid producer receipt prefix')
    progress_name = f'{a.progress_prefix}rank{rank}.json'
    qualification_name = f'{a.progress_prefix}qualification_rank{rank}.json'
    out = Path(a.output)
    if a.finalize:
        identity = json.loads((out / 'identity.json').read_text())
        if identity['population_identity'] != pop['identity_sha256'] or identity['tokens_sha256'] != fingerprint(tokens):
            raise ValueError('Population changed')
        ConditionStore(out, identity, readonly=False).finalize(tokens)
        print(json.dumps({'status': 'COMPLETE', 'scenes': len(tokens)}), flush=True)
        return
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    if file_sha(Path(a.export) / 'pytorch_model.pt') != SFT_SHA256:
        raise ValueError('Reference SFT identity mismatch')
    cfg = OmegaConf.load(Path(a.export) / 'config.yaml')
    os.environ['BASE_VLM'] = VLM
    os.environ['VLM_ATTN_IMPLEMENTATION'] = cfg.framework.qwenvl.attn_implementation
    from infer import VLAAgent
    import infer
    from starVLA.model.framework.baseline_qwen import extract_baseline_action_conditions
    from starVLA.rl.safe_flow_improvement.policy_adapter import encode_condition, sample_ode
    agent = VLAAgent(a.export, device='cuda', qwen_forward_mode='optimized')
    model = agent.model.eval().requires_grad_(False)
    profile = {'qwen_dtype': str(next(model.qwen_vl_interface.parameters()).dtype),
               'history_dtype': str(next(model.action_input_model.parameters()).dtype),
               'action_dtype': str(next(model.action_model.parameters()).dtype),
               'attention': cfg.framework.qwenvl.attn_implementation,
               'prefix_scene_batch': 1, 'root_dtype': 'torch.bfloat16', 'feature_output': 'normalized',
               'qwen_forward_mode': 'optimized',
               'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
               'cublas_workspace': os.environ['CUBLAS_WORKSPACE_CONFIG'],
               'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
               'cudnn_tf32': torch.backends.cudnn.allow_tf32,
               'bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
    if (profile['qwen_dtype'], profile['history_dtype'], profile['action_dtype']) != ('torch.bfloat16', 'torch.float32', 'torch.float32'):
        raise ValueError('Original VLAAgent module dtypes changed')
    sources = [infer.__file__, model.encode_policy_condition.__code__.co_filename,
               model.encode_policy_features.__code__.co_filename,
               extract_baseline_action_conditions.__code__.co_filename,
               model.action_model.predict_velocity.__code__.co_filename,
               'starVLA/dataloader/navsim_dataset.py']
    identity = {'schema': 'original_deployment_condition_v1', 'boundary': CONDITION_BOUNDARY,
                'SFT_sha256': SFT_SHA256, 'population_identity': pop['identity_sha256'],
                'tokens_sha256': fingerprint(tokens), 'execution': profile,
                'processor': processor_identity(VLM), 'dependencies': dependency_versions(),
                'source_files': {str(Path(x).resolve().relative_to(Path.cwd())): file_sha(x) for x in sources},
                'input_list_sha256': file_sha(TRAIN_LIST), 'data_root': DATA_ROOT,
                'contents': ['scene_identity', 'observation_features'], 'no_cached_labels': True}
    store = ConditionStore(out, identity, readonly=False)
    if (out / progress_name).exists() and not a.resume:
        raise ValueError('Existing cache progress requires --resume')
    data_cfg = copy.deepcopy(agent.model_config)
    data_cfg.datasets.vla_data.data_root = DATA_ROOT
    data_cfg.datasets.vla_data.datalist_path = TRAIN_LIST
    data_cfg.enable_image_aug = 0; data_cfg.w_depth = 0
    dataset = KeyedDataset(data_cfg)
    priority = json.loads(Path(a.priority_tokens).read_text()) if a.priority_tokens else []
    if len(priority) != len(set(priority)) or not set(priority) <= set(tokens):
        raise ValueError('Priority tokens must be unique RL_TRAIN members')
    priority_set = set(priority)
    ordered = priority + [token for token in tokens if token not in priority_set]
    assigned = ordered[rank::world]
    if a.qualification_only:
        assigned = assigned[:4]
    started = time.monotonic()
    status = {'status': 'RUNNING', 'host': host, 'rank': rank, 'world': world, 'new': 0, 'reused': 0,
              'identity_sha256': store.identity_sha, 'GPU_elapsed_seconds': 0., 'qualification_only': a.qualification_only}
    check_raw, check_obs = [], []
    for token in assigned:
        existing = store.get(token)
        if existing is not None and len(check_raw) >= 2:
            status['reused'] += 1; continue
        begin = time.monotonic()
        obs = prepare_policy_observation([dataset[(token, seed_for('condition_data', 42, token))]])
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            hidden, positions = model.encode_policy_features(obs.examples(), feature_output='normalized')
            raw = extract_baseline_action_conditions(hidden, positions['action'])
        store.put(token, raw)
        if len(check_raw) < 2:
            check_raw.append(raw.detach()); check_obs.append(obs)
        torch.cuda.synchronize()
        status['new'] += int(existing is None); status['reused'] += int(existing is not None)
        status['GPU_elapsed_seconds'] += time.monotonic() - begin
        status['wall_seconds'] = time.monotonic() - started
        if (status['new'] + status['reused']) % 16 == 0:
            atomic_json(out / progress_name, status)
        if len(check_raw) == 2 and not (out / qualification_name).exists():
            checks = []
            for b in (1, 2):
                raw = torch.cat(check_raw[:b])
                cached = torch.cat([store.get(assigned[i]).cuda() for i in range(b)])
                if not torch.equal(raw, cached):
                    raise ValueError('Raw original observation and saved condition differ')
                for g in (1, 16):
                    noise = torch.randn((b*g, 8, 4), device='cuda', dtype=torch.bfloat16,
                                        generator=torch.Generator(device='cuda').manual_seed(4242))
                    features = raw.repeat_interleave(g, 0)
                    stored = cached.repeat_interleave(g, 0)
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                        original = sample_ode(model, encode_condition(model, features), noise)
                        restored = sample_ode(model, encode_condition(model, stored), noise)
                    difference = float((original-restored).abs().max())
                    if difference != 0:
                        raise ValueError('Same-layout cached ODE changed original ODE')
                    checks.append({'B': b, 'candidate_chunk': g, 'feature_max_abs': 0., 'ODE_max_abs': difference})
            root = torch.randn((1, 8, 4), device='cuda', dtype=torch.bfloat16,
                               generator=torch.Generator(device='cuda').manual_seed(2101))
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                direct = model.predict_action_infer_1d(check_obs[0].examples(), initial_noise=root)['normalized_actions']
                restored = sample_ode(model, encode_condition(model, check_raw[0]), root).float().cpu().numpy()
            if not np.array_equal(direct, restored):
                raise ValueError('Original predict_action_infer_1d and cache path disagree')
            atomic_json(out / qualification_name, {'status': 'PASS', 'identity_sha256': store.identity_sha,
                        'checks': checks, 'direct_original_predict_exact': True,
                        'scope': 'canonical batch-one prefix; two scene and two action chunk layouts; same-layout equality'})
    status.update(status='ASSIGNED_COMPLETE', wall_seconds=time.monotonic()-started,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    atomic_json(out / progress_name, status)
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
