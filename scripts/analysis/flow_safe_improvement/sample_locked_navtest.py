"""Locked final Navtest, one original ODE output per scene and no search assets.

Independent local torchrun groups use disjoint scene shards. Rank placement is
never part of the root-noise key. A final model-selection lock is mandatory.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from scripts.analysis.flow_safe_improvement.build_candidate_banks import publish_scene
from scripts.analysis.flow_safe_improvement.run_original_evaluation_queue import authorized
from scripts.analysis.flow_safe_improvement.sample_original_export import PROFILE
from starVLA.rl.flow_grpo.data import KeyedDataset
from starVLA.rl.flow_grpo.observation import prepare_policy_observation
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import SFT_SHA256, file_sha, fingerprint, seed_for
from starVLA.rl.safe_flow_improvement.iteration import require_dispatch
from starVLA.rl.safe_flow_improvement.policy_adapter import sample_ode, decode_request


def locked_contract(path, model_name, population):
    lock = json.loads(Path(path).read_text()); digest = lock.pop('sha256')
    if fingerprint(lock) != digest or lock['status'] != 'LOCKED_BEFORE_FINAL_NAVTEST':
        raise ValueError('Final selection must be locked before Navtest access')
    if (lock['population_identity'] != population['identity_sha256'] or lock['inference_seeds'] != [2101]
            or lock['training_seeds'] != [42] or lock['navtest_tokens_sha256'] != fingerprint(population['navtest_tokens'])
            or len(set(population['navtest_tokens'])) != 12146):
        raise ValueError('Final model/population/seed contract changed')
    selected = lock['models'][model_name]
    if selected['sampler_source_sha256'] != file_sha(__file__):
        raise ValueError('Navtest sampling source changed after lock')
    return lock, digest, selected


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('final-lock', 'model', 'population', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--shard-offset', type=int, required=True)
    p.add_argument('--shard-count', type=int, required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    pop = json.loads(Path(a.population).read_text())
    lock, lock_sha, selected = locked_contract(a.final_lock, a.model, pop)
    host = require_dispatch(); authorized(host); configure_numerics()
    local_rank = int(os.environ.get('LOCAL_RANK', 0)); rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1)); shard = a.shard_offset+rank
    if not 0 <= a.shard_offset <= shard < a.shard_count or a.shard_offset+world > a.shard_count:
        raise ValueError('Invalid independent shard assignment')
    torch.cuda.set_device(local_rank)
    export = Path(selected['export'])
    if file_sha(export/'pytorch_model.pt') != selected['weight_sha256'] or file_sha(export/'config.yaml') != selected['config_sha256']:
        raise ValueError('Selected export changed after lock')
    if selected['weight_sha256'] != SFT_SHA256 and not (export/'COMPLETE').exists():
        raise ValueError('Unsealed learned export')
    cfg = OmegaConf.load(export/'config.yaml')
    os.environ['BASE_VLM'] = '/mnt/project/DriveDreamer-Policy/models/Qwen3-VL-2B-WorldAction'
    os.environ['VLM_ATTN_IMPLEMENTATION'] = cfg.framework.qwenvl.attn_implementation
    from infer import VLAAgent
    import infer
    agent = VLAAgent(str(export), device='cuda', qwen_forward_mode='optimized')
    profile = dict(name=PROFILE, qwen_parameter_dtype=str(next(agent.model.qwen_vl_interface.parameters()).dtype),
        action_parameter_dtype=str(next(agent.model.action_model.parameters()).dtype),
        history_parameter_dtype=str(next(agent.model.action_input_model.parameters()).dtype),
        attention_backend=cfg.framework.qwenvl.attn_implementation,
        actor_sources={str(Path(x).relative_to(Path.cwd())): file_sha(x) for x in
            (infer.__file__, agent.model.encode_policy_condition.__code__.co_filename,
             agent.model.action_model._euler_sample.__code__.co_filename)})
    if profile != lock['deployment_execution_profile'] or agent.model.action_model.num_inference_timesteps != 10:
        raise ValueError('Final actor numerics differ from the matched DEV contract')
    tokens = pop['navtest_tokens']; out = Path(a.output)
    identity = dict(schema='flow_safe_frozen_samples_v1', population_identity=pop['identity_sha256'],
        tokens_sha256=fingerprint(tokens), scenes=len(tokens), cohort='navtest_tokens', SFT_sha256=SFT_SHA256,
        behavior_sha256=selected['weight_sha256'], feature_identity='NO_FEATURE_CACHE_RAW_OBSERVATIONS',
        seeds=[2101], mode='ode', role='evaluation', root_precision='original_bfloat16', execution_profile=profile,
        final_selection_lock_sha256=lock_sha, source_files={Path(__file__).name: file_sha(__file__)})
    identity['sha256'] = fingerprint(identity); out.mkdir(parents=True, exist_ok=True)
    with (out/'manifest.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        if (out/'manifest.json').exists():
            if json.loads((out/'manifest.json').read_text()) != identity:
                raise ValueError('Final inference identity changed')
            if (out/f'shard{shard}.json').exists() and not a.resume:
                raise ValueError('Explicit shard recovery required')
        else:
            atomic_json(out/'manifest.json', identity)
    worker_guard = (out/f'shard{shard}.lock').open('a')
    fcntl.flock(worker_guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    data_cfg = copy.deepcopy(agent.model_config)
    data_cfg.datasets.vla_data.data_root = lock['data_root']
    data_cfg.datasets.vla_data.datalist_path = lock['navtest_datalist']
    data_cfg.enable_image_aug = 0; data_cfg.w_depth = 0
    dataset = KeyedDataset(data_cfg, split='test')
    if set(dataset.indices) != set(tokens):
        raise ValueError('Native current-input population differs from the final Navtest lock')
    status = dict(status='RUNNING', host=host, pid=os.getpid(), shard=shard, shard_count=a.shard_count,
        identity_sha256=identity['sha256'], new_scenes=0, reused_scenes=0, velocity_forward_calls=0,
        original_direct_predict_checks=0, GPU_generation_seconds=0., wall_seconds=0.)
    started = time.monotonic()
    for token in tokens[shard::a.shard_count]:
        path = out/'scenes'/token[:2]/(token+'.pt')
        if path.with_suffix('.json').exists():
            seal = json.loads(path.with_suffix('.json').read_text())
            if seal['identity_sha256'] != identity['sha256'] or seal['payload_sha256'] != file_sha(path):
                raise ValueError('Previously completed prediction changed')
            status['reused_scenes'] += 1; continue
        t0 = time.monotonic(); key = seed_for('original_ode_initial', 2101, token)
        observation = prepare_policy_observation([dataset[(token, key)]])
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            condition, dtype = agent.model.encode_policy_condition(observation, return_raw_dtype=True, feature_output='normalized')
            noise = torch.randn((1, 8, 4), device='cuda', dtype=dtype, generator=torch.Generator(device='cuda').manual_seed(key))
            actions = sample_ode(agent.model, condition, noise).float().cpu()
            if status['new_scenes'] == 0:
                direct = agent.model.predict_action_infer_1d(observation.examples(), initial_noise=noise)['normalized_actions']
                if not np.array_equal(direct, actions.numpy()):
                    raise ValueError('Final wrapper differs from the original direct inference')
                status['original_direct_predict_checks'] += 1
        requests = torch.from_numpy(decode_request(actions.numpy(), act_norm=int(data_cfg.datasets.vla_data.act_norm)))
        if not torch.isfinite(actions).all() or not torch.isfinite(requests).all():
            raise ValueError('Invalid prediction retained in denominator; stop affected shard')
        torch.cuda.synchronize(); elapsed = time.monotonic()-t0
        value = dict(token=token, log_name=pop['token_logs'][token], behavior_sha256=selected['weight_sha256'],
            identity_sha256=identity['sha256'], normalized_actions=actions, requested_trajectories=requests,
            initial_noise=noise.cpu(), noise_keys=[dict(seed=2101, root_key=key, root_namespace='original_ode_initial')],
            initial_noise_dtype=str(dtype), sampler='ode', role='evaluation', generation_seconds=elapsed, execution_profile=PROFILE)
        publish_scene(path, value, identity['sha256'])
        status['new_scenes'] += 1; status['velocity_forward_calls'] += 10
        status['GPU_generation_seconds'] += elapsed; status['wall_seconds'] = time.monotonic()-started
        if status['new_scenes'] % 8 == 0:
            atomic_json(out/f'shard{shard}.json', status)
    status.update(status='ASSIGNED_POPULATION_COMPLETE', wall_seconds=time.monotonic()-started,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                  extra_qualification_velocity_calls=10*status['original_direct_predict_checks'])
    atomic_json(out/f'shard{shard}.json', status)
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
