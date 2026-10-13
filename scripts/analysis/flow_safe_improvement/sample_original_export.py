"""Fresh single-candidate ODE evaluation through the uncached original VLAAgent.

Preserves the original mixed parameter dtypes, including FP32 history/action
modules. The frozen BF16 training feature cache is deliberately not installed.
Five seeds reuse only the current raw observation encoding within a scene.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
from omegaconf import OmegaConf
import torch

from scripts.analysis.flow_safe_improvement.build_candidate_banks import publish_scene
from starVLA.rl.flow_grpo.data import KeyedDataset
from starVLA.rl.flow_grpo.observation import prepare_policy_observation
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import SFT_SHA256, file_sha, fingerprint, seed_for
from starVLA.rl.safe_flow_improvement.iteration import require_dispatch
from starVLA.rl.safe_flow_improvement.policy_adapter import sample_ode, decode_request
from starVLA.rl.safe_flow_improvement.evaluation_override import evaluation_seeds


PROFILE = 'original_VLAAgent_default_module_dtypes_uncached_v1'


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('export', 'export-sha256', 'population', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--cohort', choices=['dev_tokens', 'diagnostic_tokens'], default='dev_tokens')
    p.add_argument('--cohorts')
    p.add_argument('--role', choices=['reference', 'evaluation'], default='evaluation')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--evaluation-override')
    p.add_argument('--reuse-samples', help='Read-only sealed original-export samples; only matching seed outputs may be reused')
    a = p.parse_args()
    host = require_dispatch(); configure_numerics()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    export, out = Path(a.export), Path(a.output)
    if file_sha(export/'pytorch_model.pt') != a.export_sha256:
        raise ValueError('Export hash changed')
    if a.export_sha256 != SFT_SHA256 and not (export/'COMPLETE').exists():
        raise ValueError('Updated model export is not sealed')
    pop = json.loads(Path(a.population).read_text())
    cohorts = json.loads(Path(a.cohorts).read_text()) if a.cohorts else None
    tokens = pop['dev_tokens'] if a.cohort == 'dev_tokens' else cohorts['diagnostic_tokens']
    if a.cohort == 'diagnostic_tokens' and (cohorts['population_identity'] != pop['identity_sha256']
            or len(tokens) != 512 or not set(tokens) <= set(pop['train_tokens'])):
        raise ValueError('Fixed TRAIN diagnostic identity changed')
    if set(tokens) & set(pop['navtest_tokens']):
        raise ValueError('This entrypoint cannot access Navtest before final lock')
    seeds, override = evaluation_seeds(pop, a.evaluation_override) if a.role == 'evaluation' else (pop['reference_seeds'], None)
    if a.evaluation_override and a.role != 'evaluation':
        raise ValueError('User seed override does not redefine the fixed reference distribution')
    cfg = OmegaConf.load(export/'config.yaml')
    os.environ['BASE_VLM'] = '/mnt/project/DriveDreamer-Policy/models/Qwen3-VL-2B-WorldAction'
    os.environ['VLM_ATTN_IMPLEMENTATION'] = cfg.framework.qwenvl.attn_implementation
    from infer import VLAAgent
    import infer
    agent = VLAAgent(str(export), device='cuda', qwen_forward_mode='optimized')
    if agent.model.action_model.num_inference_timesteps != 10:
        raise ValueError('Original ODE step count changed')
    profile = dict(name=PROFILE, qwen_parameter_dtype=str(next(agent.model.qwen_vl_interface.parameters()).dtype),
                   action_parameter_dtype=str(next(agent.model.action_model.parameters()).dtype),
                   history_parameter_dtype=str(next(agent.model.action_input_model.parameters()).dtype),
                   attention_backend=cfg.framework.qwenvl.attn_implementation,
                   actor_sources={str(Path(x).relative_to(Path.cwd())): file_sha(x) for x in
                                  (infer.__file__, agent.model.encode_policy_condition.__code__.co_filename,
                                   agent.model.action_model._euler_sample.__code__.co_filename)})
    if profile['qwen_parameter_dtype'] != 'torch.bfloat16' or profile['action_parameter_dtype'] != 'torch.float32' or profile['history_parameter_dtype'] != 'torch.float32':
        raise ValueError('Original VLAAgent numerical module profile changed')
    identity = dict(schema='flow_safe_frozen_samples_v1', population_identity=pop['identity_sha256'],
                    tokens_sha256=fingerprint(tokens), scenes=len(tokens), cohort=a.cohort,
                    SFT_sha256=SFT_SHA256, behavior_sha256=a.export_sha256,
                    feature_identity='NO_FEATURE_CACHE_RAW_OBSERVATIONS', seeds=seeds, mode='ode', role=a.role,
                    root_precision='original_bfloat16', execution_profile=profile,
                    source_files={Path(__file__).name: file_sha(__file__)})
    if override:
        identity['evaluation_override'] = override
    reuse = None
    if a.reuse_samples:
        reuse = json.loads((Path(a.reuse_samples)/'manifest.json').read_text())
        for key in ('population_identity', 'tokens_sha256', 'behavior_sha256', 'execution_profile', 'mode', 'role', 'root_precision'):
            if reuse[key] != identity[key]:
                raise ValueError('Reuse changed original actor, population, sampler or precision: '+key)
        if not set(seeds) <= set(reuse['seeds']):
            raise ValueError('Requested seed is absent from sealed reusable outputs')
        identity['read_only_sample_reuse'] = dict(root=str(a.reuse_samples), identity_sha256=reuse['sha256'])
    if a.cohort == 'diagnostic_tokens':
        identity.update(explicit_tokens=tokens, cohort_identity=cohorts['sha256'], cohort_manifest_sha256=file_sha(a.cohorts))
    identity['sha256'] = fingerprint(identity)
    out.mkdir(parents=True, exist_ok=True)
    with (out/'manifest.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (out/'manifest.json').exists():
            if json.loads((out/'manifest.json').read_text()) != identity:
                raise ValueError('Original deployment identity changed')
            if (out/f'rank{rank}.json').exists() and not a.resume:
                raise ValueError('Resume must be explicit')
        else:
            atomic_json(out/'manifest.json', identity)
    data_cfg = copy.deepcopy(agent.model_config)
    data_cfg.datasets.vla_data.data_root = '/mnt/project/DriveDreamer-Policy-paired/runs/paired_full_assets_v1/dataset'
    data_cfg.datasets.vla_data.datalist_path = '/mnt/project/DriveDreamer-Policy-paired/runs/paired_full_assets_v1/navtrain_tokens.json'
    data_cfg.enable_image_aug = 0; data_cfg.w_depth = 0
    dataset = KeyedDataset(data_cfg)
    status = dict(status='RUNNING', rank=rank, world_size=world, host=host, pid=os.getpid(), identity_sha256=identity['sha256'],
                  new_scenes=0, reused_scenes=0, new_candidates=0, velocity_forward_calls=0,
                  raw_observation_encoder_calls=0, original_direct_predict_checks=0, GPU_generation_seconds=0., wall_seconds=0.,
                  reused_from_previous_evaluation_scenes=0)
    started = time.monotonic()
    for token in tokens[rank::world]:
        path = out/'scenes'/token[:2]/(token+'.pt')
        if path.with_suffix('.json').exists():
            seal = json.loads(path.with_suffix('.json').read_text())
            if seal['identity_sha256'] != identity['sha256'] or seal['payload_sha256'] != file_sha(path):
                raise ValueError('Previously sealed samples changed')
            status['reused_scenes'] += 1
            continue
        t0 = time.monotonic()
        if reuse:
            source = Path(a.reuse_samples)/'scenes'/token[:2]/(token+'.pt')
            if source.with_suffix('.json').exists():
                seal = json.loads(source.with_suffix('.json').read_text())
                if seal['identity_sha256'] != reuse['sha256'] or seal['payload_sha256'] != file_sha(source):
                    raise ValueError('Sealed reusable output changed')
                value = torch.load(source, weights_only=True, map_location='cpu')
                indices = [reuse['seeds'].index(seed) for seed in seeds]
                for key in ('normalized_actions', 'requested_trajectories', 'initial_noise'):
                    value[key] = value[key][indices]
                value['noise_keys'] = [value['noise_keys'][i] for i in indices]
                value.update(identity_sha256=identity['sha256'], generation_seconds=0.,
                             reused_from=dict(payload_sha256=seal['payload_sha256'], manifest_sha256=reuse['sha256'],
                                              source=str(source), selected_seeds=seeds))
                publish_scene(path, value, identity['sha256'])
                status['reused_from_previous_evaluation_scenes'] += 1
                continue
        example = dataset[(token, seed_for('original_ode_initial', seeds[0], token))]
        observation = prepare_policy_observation([example]); del example
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            condition, raw_dtype = agent.model.encode_policy_condition(observation, return_raw_dtype=True, feature_output='normalized')
            roots, keys, actions = [], [], []
            for seed in seeds:
                key = seed_for('original_ode_initial', seed, token)
                root = torch.randn((1, 8, 4), device='cuda', dtype=raw_dtype,
                                   generator=torch.Generator(device='cuda').manual_seed(key))
                roots.append(root.cpu()); keys.append(dict(seed=seed, root_key=key, root_namespace='original_ode_initial'))
                actions.append(sample_ode(agent.model, condition, root).float().cpu())
                if status['new_scenes'] == 0 and seed == seeds[0]:
                    direct = agent.model.predict_action_infer_1d(observation.examples(), initial_noise=root)['normalized_actions']
                    if not np.array_equal(direct, actions[-1].numpy()):
                        raise ValueError('Within-scene encoding reuse changed original predict output')
                    status['original_direct_predict_checks'] += 1
        actions = torch.cat(actions)
        requests = torch.from_numpy(decode_request(actions.numpy(), act_norm=int(data_cfg.datasets.vla_data.act_norm)))
        if not torch.isfinite(actions).all() or not torch.isfinite(requests).all():
            raise ValueError('Nonfinite original actor output; scene remains in denominator')
        torch.cuda.synchronize(); elapsed = time.monotonic()-t0
        value = dict(token=token, log_name=pop['token_logs'][token], behavior_sha256=a.export_sha256,
                     identity_sha256=identity['sha256'], normalized_actions=actions, requested_trajectories=requests,
                     initial_noise=torch.cat(roots), noise_keys=keys, initial_noise_dtype=str(raw_dtype), sampler='ode',
                     role=a.role, generation_seconds=elapsed, execution_profile=PROFILE)
        publish_scene(path, value, identity['sha256'])
        status['new_scenes'] += 1; status['new_candidates'] += len(seeds)
        status['velocity_forward_calls'] += 10*len(seeds)
        status['raw_observation_encoder_calls'] += 1
        status['GPU_generation_seconds'] += elapsed; status['wall_seconds'] = time.monotonic()-started
        if status['new_scenes'] % 8 == 0:
            atomic_json(out/f'rank{rank}.json', status)
    status.update(status='ASSIGNED_POPULATION_COMPLETE', wall_seconds=time.monotonic()-started,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                  extra_qualification_velocity_calls=10*status['original_direct_predict_checks'])
    atomic_json(out/f'rank{rank}.json', status)
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
