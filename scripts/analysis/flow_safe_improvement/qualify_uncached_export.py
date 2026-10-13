"""Run the original exported actor on raw observations with research assets denied.

This is a deployment interface check, not a planning score. No feature store,
teacher, candidate bank, search or reward service is constructed.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from omegaconf import OmegaConf
import torch

from starVLA.rl.flow_grpo.data import KeyedDataset
from starVLA.rl.flow_grpo.observation import prepare_policy_observation
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import file_sha, fingerprint, seed_for
from starVLA.rl.safe_flow_improvement.iteration import require_dispatch


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('export', 'export-sha256', 'population', 'cohorts', 'output'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    require_dispatch()
    configure_numerics()
    export, out = Path(a.export), Path(a.output)
    if not (export/'COMPLETE').exists() or file_sha(export/'pytorch_model.pt') != a.export_sha256:
        raise ValueError('Require a sealed full original-format export with its exact hash')
    population = json.loads(Path(a.population).read_text())
    cohorts = json.loads(Path(a.cohorts).read_text())
    if cohorts['population_identity'] != population['identity_sha256']:
        raise ValueError('Population identity changed')
    tokens = cohorts['diagnostic_tokens'][:2]
    if not set(tokens) <= set(population['train_tokens']):
        raise ValueError('Qualification requires the fixed TRAIN diagnostic')
    out.mkdir(parents=True, exist_ok=False)
    cfg = OmegaConf.load(export/'config.yaml')
    os.environ['BASE_VLM'] = cfg.framework.qwenvl.base_vlm
    os.environ['VLM_ATTN_IMPLEMENTATION'] = cfg.framework.qwenvl.attn_implementation
    denied_roots = (
        '/mnt/project/DriveDreamer-Policy-action-rl/runs/action_head/features_F',
        '/mnt/project/DriveDreamer-Policy-flow-grpo/artifacts/action-only-checkpoints-v1',
        '/mnt/project/flow-safe-improvement-20261013/banks',
        '/mnt/project/flow-safe-improvement-20261013/targets',
        '/mnt/project/flow-safe-improvement-20261013/metric_cache_v1',
        '/mnt/project/flow-safe-improvement-20261013/reference_scores',
    )
    attempted = []

    def guard(event, arguments):
        if event != 'open' or not arguments or not isinstance(arguments[0], (str, bytes, os.PathLike)):
            return
        path = os.path.realpath(os.fsdecode(arguments[0]))
        if any(path == root or path.startswith(root+'/') for root in denied_roots):
            attempted.append(path)
            raise PermissionError('Research-only asset accessed by uncached deployment')

    sys.addaudithook(guard)
    from infer import VLAAgent, deal_action_1225
    started = time.monotonic()
    agent = VLAAgent(str(export), device='cuda', qwen_forward_mode='optimized')
    load_seconds = time.monotonic()-started
    data_cfg = copy.deepcopy(agent.model_config)
    data_cfg.enable_image_aug = 0
    data_cfg.w_depth = 0
    dataset = KeyedDataset(data_cfg)
    if agent.model.action_model.num_inference_timesteps != 10:
        raise ValueError('Original deployment ODE changed')
    actions, requests, times, input_keys = [], [], [], []
    seed = population['evaluation_seeds'][0]
    for token in tokens:
        root_key = seed_for('original_ode_initial', seed, token)
        sample = dataset[(token, root_key)]
        batch = prepare_policy_observation([sample]).examples()
        input_keys.append(sorted(batch[0]))
        del sample
        dtype = next(agent.model.qwen_vl_interface.parameters()).dtype
        root = torch.randn((1, 8, 4), device='cuda', dtype=dtype,
                           generator=torch.Generator(device='cuda').manual_seed(root_key))
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            action = agent.model.predict_action_infer_1d(batch, initial_noise=root)['normalized_actions']
        torch.cuda.synchronize()
        times.append(time.monotonic()-started)
        request = deal_action_1225(action, act_norm=int(data_cfg.datasets.vla_data.act_norm))
        if action.shape != (1, 8, 4) or request.shape != (1, 8, 3) or not np.isfinite(request).all():
            raise ValueError('Invalid original exported actor output')
        actions.append(action[0]); requests.append(request[0])
    np.savez_compressed(out/'outputs.npz', tokens=np.asarray(tokens), inference_seed=seed,
                        normalized_actions=np.stack(actions), requested_trajectories=np.stack(requests))
    report = dict(status='PASS_UNCACHED_ORIGINAL_DEPLOYMENT', behavior_sha256=a.export_sha256,
                  source_sha256=file_sha(__file__), population_identity=population['identity_sha256'],
                  cohort_identity=cohorts['sha256'], scenes=len(tokens), inference_seed=seed,
                  actor_input_keys=input_keys, research_asset_access_attempts=len(attempted),
                  forbidden_asset_roots=list(denied_roots), no_frozen_feature_installation=True,
                  no_teacher_search_bank_or_scorer_in_actor=True, original_VLAAgent=True,
                  original_ODE_steps=10, root_dtype=str(dtype),
                  action_parameter_dtype=str(next(agent.model.action_model.parameters()).dtype),
                  model_load_seconds=load_seconds, per_scene_full_inference_seconds=times,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                  latency_scope='two raw-observation scenes, shared GPU; not a stable latency benchmark')
    report['sha256'] = fingerprint(report)
    atomic_json(out/'report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
