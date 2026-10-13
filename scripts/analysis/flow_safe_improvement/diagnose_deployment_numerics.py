"""Isolate original-export versus frozen-training feature/precision differences."""
import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from starVLA.rl.flow_grpo.action_head_policy import FrozenFeatureStore, feature_identity
from starVLA.rl.flow_grpo.config import resolve_config
from starVLA.rl.flow_grpo.data import KeyedDataset
from starVLA.rl.flow_grpo.observation import prepare_policy_observation
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import file_sha, fingerprint, seed_for
from starVLA.rl.safe_flow_improvement.iteration import require_dispatch
from starVLA.rl.safe_flow_improvement.policy_adapter import encode_condition, sample_ode, decode_request


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('export', 'export-sha256', 'population', 'cohorts', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--config', default='configs/flow_grpo/frozen_action_head_batched_single8_epoch1.yaml')
    p.add_argument('--source-config', default='/mnt/project/DriveDreamer-Policy-flow-grpo/artifacts/action-only-source/checkpoint_code/action-only-checkpoints-v1/configs/frozen-visual-step100000-config.yaml')
    a = p.parse_args()
    require_dispatch(); configure_numerics()
    export, out = Path(a.export), Path(a.output)
    if file_sha(export/'pytorch_model.pt') != a.export_sha256:
        raise ValueError('Export changed')
    out.mkdir(parents=True, exist_ok=False)
    pop, cohorts = [json.loads(Path(x).read_text()) for x in (a.population, a.cohorts)]
    tokens = cohorts['diagnostic_tokens'][:2]
    cfg, sft = resolve_config(a.config, overrides=['checkpoint_contract.source_config='+a.source_config])
    store = FrozenFeatureStore(cfg['frozen_feature_cache']['root'], feature_identity(cfg, sft), readonly=True)
    exported_cfg = OmegaConf.load(export/'config.yaml')
    os.environ['BASE_VLM'] = exported_cfg.framework.qwenvl.base_vlm
    os.environ['VLM_ATTN_IMPLEMENTATION'] = exported_cfg.framework.qwenvl.attn_implementation
    from infer import VLAAgent
    from starVLA.model.framework.baseline_qwen import extract_baseline_action_conditions
    agent = VLAAgent(str(export), device='cuda', qwen_forward_mode='optimized')
    data_cfg = copy.deepcopy(agent.model_config)
    data_cfg.enable_image_aug = 0; data_cfg.w_depth = 0
    dataset = KeyedDataset(data_cfg)
    batches = [prepare_policy_observation([dataset[(t, seed_for('original_ode_initial', 2101, t))]]).examples() for t in tokens]
    raw_cache = [store.get(t)['raw'].cuda() for t in tokens]
    roots = [torch.randn((1, 8, 4), device='cuda', dtype=torch.bfloat16,
                         generator=torch.Generator(device='cuda').manual_seed(seed_for('original_ode_initial', 2101, t))) for t in tokens]
    raw_by_mode, actions = {}, {}
    for mode in ('original_VLAAgent_dtypes', 'all_BF16_training_dtypes'):
        if mode == 'all_BF16_training_dtypes':
            agent.model.bfloat16()
        raw_by_mode[mode], actions[mode] = [], []
        for batch, cached, root in zip(batches, raw_cache, roots):
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                hidden, positions = agent.model.encode_policy_features(batch, feature_output='normalized')
                raw = extract_baseline_action_conditions(hidden, positions['action'])
                uncached_action = sample_ode(agent.model, encode_condition(agent.model, raw), root)
                cached_action = sample_ode(agent.model, encode_condition(agent.model, cached), root)
            raw_by_mode[mode].append(raw.detach().float().cpu().numpy())
            actions[mode].append(dict(raw=uncached_action.float().cpu().numpy(), cached=cached_action.float().cpu().numpy()))
    rows = []
    for i in range(len(tokens)):
        raw = raw_cache[i].float().cpu().numpy()
        modes = {}
        for mode in actions:
            value = actions[mode][i]
            modes[mode] = dict(feature_max_abs=float(np.max(np.abs(raw_by_mode[mode][i]-raw))),
                               action_raw_vs_cached_max_abs=float(np.max(np.abs(value['raw']-value['cached']))))
        original = actions['original_VLAAgent_dtypes'][i]
        bf16 = actions['all_BF16_training_dtypes'][i]
        modes['cross_mode'] = dict(same_cached_feature_action_max_abs=float(np.max(np.abs(original['cached']-bf16['cached']))),
                                   raw_actor_action_max_abs=float(np.max(np.abs(original['raw']-bf16['raw']))),
                                   raw_actor_request_max_abs=float(np.max(np.abs(decode_request(original['raw'], act_norm=1)-decode_request(bf16['raw'], act_norm=1)))))
        rows.append(modes)
    report = dict(status='COMPLETE_NUMERICAL_ISOLATION_NOT_PLANNING_SCORE', behavior_sha256=a.export_sha256,
                  population_identity=pop['identity_sha256'], scenes=len(tokens), source_sha256=file_sha(__file__),
                  feature_identity=store.identity_sha, comparisons=rows, no_source_or_checkpoint_mutations=True)
    report['sha256'] = fingerprint(report)
    atomic_json(out/'report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
