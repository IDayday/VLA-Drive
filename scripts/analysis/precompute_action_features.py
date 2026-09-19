"""Precompute frozen F encoder features, disjoint token shards, no policy updates."""
import argparse
import json
import os
from pathlib import Path
import time
import torch
from starVLA.rl.flow_grpo.config import resolve_config, split_tokens
from starVLA.rl.flow_grpo.loading import load_policy
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.action_head_policy import (
    feature_identity, FrozenFeatureStore, install_frozen_features, freeze_for_action_head,
)
from starVLA.rl.flow_grpo.data import KeyedDataset
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--shard', type=int, default=None)
    p.add_argument('--shards', type=int, default=None)
    p.add_argument('--finalize', action='store_true')
    a = p.parse_args()
    cfg, sft = resolve_config(a.config)
    if cfg['trainable_policy'] != 'action_head': raise ValueError('only frozen-prefix action-head RL')
    configure_numerics()
    store = FrozenFeatureStore(cfg['frozen_feature_cache']['root'], feature_identity(cfg, sft))
    tokens, _ = split_tokens(cfg)
    if a.finalize:
        print(json.dumps(store.finalize(tokens))); return
    rank = a.shard if a.shard is not None else int(os.getenv('RANK', 0))
    world = a.shards if a.shards is not None else int(os.getenv('WORLD_SIZE', 1))
    if not 0 <= rank < world: raise ValueError('invalid shard')
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
    policy = load_policy(cfg, sft).cuda().bfloat16().eval()
    freeze_for_action_head(policy)
    install_frozen_features(policy, store)
    dataset = KeyedDataset(sft)
    todo = tokens[rank::world]
    start = time.monotonic()
    progress = store.root / f'precompute_world{world}_rank{rank}.json'
    for index, token in enumerate(todo):
        entry = store.get(token)
        if entry is None:
            sample = dataset[(token, cfg['runtime']['seed'])]
            policy.encode_policy_features([sample])
        elif 'action' not in entry:
            raise ValueError('incomplete replay entry; preserve and repair separately: ' + token)
        if index % 100 == 0 or index + 1 == len(todo):
            atomic_json(progress, {'status': 'COMPLETE' if index + 1 == len(todo) else 'RUNNING',
                       'rank': rank, 'world_size': world, 'count': index + 1, 'total': len(todo),
                       'elapsed_seconds': time.monotonic() - start, 'identity': store.identity_sha})
    print(progress.read_text(), flush=True)


if __name__ == '__main__': main()
