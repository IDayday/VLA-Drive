"""Check a trained cached bridge against the actual current-image online policy."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from starVLA.model.modules.joint_world.policy import JointTrajectoryPolicy
from tools.joint_world.bev_cache import BEVFeatureStore
from tools.joint_world.extract_conditions import CACHE_FIELDS
from tools.joint_world.planner_runtime import CachedCurrentPlanner, file_sha256, predict
from tools.structured_world.runtime import load_baseline, load_dataset, load_world_batch, seed_all
from tools.structured_world_v1p1.budget import start, record


def restore(policy, state):
    names = ['graph', 'graph_to_world', 'adapter']
    if policy.bev_encoder is not None:
        names += ['bev_encoder', 'bev_fusion', 'interaction_head']
    consumed = set()
    for name in names:
        selected = {k[len(name)+1:]: v for k, v in state.items() if k.startswith(name+'.')}
        getattr(policy, name).load_state_dict(selected, strict=True)
        consumed.update(name+'.'+k for k in selected)
    if consumed != set(state):
        raise ValueError('Unaccounted learned checkpoint keys')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['world-checkpoint', 'bridge-checkpoint', 'cache', 'manifest', 'output', 'ledger', 'run-id']:
        p.add_argument('--'+key, required=True)
    p.add_argument('--bev-index'); p.add_argument('--bev-config'); p.add_argument('--observations')
    p.add_argument('--seed', type=int, default=20260926)
    a = p.parse_args(); out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    world = torch.load(a.world_checkpoint, map_location='cpu', weights_only=False)
    bridge = torch.load(a.bridge_checkpoint, map_location='cpu', weights_only=False)
    manifest = json.loads((Path(a.cache)/'manifest.json').read_text())
    if manifest['identity']['world_checkpoint_sha256'] != file_sha256(a.world_checkpoint):
        raise ValueError('Wrong upstream world checkpoint')
    bev_enabled = bridge['identity'].get('bev_enabled', False)
    if bev_enabled != bool(a.bev_index and a.bev_config and a.observations):
        raise ValueError('BEV checkpoint/config/observation mismatch')
    identity = {'arguments': vars(a), 'code_sha': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                'bridge_sha256': file_sha256(a.bridge_checkpoint), 'world_sha256': file_sha256(a.world_checkpoint),
                'normalized_action_atol': 1e-5, 'normalized_action_rtol': 1e-5, 'joint_atol_metres': 1e-4,
                'inputs': 'current three front images, navigation and original state', 'WorldTargets_loaded': False,
                'limitation': 'Dataset loader reads the existing metadata container; future/action fields are stripped before policy input.'}
    start(a.ledger, a.run_id, 0, identity)
    try:
        args = world['identity']['arguments']; seed_all(42)
        agent = load_baseline(args['checkpoint'], args['vlm']); agent.model.requires_grad_(False)
        cfg = json.loads(Path(a.bev_config).read_text()) if bev_enabled else None
        policy = JointTrajectoryPolicy(agent.model, world['identity']['config'], bridge['identity']['graph_config'], cfg).cuda().eval().requires_grad_(False)
        for name in ['reader', 'heads']:
            getattr(policy.world, name).load_state_dict(world[name], strict=True)
        restore(policy, bridge['model'])
        cached = CachedCurrentPlanner(agent.model_config.framework.qwenvl.vl_hidden_dim,
                                      bridge['identity']['graph_config'], bev_enabled).cuda().eval().requires_grad_(False)
        cached.load_state_dict(bridge['model'], strict=True)
        store = BEVFeatureStore(a.bev_index) if bev_enabled else None
        ds = load_dataset(agent, a.manifest, args['data_root'])
        expected = json.loads(Path(a.manifest).read_text())
        if not 1 <= len(expected) <= 32 or len(ds) != len(expected): raise ValueError('Bounded explicit scene manifest required')
        records = {r['token']: r for r in manifest['records']}; rows = []
        from infer import deal_action_1225
        for raw in ds:
            if not record(a.ledger, a.run_id, 0): raise RuntimeError('Budget reached')
            e = {k: raw[k] for k in ['image', 'lang', 'state', 'token']}; token = e['token']
            path = Path(a.cache)/(token+'.pt')
            if file_sha256(path) != records[token]['sha256']: raise ValueError('Changed current cache')
            c = torch.load(path, map_location='cpu', weights_only=True)
            if set(c) != CACHE_FIELDS or c['token'] != token: raise ValueError('Wrong current fields')
            sample = {'cache': c}
            inputs = None
            if bev_enabled:
                sample['bev'] = store[token]
                inputs, _ = load_world_batch([e], a.observations, load_targets=False)
            value = int.from_bytes(hashlib.sha256(f'{a.seed}:{token}'.encode()).digest()[:4], 'little')
            seed_all(value); torch.cuda.synchronize(); begin = time.perf_counter()
            online = policy.predict_action([e], inputs); torch.cuda.synchronize(); latency = time.perf_counter()-begin
            offline, joint = predict(agent.model.action_model, cached, sample, a.seed)
            dirty = dict(e, WorldTargets={'future': float('nan')}, action=np.full((8,4), 1e9), future_file='/absent/future.png')
            seed_all(value); poisoned = policy.predict_action([dirty], inputs)
            restore(policy, bridge['model']); seed_all(value); restored = policy.predict_action([e], inputs)
            disabled, _ = predict(agent.model.action_model, cached, sample, a.seed, disable_graph=True)
            decode = lambda x: deal_action_1225(x, act_norm=int(agent.model_config.datasets.vla_data.act_norm))[0]
            x = online['normalized_actions']; y = online['joint_trajectories_xy'].cpu().numpy()
            row = {'token': token, 'online_cached_action_exact': bool(np.array_equal(x, offline)),
                   'online_cached_action_max_delta': float(np.abs(x-offline).max()),
                   'online_cached_joint_max_delta_m': float(np.abs(y-joint).max()),
                   'online_cached_action_within_tolerance': bool(np.allclose(x, offline, atol=1e-5, rtol=1e-5)),
                   'online_cached_joint_within_tolerance': bool(np.allclose(y, joint, atol=1e-4, rtol=1e-5)),
                   'target_poison_action_exact': bool(np.array_equal(x, poisoned['normalized_actions'])),
                   'target_poison_joint_exact': torch.equal(online['joint_trajectories_xy'], poisoned['joint_trajectories_xy']),
                   'strict_restore_action_exact': bool(np.array_equal(x, restored['normalized_actions'])),
                   'strict_restore_joint_exact': torch.equal(online['joint_trajectories_xy'], restored['joint_trajectories_xy']),
                   'disabled_world_ego_xy_max_delta_m': float(np.abs(decode(x)[:,:2]-decode(disabled)[:,:2]).max()),
                   'learned_action_gate': float(policy.adapter.gate), 'full_online_seconds': latency}
            if bev_enabled:
                hook = policy.bev_encoder.register_forward_pre_hook(lambda m, values: (torch.zeros_like(values[0]), *values[1:]))
                try:
                    seed_all(value); blank = policy.predict_action([e], inputs)
                finally: hook.remove()
                row.update(learned_bev_gate=float(policy.bev_fusion.gate),
                    blank_bev_joint_max_delta_m=float((online['joint_trajectories_xy']-blank['joint_trajectories_xy']).abs().max()),
                    blank_bev_ego_xy_max_delta_m=float(np.abs(decode(x)[:,:2]-decode(blank['normalized_actions'])[:,:2]).max()))
            rows.append(row); print(json.dumps(row), flush=True)
        passed = all(all(v for k,v in r.items() if k.endswith('within_tolerance') or k.startswith(('target_poison_', 'strict_restore_'))) for r in rows)
        report = {'identity': identity, 'scenes': rows, 'status': 'PASS' if passed else 'FAIL',
                  'peak_gpu_bytes': torch.cuda.max_memory_allocated(), 'optimizer_updates': 0,
                  'diagnostic_limit': 'Disabling world/blanking BEV is potentially out of distribution; sensitivity is not proof of planning benefit.'}
        (out/'LEARNED_POLICY_CHECK.json').write_text(json.dumps(report, indent=2))
        if not passed: raise AssertionError('Online/cached or target isolation/restore checks failed')
        record(a.ledger, a.run_id, 0, 'complete')
    except BaseException:
        record(a.ledger, a.run_id, 0, 'failed'); raise


if __name__ == '__main__': main()
