"""Real frozen-Qwen/DiT integration, prediction isolation, and two charged updates."""
import argparse
import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from tools.structured_world.runtime import load_baseline, load_dataset, load_world_batch, seed_all
from tools.structured_world_v1p1.budget import start, record
from starVLA.model.modules.joint_world.policy import JointTrajectoryPolicy
from starVLA.model.modules.joint_world.flow import actor_mask, training_loss_sums
from starVLA.model.modules.joint_world.targets import joint_targets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['world-checkpoint', 'graph-config', 'targets', 'output', 'ledger', 'run-id']:
        p.add_argument('--' + key, required=True)
    a = p.parse_args(); out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    blob = Path(a.world_checkpoint).read_bytes()
    saved = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=False)
    args, cfg = saved['identity']['arguments'], saved['identity']['config']
    graph_cfg = json.loads(Path(a.graph_config).read_text())
    identity = {'code_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'world_checkpoint_sha256': hashlib.sha256(blob).hexdigest(), 'arguments': vars(a),
                'graph_config': graph_cfg, 'world_config': cfg}
    start(a.ledger, a.run_id, 2, identity); completed = 0
    try:
        seed_all(42); agent = load_baseline(args['checkpoint'], args['vlm']); agent.model.requires_grad_(False)
        policy = JointTrajectoryPolicy(agent.model, cfg, graph_cfg).cuda().eval()
        for name in ['reader', 'heads']:
            getattr(policy.world, name).load_state_dict(saved[name], strict=True)
        policy.world.requires_grad_(False)
        raw = load_dataset(agent, args['manifest'], args['data_root'], 1)[0]
        e = {k: raw[k] for k in ['image', 'lang', 'state', 'token']}
        seed_all(51); native = agent.model.predict_action_infer_1d([e])['normalized_actions']
        seed_all(51); initial = policy.predict_action([e])
        dirty = dict(e, action=np.full((8, 4), 1e9), WorldTargets={'boxes': torch.full((5, 8), float('nan'))}, future_file='/absent/future.png')
        seed_all(51); poisoned = policy.predict_action([dirty])
        checks = {'gate0_native_exact': bool(np.array_equal(native, initial['normalized_actions'])),
                  'poisoned_action_exact': bool(np.array_equal(initial['normalized_actions'], poisoned['normalized_actions'])),
                  'poisoned_graph_exact': torch.equal(initial['joint_trajectories_xy'], poisoned['joint_trajectories_xy'])}
        if not all(checks.values()):
            raise AssertionError(checks)
        # Features can be fixed here because every module before the graph is frozen.
        with torch.no_grad(): native_conditions, pred, current = policy.encode_current([e])
        _, targets = load_world_batch([e], a.targets)
        from infer import deal_action_1225
        physical = deal_action_1225(raw['action'][None], act_norm=agent.model_config.datasets.vla_data.act_norm)[..., :2]
        target_xy, valid, _ = joint_targets(pred, targets, torch.as_tensor(physical, device='cuda', dtype=torch.float32))
        actions = torch.as_tensor(raw['action'][None], device='cuda', dtype=torch.float32)
        params = [p for p in policy.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-4)
        gradients = []
        for step in range(2):
            if not record(a.ledger, a.run_id, completed): raise RuntimeError('GPU cap')
            opt.zero_grad(set_to_none=True)
            noise = torch.randn_like(target_xy)
            condition, joint, _ = policy.rollout_condition(native_conditions, current, noise)
            ego = agent.model.action_model(condition, actions, None)
            # First update isolates gate gradient from ego planning, with no auxiliary loss.
            graph_grad = torch.autograd.grad(ego, policy.graph.velocity[-1].weight, retain_graph=True, allow_unused=True)[0]
            gate_grad = torch.autograd.grad(ego, policy.adapter.gate, retain_graph=True)[0]
            hidden = actor_mask(1, target_xy.shape[1], 'cuda')
            sums, counts = training_loss_sums(policy.graph, target_xy, valid, hidden, noise, torch.rand(1, device='cuda'), **current)
            aux = sum(sums[k] / max(counts[k], 1) for k in sums)
            loss = ego if step == 0 else ego + aux
            before = policy.graph.velocity[-1].weight.detach().clone()
            loss.backward()
            preclip = float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True))
            opt.step(); completed += 1
            gradients.append({'step': completed, 'ego_loss': float(ego.detach()), 'joint_aux_loss': float(aux.detach()),
                              'gate_ego_gradient': float(gate_grad), 'graph_ego_gradient': None if graph_grad is None else float(graph_grad.norm()),
                              'gradient_before_clip': preclip, 'gate': float(policy.adapter.gate.detach()),
                              'graph_update_max': float((before - policy.graph.velocity[-1].weight).abs().max())})
            record(a.ledger, a.run_id, completed)
        assert gradients[0]['gate_ego_gradient'] != 0
        assert gradients[1]['graph_ego_gradient'] is not None and gradients[1]['graph_ego_gradient'] > 0
        assert gradients[1]['graph_update_max'] > 0
        modules = {'reader': policy.world.reader, 'heads': policy.world.heads, 'graph': policy.graph,
                   'graph_to_world': policy.graph_to_world, 'adapter': policy.adapter}
        delta = {k: module.state_dict() for k, module in modules.items()}
        torch.save({'identity': identity, 'delta': delta, 'optimizer': opt.state_dict(), 'step': completed}, out / 'checkpoint.pt')
        seed_all(51); before = policy.predict_action([e])
        restore = torch.load(out / 'checkpoint.pt', map_location='cpu', weights_only=False)
        restored = JointTrajectoryPolicy(agent.model, cfg, graph_cfg).cuda().eval()
        restored_modules = {'reader': restored.world.reader, 'heads': restored.world.heads,
                            'graph': restored.graph, 'graph_to_world': restored.graph_to_world, 'adapter': restored.adapter}
        for name, module in restored_modules.items(): module.load_state_dict(restore['delta'][name], strict=True)
        seed_all(51); after = restored.predict_action([e])
        checks['checkpoint_action_exact'] = bool(np.array_equal(before['normalized_actions'], after['normalized_actions']))
        checks['checkpoint_joint_exact'] = torch.equal(before['joint_trajectories_xy'], after['joint_trajectories_xy'])
        if not all(checks.values()): raise AssertionError(checks)
        result = dict(identity=identity, checks=checks, gradients=gradients, token=e['token'], optimizer_updates=completed,
                      target_points=int(valid.sum()), peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                      status='PASS', limitation='One real training scene and two update-path checks; not convergence or planning evidence.')
        (out / 'REAL_PIPELINE.json').write_text(json.dumps(result, indent=2)); print(json.dumps(result), flush=True)
        record(a.ledger, a.run_id, completed, 'complete')
    except BaseException:
        record(a.ledger, a.run_id, completed, 'failed'); raise


if __name__ == '__main__':
    main()
