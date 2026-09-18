"""Initial-policy step credit on saved real banks, without sampling new actions.

Only the Jacobian at the projected condition is measured. Parameters are frozen
for this observer to avoid allocating full model gradients; this is explicitly
NOT evidence of RL gradients reaching Qwen. The separate native RL-only runs
provide that evidence. Every saved candidate from the predeclared scenes is used.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from starVLA.rl.flow_grpo.config import resolve_config
from starVLA.rl.flow_grpo.loading import load_policy, file_sha
from starVLA.rl.flow_grpo.data import KeyedDataset
from starVLA.rl.flow_grpo.observation import prepare_policy_observation
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.rollout import velocity
from starVLA.rl.flow_grpo.math import transition, reduce_dimensions, group_advantages
from starVLA.rl.flow_grpo.credit import transition_credit_weights
from starVLA.rl.flow_grpo.audit import capture_source_environment


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--bank', required=True, nargs='+')
    parser.add_argument('--output', required=True)
    a = parser.parse_args()
    cfg, sft = resolve_config(a.config)
    configure_numerics()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    capture_source_environment(out, cfg)
    manifest = json.loads(Path(a.manifest).read_text())
    records, locations, bank_digests = {}, {}, {}
    for folder in a.bank:
        bank = Path(folder)
        data = json.loads((bank/'results.json').read_text())
        complete = json.loads((bank/'COMPLETE').read_text())
        if file_sha(bank/'results.json') != complete['results_sha256']:
            raise ValueError('saved bank changed')
        if data['identity']['checkpoint_sha256'] != cfg['checkpoint_contract']['sha256']:
            raise ValueError('bank initialized from a different checkpoint')
        bank_digests[folder] = complete['results_sha256']
        for row in data['scenes']:
            if row['token'] in records:
                raise ValueError('duplicate scene in saved bank')
            records[row['token']] = row
            locations[row['token']] = bank/(row['token']+'.npz')
    selected = manifest['tokens']
    if not set(selected) <= records.keys():
        raise ValueError('predeclared probe scenes missing')
    policy = load_policy(cfg, sft).cuda().bfloat16().eval()
    policy.requires_grad_(False)
    policy._inference_qwen_forward_mode = 'optimized'
    dataset = KeyedDataset(sft)
    result = {'scope': 'condition Jacobian only; no training, no all-parameter-gradient claim',
              'checkpoint_sha256': cfg['checkpoint_contract']['sha256'],
              'manifest_sha256': file_sha(a.manifest), 'bank_results_sha256': bank_digests,
              'script_sha256': file_sha(__file__), 'rows': []}
    started = time.monotonic()
    try:
        for token in selected:
            obs = prepare_policy_observation([dataset[(token, 42)]])
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                condition = policy.encode_policy_condition(obs)
            if condition.dtype != torch.float32:
                raise ValueError('expected actual FP32 projected condition')
            condition = condition.detach().requires_grad_(True)
            gradients = {}
            with np.load(locations[token], allow_pickle=False) as saved:
                for setting in manifest['settings']:
                    record = records[token]['settings'][setting]
                    chain = torch.from_numpy(saved[setting+'_chain']).cuda()
                    rewards = torch.tensor([[r['score'] for r in record['scores']]], device='cuda')
                    adv = group_advantages(rewards, epsilon=1e-6, clip=5.)[0]
                    g, kp, h, d = chain.shape
                    k = kp-1
                    times = torch.arange(k+1, device='cuda', dtype=torch.float32)/k
                    per_step = []
                    for step in range(k):
                        grad = torch.zeros_like(condition)
                        for candidate in range(g):
                            xt = chain[candidate:candidate+1, step]
                            xn = chain[candidate:candidate+1, step+1]
                            bucket = torch.full((1,), int(step/k*policy.action_model.num_timestep_buckets),
                                                device='cuda', dtype=torch.long)
                            v = velocity(policy, xt, bucket, condition)
                            dist = transition(xt, v, times[step], times[step+1]-times[step],
                                noise_level=.1, first_dt=times[1], temporal_correlation=.8,
                                transition_mode=record['transition_mode'])
                            # At the behavior snapshot this score gradient equals
                            # the PPO surrogate gradient, where ratio is exactly1.
                            loss = -adv[candidate]*reduce_dimensions(dist.logprob(xn)).mean()/g
                            grad += torch.autograd.grad(loss, condition)[0]
                        per_step.append(grad.detach().flatten())
                    vectors = torch.stack(per_step).double()
                    weights = transition_credit_weights(torch.ones(1, g, k, device='cuda', dtype=torch.bool),
                                                         .6, dtype=torch.float64, normalization='scene_mean_one')[0, 0]
                    uniform, discounted = vectors.mean(0), (vectors*weights[:, None]).mean(0)
                    norms = vectors.norm(dim=-1)
                    denom = norms[:, None]*norms[None, :]
                    cosine = torch.where(denom>0, vectors@vectors.T/denom.clamp_min(1e-300), 0)
                    norm_product = float(uniform.norm()*discounted.norm())
                    row = {'token': token, 'setting': setting,
                        'reward_std': float(rewards.double().std(unbiased=False)),
                        'nonzero_advantage': bool(adv.abs().max()>0),
                        'condition_dtype': str(condition.dtype), 'condition_gradient_accumulation_dtype': str(grad.dtype),
                        'per_step_l2': norms.tolist(), 'step_cosine': cosine.tolist(),
                        'discount_weights': weights.tolist(),
                        'uniform_l2': float(uniform.norm()), 'discounted_l2': float(discounted.norm()),
                        'uniform_discounted_cosine': float(uniform@discounted)/norm_product if norm_product else None}
                    result['rows'].append(row)
                    gradients[setting] = vectors.cpu().numpy()
                    print(token, setting, 'step norms', row['per_step_l2'], flush=True)
            np.savez(out/(token+'.npz'), **gradients)
            (out/'probe.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        result.update(status='COMPLETE', seconds=time.monotonic()-started)
        (out/'probe.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        (out/'COMPLETE').write_text(json.dumps({'sha256':file_sha(out/'probe.json')}))
    except BaseException as exc:
        (out/'failure.json').write_text(json.dumps({'status':'FAIL','error':repr(exc)}))
        raise


if __name__ == '__main__':
    main()
