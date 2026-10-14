"""Actual failed C_SDE transition: values, original guard and full-head gradients."""
import argparse
import json
from pathlib import Path
import time

import torch

from starVLA.rl.constrained_flow_noise.contracts import file_sha, require_dispatch
from starVLA.rl.constrained_flow_noise.loading import make_noise_policy
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json
from .stable_saved_probability import StableSavedProbability, replace_transitions


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--failure', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    host = require_dispatch(); configure_numerics(); torch.cuda.set_device(0)
    cp, output = Path(args.checkpoint), Path(args.output)
    if output.exists(): raise FileExistsError(output)
    identity = json.loads((cp/'identity.json').read_text())
    if identity['variant'] != 'C_SDE' or not (cp/'COMPLETE').exists():
        raise ValueError('Expected complete recovered C checkpoint')
    actor, _, _ = make_noise_policy('C_SDE', (identity['kernel']['min_std'], identity['kernel']['max_std']))
    model = cp/'pytorch_model/mp_rank_00_model_states.pt'
    seals = json.loads((cp/'files.json').read_text())
    if file_sha(model) != seals[str(model.relative_to(cp))]: raise ValueError('Changed checkpoint')
    actor.load_state_dict(torch.load(model, map_location='cpu', mmap=True, weights_only=False)['module'], strict=True)
    actor.requires_grad_(True).eval()
    failed = torch.load(args.failure, map_location='cpu', weights_only=False)['state']
    collector = failed['collector']; slot = 3
    row = collector['rows'][collector['accepted'][slot]]
    pg = {k:row['chain'][k].cuda() for k in ('raw','chain','old_elementwise_logp')}
    pg['scene_key'] = row['scene_key']
    controller = StableSavedProbability(); controller.bind(collector['behavior_identity'], slot)
    start = time.monotonic()
    current, entry = controller.statistics(actor, actor.saved_statistics, pg)
    indices = entry['indices']
    if not indices: raise ValueError('Recorded numerical fault was not reproduced')
    mask = torch.tensor(indices, device='cuda')
    # A selected transition's gradient must be the real B1 kernel gradient,
    # including all qwen_proj/DiT parameters; the unused flat graph adds zero.
    selected = current['elementwise_logp'].reshape(-1,8,4)[mask]
    value = selected.detach().cpu()
    selected.sum().backward()
    grads = {n:p.grad.detach().cpu().clone() for n,p in actor.named_parameters() if p.grad is not None}
    if len(grads) != 359 or not all(torch.isfinite(g).all() for g in grads.values()):
        raise ValueError('Incomplete or nonfinite full-head gradients')
    del current, selected
    actor.zero_grad(set_to_none=True)
    # Construct only the reference B1 transition, not 160 trainable graphs.
    dummy = {key:torch.zeros_like(pg['chain'][:,:,:-1]) for key in ('mean','std','elementwise_logp')}
    direct = replace_transitions(actor, pg['raw'], pg['chain'], dummy, indices)
    selected = direct['elementwise_logp'].reshape(-1,8,4)[mask]
    if not torch.equal(selected.detach().cpu(), value): raise ValueError('B1 values differ')
    selected.sum().backward()
    mismatch = [n for n,p in actor.named_parameters() if p.grad is None or not torch.equal(p.grad.detach().cpu(),grads[n])]
    if mismatch: raise ValueError('B1 full-head gradients differ: '+str(mismatch))
    actor.zero_grad(set_to_none=True)
    saved = controller.state_dict()
    restored = StableSavedProbability(); restored.load_state_dict(saved, pending_inner=True)
    restored.bind(collector['behavior_identity'], slot)
    with torch.no_grad():
        second, second_entry = restored.statistics(actor, actor.saved_statistics, pg)
    if second_entry['indices'] != indices or second_entry['calls'] != 2:
        raise ValueError('Inner resume changed layout choice')
    atomic_json(output,dict(status='PASS',host=host,checkpoint=str(cp),checkpoint_model_sha256=seals[str(model.relative_to(cp))],
        failure_sha256=file_sha(args.failure),adapter_sha256=file_sha(Path(__file__).with_name('stable_saved_probability.py')),
        failing_chain_probability_pass=True,transition_values_gradients_pass=True,
        original_probability_guard=1e-4,decision=entry,parameter_gradient_tensors=359,
        gradient_reference='Identical selected log-density sum through original B1 transition; all parameter gradients exactly equal',
        restored_inner_selection_exact=True,actual_new_scores=0,optimizer_updates=0,
        wall_seconds=time.monotonic()-start))
    print(json.dumps(dict(status='PASS',indices=indices,entry=entry,gradient_tensors=359)),flush=True)


if __name__=='__main__':main()
