"""Read-only full-model layout diagnostic on real saved failure states.

This loads the last complete checkpoint, not the unavailable failed-update
weights. It cannot by itself certify exact reconstruction of the failed actor.
"""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

from starVLA.rl.constrained_flow_noise.contracts import file_sha, require_dispatch
from starVLA.rl.constrained_flow_noise.loading import make_noise_policy
from starVLA.rl.constrained_flow_noise.rollout_graph import configure
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json


@torch.no_grad()
def compare(actor, row, fresh):
    raw = row['chain']['raw'].cuda()
    saved = row['chain']['chain'].cuda()
    if fresh:
        configure(actor, True)
        sample = actor.sample_chain(raw, [row['token']], [row['position']], version={'read_only_probe':True})
        saved = sample['chain']
        old = sample['old_elementwise_logp']
    else:
        old = None
    configure(actor, False)
    serial = actor.saved_statistics(raw, saved, chunk=1)
    flat = actor.saved_statistics(raw, saved)
    result = dict(position=row['position'], fresh_chain_from_loaded_checkpoint=fresh,
                  mean_max_abs=float((flat['mean']-serial['mean']).abs().max()),
                  std_max_abs=float((flat['std']-serial['std']).abs().max()),
                  elementwise_logp_max_abs=float((flat['elementwise_logp']-serial['elementwise_logp']).abs().max()),
                  mean_ratio_flat_vs_serial_max_abs=float(((flat['mean_transition_logp']-serial['mean_transition_logp']).exp()-1).abs().max()))
    if old is not None:
        for name, values in [('serial', serial), ('flat', flat)]:
            result[name+'_fresh_ratio_error'] = float(((values['mean_transition_logp']-old.mean((-2,-1))).exp()-1).abs().max())
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--failure', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    host = require_dispatch(); configure_numerics(); torch.cuda.set_device(0)
    output = Path(a.output)
    if output.exists(): raise FileExistsError(output)
    cp = Path(a.checkpoint)
    seal = json.loads((cp/'files.json').read_text())
    if not (cp/'COMPLETE').is_file(): raise ValueError('Complete checkpoint required')
    identity = json.loads((cp/'identity.json').read_text())
    for name, digest in identity['source_files'].items():
        if file_sha(name) != digest: raise ValueError('Changed formal source: '+name)
    model = cp/'pytorch_model/mp_rank_00_model_states.pt'
    if file_sha(model) != seal[str(model.relative_to(cp))]: raise ValueError('Changed saved model')
    state = torch.load(a.failure, map_location='cpu', weights_only=False)['state']
    collector = state['collector']
    actor, _, _ = make_noise_policy(identity['variant'], (identity['kernel']['min_std'], identity['kernel']['max_std']))
    actor.load_state_dict(torch.load(model, map_location='cpu', mmap=True, weights_only=False)['module'], strict=True)
    actor.requires_grad_(False).eval()
    record = dict(status='RUNNING', host=host, checkpoint=str(cp),
                  checkpoint_boundary=json.loads((cp/'boundary.json').read_text()),
                  failure_updates=state['updates'], failure_sha256=file_sha(a.failure),
                  analysis_source_sha256=file_sha(__file__), checks=[],
                  scope='No optimizer, no scoring, no new training; loaded checkpoint differs from failed current actor')
    atomic_json(output,record)
    started = time.monotonic()
    # Rank3 can fail in slot3 or slot11. Include their actual accepted groups.
    for slot in (3,11):
        row = collector['rows'][collector['accepted'][slot]]
        for fresh in (False, True):
            item = compare(actor,row,fresh); item['rank3_slot'] = slot
            record['checks'].append(item); atomic_json(output,record)
            print(json.dumps(item),flush=True)
    record.update(status='COMPLETE_DIAGNOSTIC',wall_seconds=time.monotonic()-started,
                  peak_cuda_GiB=torch.cuda.max_memory_allocated()/2**30)
    atomic_json(output,record)


if __name__ == '__main__': main()
