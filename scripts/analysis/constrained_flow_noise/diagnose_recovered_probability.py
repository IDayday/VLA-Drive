"""Read-only probability audit at the recovered C_SDE failure boundary.

No new score, optimizer update or changed consistency tolerance. Compare
actual saved old densities with B1 and flat recomputation, and regenerate the
same chain to establish that the recovered weights are the recorded behavior.
"""
import argparse
import json
from pathlib import Path
import time

import torch

from starVLA.rl.constrained_flow_noise.contracts import file_sha, require_dispatch
from starVLA.rl.constrained_flow_noise.loading import make_noise_policy
from starVLA.rl.constrained_flow_noise.rollout_graph import configure
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--recovery-run', required=True)
    parser.add_argument('--failure', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    host = require_dispatch()
    configure_numerics()
    torch.cuda.set_device(0)
    root, output = Path(args.recovery_run), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    checkpoints = []
    for rank in range(8):
        report = json.loads((root / f'recovery_rank{rank}.json').read_text())
        if report['status'] != 'RECOVERED_EXACT_OBSERVED_SEQUENCE' or report['last_verified_update'] != 1850:
            raise ValueError('All eight ranks must finish verified recovery')
        checkpoints.append(report['result_checkpoint'])
    if len(set(checkpoints)) != 1:
        raise ValueError('Ranks disagree on recovery checkpoint')
    checkpoint = Path(checkpoints[0])
    if not (checkpoint / 'COMPLETE').is_file():
        raise ValueError('Incomplete recovered checkpoint')
    identity = json.loads((checkpoint / 'identity.json').read_text())
    for name, digest in identity['source_files'].items():
        if file_sha(name) != digest:
            raise ValueError('Changed formal source: ' + name)
    model = checkpoint / 'pytorch_model/mp_rank_00_model_states.pt'
    seal = json.loads((checkpoint / 'files.json').read_text())
    if file_sha(model) != seal[str(model.relative_to(checkpoint))]:
        raise ValueError('Changed recovered model')
    state = torch.load(args.failure, map_location='cpu', weights_only=False)['state']
    if state['updates'] != 1850 or state['next_inner'] != 0:
        raise ValueError('This diagnostic is for the recorded first-inner failure')
    collector = state['collector']
    actor, _, _ = make_noise_policy(identity['variant'], (identity['kernel']['min_std'], identity['kernel']['max_std']))
    actor.load_state_dict(torch.load(model, map_location='cpu', mmap=True, weights_only=False)['module'], strict=True)
    actor.requires_grad_(False).eval()
    record = dict(status='RUNNING', host=host, recovered_checkpoint=str(checkpoint),
                  source_sha256=file_sha(__file__), failure_sha256=file_sha(args.failure),
                  tolerance_unchanged=1e-4, checks=[], actual_new_score_calls=0,
                  scope='Read-only recovered behavior audit; no optimizer or probability repair')
    atomic_json(output, record)
    started = time.monotonic()
    for slot in (3, 11):
        row = collector['rows'][collector['accepted'][slot]]
        raw, chain = row['chain']['raw'].cuda(), row['chain']['chain'].cuda()
        old = row['chain']['old_elementwise_logp'].cuda().mean((-2, -1))
        configure(actor, True)
        fresh = actor.sample_chain(raw, [row['token']], [row['position']], version=collector['behavior_identity'])
        regeneration = {key: torch.equal(fresh[key].cpu(), row['chain'][key])
                        for key in ('chain', 'old_mean', 'old_std', 'old_elementwise_logp')}
        configure(actor, False)
        serial = actor.saved_statistics(raw, chain, chunk=1)
        flat = actor.saved_statistics(raw, chain)
        item = dict(rank3_slot=slot, position=row['position'], same_behavior_regeneration=regeneration,
                    mean_max_abs=float((serial['mean'] - flat['mean']).abs().max()),
                    std_max_abs=float((serial['std'] - flat['std']).abs().max()))
        for name, statistics in [('serial', serial), ('flat', flat)]:
            error = ((statistics['mean_transition_logp'] - old).exp() - 1).abs()
            item[name] = dict(ratio_max_abs_from_one=float(error.max()),
                              transitions_above_original_tolerance=int((error > 1e-4).sum()),
                              mean_max_abs_vs_recorded=float((statistics['mean'] - row['chain']['old_mean'].cuda()).abs().max()))
        record['checks'].append(item)
        atomic_json(output, record)
        print(json.dumps(item), flush=True)
    record.update(status='COMPLETE_DIAGNOSTIC', wall_seconds=time.monotonic() - started,
                  actual_regenerated_candidates=32,
                  all_saved_behavior_regenerations_exact=all(all(x['same_behavior_regeneration'].values()) for x in record['checks']))
    atomic_json(output, record)


if __name__ == '__main__':
    main()
