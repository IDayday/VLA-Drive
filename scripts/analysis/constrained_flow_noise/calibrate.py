"""Finite TRAIN-only noise and physical-cost calibration; no policy updates."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.policy_adapter import decode_request
from starVLA.rl.constrained_flow_noise.contracts import file_sha, fingerprint, require_dispatch, SFT_SHA256
from starVLA.rl.constrained_flow_noise.condition_cache import ConditionStore
from starVLA.rl.constrained_flow_noise.loading import make_noise_policy
from starVLA.rl.constrained_flow_noise.noise_policy import legacy_std_range
from starVLA.rl.constrained_flow_noise.safety_costs import ConstrainedRewardService, costs_from_record


def save_payload(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    torch.save(value, temporary); os.replace(temporary, path)


def summarize(rows):
    records = [r for row in rows for r in row['records']]
    metrics = {key: float(np.mean([r['metrics'][key] for r in records])) for key in records[0]['metrics']}
    safe = np.array([all(r['metrics'][key] == 1 for key in ('no_at_fault_collisions', 'drivable_area_compliance', 'time_to_collision_within_bound')) for r in records])
    jitter = np.concatenate([np.linalg.norm(np.diff(np.asarray(row['requests'])[..., :2], n=2, axis=1), axis=-1).mean(-1) for row in rows])
    ade = []
    for row in rows:
        x = np.asarray(row['requests'])[..., :2]
        differences = np.linalg.norm(x[:,None]-x[None,:], axis=-1).mean(-1)
        ade.extend(differences[np.triu_indices(16,1)].tolist())
    return dict(scenes=len(rows), candidates=len(records), metrics=metrics,
                joint_safe_fraction=float(safe.mean()), median_second_difference_m=float(np.median(jitter)),
                mean_pairwise_ADE_m=float(np.mean(ade)), candidate_forward_passes=len(rows)*16*10,
                GPU_seconds=sum(row['rollout_seconds'] for row in rows),
                scoring_service_seconds=sum(row['scoring_seconds'] for row in rows))


def load_rows(out, population, tokens):
    rows = []
    for token in tokens:
        path = out/'scenes'/(token+'.pt')
        row = torch.load(path, map_location='cpu', weights_only=False)
        seal = json.loads(path.with_suffix('.json').read_text())
        if seal['sha256'] != file_sha(path) or row['token'] != token or row['population_identity'] != population:
            raise ValueError('Calibration scene changed')
        rows.append(row)
    return rows


def lock(a, registration, pop):
    root = Path(a.output); tokens = registration['calibration_tokens']
    baseline = load_rows(root/'C_SDE', pop['identity_sha256'], tokens)
    summary = {'C_SDE': summarize(baseline)}
    thresholds = {}
    for name, field in [('road_margin', 'road_clearance_m'), ('object_margin', 'object_distance_m')]:
        values = [x for row in baseline for r in row['records']
                  if all(r['metrics'][key] == 1 for key in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound'))
                  for x in r['geometry'][field][1:] if x is not None and x > 0]
        if not values:
            raise ValueError('No positive safe physical margin; calibration stopped')
        threshold = float(np.quantile(values, .05))
        if not .01 <= threshold <= 20:
            raise ValueError(f'Nonphysical/degenerate {name} threshold {threshold}; calibration stopped')
        thresholds[name] = threshold
    costs = np.stack([costs_from_record(r, thresholds) for row in baseline for r in row['records']])
    chosen = None
    for scale in (1., .5):
        tag = 'N_FLOW_NOISE' if scale == 1 else 'N_FLOW_NOISE_half'
        if not (root/tag).exists():
            continue
        rows = load_rows(root/tag, pop['identity_sha256'], tokens)
        result = summarize(rows); summary[tag] = result
        safe_ok = result['joint_safe_fraction'] >= summary['C_SDE']['joint_safe_fraction']-.05
        jitter_ok = result['median_second_difference_m'] <= 2*summary['C_SDE']['median_second_difference_m']
        result.update(safe_acceptance=safe_ok, jitter_acceptance=jitter_ok)
        if safe_ok and jitter_ok:
            chosen = scale; break
    report = dict(schema='constrained_flow_noise_calibration_v1', population_identity=pop['identity_sha256'],
                  registration_sha256=file_sha(a.registration), thresholds=thresholds, budgets=costs.mean(0).tolist(),
                  component_floors=[.01, 1e-8, 1e-8, 1e-8, *np.maximum(.001,costs[:,3:].std(0)).tolist()],
                  summaries=summary, selected_range_scale=chosen, status='PASS' if chosen else 'RANGE_NOT_ACCEPTED')
    bounds = legacy_std_range()[:2]
    if chosen:
        tag = 'N_FLOW_NOISE' if chosen == 1 else 'N_FLOW_NOISE_half'
        report.update(noise_std_bounds=[chosen*x for x in bounds],
                      noise_initial_path=str(root/tag/'initial_noise.pt'), noise_initial_sha256=file_sha(root/tag/'initial_noise.pt'))
    report['sha256'] = fingerprint(report)
    dest = root/('calibration_lock.json' if chosen else 'range1_rejected.json')
    if dest.exists():
        raise FileExistsError(dest)
    atomic_json(dest, report)
    print(json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--registration', required=True); p.add_argument('--population', required=True)
    p.add_argument('--condition-cache', required=True); p.add_argument('--output', required=True)
    p.add_argument('--variant', choices=['C_SDE','N_FLOW_NOISE','lock'], required=True)
    p.add_argument('--scale', type=float, choices=[1.,.5], default=1.)
    p.add_argument('--metric-cache', action='append', required=True)
    p.add_argument('--resume', action='store_true'); p.add_argument('--workers', type=int, default=2)
    a = p.parse_args(); host = require_dispatch(); configure_numerics()
    registration, pop = [json.loads(Path(path).read_text()) for path in (a.registration,a.population)]
    tokens = registration['calibration_tokens']
    if not set(tokens) <= set(pop['train_tokens']) or registration['population_identity'] != pop['identity_sha256']:
        raise ValueError('Only preregistered TRAIN may calibrate costs/noise')
    if a.variant == 'lock':
        return lock(a,registration,pop)
    if a.scale != 1 and (a.variant != 'N_FLOW_NOISE' or not (Path(a.output)/'range1_rejected.json').is_file()):
        raise ValueError('Only the single preregistered fallback after first-range rejection is permitted')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)))
    rank,world = int(os.environ.get('RANK',0)),int(os.environ.get('WORLD_SIZE',1))
    root = Path(a.output)/(a.variant if a.scale == 1 else a.variant+'_half')
    root.mkdir(parents=True,exist_ok=True)
    if (root/f'rank{rank}.json').exists() and not a.resume:
        raise FileExistsError('Use --resume for existing calibration evidence')
    identity = json.loads((Path(a.condition_cache)/'identity.json').read_text())
    # Calibration may consume individually sealed priority scenes while the
    # full producer continues. Formal training requires the complete read-only store.
    store = ConditionStore(a.condition_cache, identity, readonly=False)
    bounds = [a.scale*x for x in legacy_std_range()[:2]]
    actor,_,sft = make_noise_policy(a.variant,bounds)
    version = dict(SFT_sha256=SFT_SHA256,kernel=vars(actor.spec),condition_identity=store.identity_sha,
                   action_storage='BF16', probability_arithmetic='FP32', phase='calibration_only')
    from starVLA.rl.constrained_flow_noise import noise_policy, safety_costs, loading, transition
    version['source_files'] = {str(Path(m.__file__).relative_to(Path.cwd())):file_sha(m.__file__)
                               for m in (noise_policy,safety_costs,loading,transition)}
    if actor.noise_head is not None and rank == 0:
        path = root/'initial_noise.pt'
        state = {k:v.detach().cpu() for k,v in actor.noise_head.state_dict().items()}
        if path.exists():
            previous = torch.load(path,weights_only=True)
            if any(not torch.equal(state[k],previous[k]) for k in state):
                raise ValueError('Noise initialization changed')
        else:
            save_payload(path,state)
    service = ConstrainedRewardService(dict(repo=str(Path.cwd()),maps_root='/mnt/navsim/maps',cache_roots=a.metric_cache),workers=a.workers)
    started = time.monotonic(); completed=0
    try:
        for position in range(rank,len(tokens),world):
            token=tokens[position]; path=root/'scenes'/(token+'.pt')
            if path.with_suffix('.json').exists():
                if not a.resume: raise FileExistsError(path)
                if json.loads(path.with_suffix('.json').read_text())['sha256'] != file_sha(path): raise ValueError('Changed calibration evidence')
                completed += 1; continue
            raw = store.get(token)
            if raw is None: raise FileNotFoundError('Priority condition missing: '+token)
            t0=time.monotonic()
            chain=actor.sample_chain(raw.cuda(),[token],[position],version=version)
            torch.cuda.synchronize(); rollout_seconds=time.monotonic()-t0
            requests=decode_request(chain['chain'][0,:,-1].float().cpu().numpy(),act_norm=sft.datasets.vla_data.act_norm)
            t0=time.monotonic(); records=service.score([(token,pop['token_logs'][token],requests)])[0]
            scoring_seconds=time.monotonic()-t0
            chain={k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in chain.items()}
            row=dict(token=token,log=pop['token_logs'][token],population_identity=pop['identity_sha256'],
                     records=records,requests=requests,chain=chain,rollout_seconds=rollout_seconds,scoring_seconds=scoring_seconds)
            save_payload(path,row); atomic_json(path.with_suffix('.json'),dict(sha256=file_sha(path),version=version))
            completed+=1
            atomic_json(root/f'rank{rank}.json',dict(status='RUNNING',host=host,completed=completed,world=world,rank=rank,wall_seconds=time.monotonic()-started))
    finally:
        service.close()
    atomic_json(root/f'rank{rank}.json',dict(status='COMPLETE',host=host,completed=completed,world=world,rank=rank,
                 wall_seconds=time.monotonic()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))


if __name__ == '__main__':
    main()
