"""Read-only reward counterfactual on preselected, authenticated TRAIN chains.

No refitting, new reward call, actor update, or claimed counterfactual policy.
Global statistics retain all eight original ranks before tied-group masking.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from starVLA.rl.fm_transfer.contracts import require_dispatch
from starVLA.rl.simwam_recipe.contracts import POPULATION, file_sha, fingerprint
from starVLA.rl.simwam_recipe.recovery_state import rollout_digest
from starVLA.rl.simwam_recipe.reward_readouts import readouts
from starVLA.rl.simwam_recipe.source_advantages import advantages
from starVLA.rl.flow_grpo.transactions import atomic_json


def compare_rewards(source, canonical, metrics):
    """One fresh [world=8,G=8] collection; return sufficient statistics."""
    source, canonical = source.float(), canonical.float()
    if source.shape != (8, 8) or canonical.shape != source.shape or metrics.shape != (8, 8, 5):
        raise ValueError('Require eight complete scene groups of eight candidates')
    s, c = advantages(source), advantages(canonical)
    a = (s['advantages'] * s['adv_mask']).reshape(8, 8).clamp(-5, 5).double().numpy()
    b = (c['advantages'] * c['adv_mask']).reshape(8, 8).clamp(-5, 5).double().numpy()
    safe = (metrics[:, :, :3] == 1).all(-1)
    progress_only = (np.ptp(metrics[:, :, [0, 1, 2, 4]], axis=1) == 0).all(-1) & (np.ptp(source.numpy(), axis=1) > 0)
    counts = Counter(groups=8, candidates=64, progress_only_groups=int(progress_only.sum()),
        source_effective_groups=int((s['adv_mask'].reshape(8, 8).sum(1) > 0).sum()),
        canonical_effective_groups=int((c['adv_mask'].reshape(8, 8).sum(1) > 0).sum()),
        source_positive=int((a > 0).sum()), canonical_positive=int((b > 0).sum()),
        source_positive_unsafe=int(((a > 0) & ~safe).sum()),
        canonical_positive_unsafe=int(((b > 0) & ~safe).sum()),
        source_positive_unsafe_with_safe_alternative=int(((a > 0) & ~safe & safe.any(1)[:, None]).sum()),
        canonical_positive_unsafe_with_safe_alternative=int(((b > 0) & ~safe & safe.any(1)[:, None]).sum()),
        robust_advantage_sign_flips=int(((a*b < 0) & (np.abs(a) > 1e-6) & (np.abs(b) > 1e-6)).sum()),
        progress_only_robust_sign_flips=int(((a*b < 0) & (np.abs(a) > 1e-6) & (np.abs(b) > 1e-6) & progress_only[:, None]).sum()),
        source_unsafe_positive_becomes_nonpositive=int(((a > 1e-6) & ~safe & (b <= 1e-6)).sum()),
        safe_candidate_groups=int(safe.any(1).sum()), all_unsafe_groups=int((~safe.any(1)).sum()))
    for name, r in [('source', source.numpy()), ('canonical', canonical.numpy())]:
        bad_pair = (~safe[:, :, None]) & safe[:, None, :] & (r[:, :, None] > r[:, None, :] + 1e-7)
        counts[name+'_unsafe_outranks_safe_pairs'] = int(bad_pair.sum())
        counts[name+'_unsafe_outranks_safe_groups'] = int(bad_pair.any((1, 2)).sum())
    counts['best_candidate_index_changed_groups'] = int((source.argmax(1) != canonical.argmax(1)).sum())
    mass = dict(source_L1=float(np.abs(a).sum()), canonical_L1=float(np.abs(b).sum()),
        source_progress_only_L1=float(np.abs(a[progress_only]).sum()),
        canonical_progress_only_L1=float(np.abs(b[progress_only]).sum()),
        source_unsafe_L1=float(np.abs(a[~safe]).sum()), canonical_unsafe_L1=float(np.abs(b[~safe]).sum()))
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    return dict(counts=dict(counts), mass=mass,
        advantage_cosine=float((a*b).sum()/norm) if norm else None,
        source_std=float(s['global_std']), canonical_std=float(c['global_std']),
        source_advantages=s['advantages'], source_mask=s['adv_mask'])


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args=p.parse_args(); require_dispatch(); start=time.monotonic()
    if args.output.exists(): raise FileExistsError(args.output)
    identity=json.loads((args.run/'identity.json').read_text())
    if identity['purpose']!='formal' or identity['world']!=8 or identity['recipe']['max_steps']!=20000:
        raise ValueError('Not the fixed completed 20k experiment')
    chosen=set(range(1, 5001, 10))
    journals=[]
    for rank in range(8):
        selected={}; seen=set()
        with (args.run/f'metrics_rank{rank}.jsonl').open() as f:
            for line in f:
                row=json.loads(line); u=row['updates']
                if u in seen or u!=len(seen)+1 or row['run']!=identity['sha256']:
                    raise ValueError('Ambiguous journal ancestry/order')
                seen.add(u)
                if row['inner_completed']==1 and row['collection'] in chosen:
                    selected[row['collection']]=row
        if len(seen)!=20000 or set(selected)!=chosen: raise ValueError('Incomplete training evidence')
        journals.append(selected)
    population=json.loads(Path(POPULATION).read_text()); train=set(population['train_tokens'])
    counts=Counter();mass=Counter();cosines=[];stds=[];digests=[];unique=set();phase={}
    for collection in sorted(chosen):
        rows=[]; source=[];canonical=[];metrics=[]
        for rank in range(8):
            path=args.run/'behavior'/f'collection{collection:06}'/f'rank{rank}.pt'
            row=torch.load(path,map_location='cpu',weights_only=False)
            journal=journals[rank][collection]
            if (row['token'] not in train or row['token']!=journal['token'] or
                row['behavior_version']!=journal['behavior'] or rollout_digest(row)!=journal['chain_identity'] or
                row['behavior_version']['update']!=(collection-1)*4):
                raise ValueError('Changed or non-TRAIN saved behavior')
            rec=row['records']; values=[readouts(r) for r in rec]
            r=torch.tensor([v['train_reward_source'] for v in values],dtype=torch.float32)
            if not torch.equal(r,row['rewards'].reshape(-1)): raise ValueError('Stored reward mismatch')
            if len(rec)!=8: raise ValueError('Partial candidate group')
            source.append(r); canonical.append(torch.tensor([v['canonical_PDMS'] for v in values]))
            metrics.append([[r['metrics'][k] for k in ('no_at_fault_collisions','drivable_area_compliance',
                'time_to_collision_within_bound','ego_progress','comfort')] for r in rec])
            rows.append(row);unique.add(row['token']);digests.append(file_sha(path))
        out=compare_rewards(torch.stack(source),torch.stack(canonical),np.asarray(metrics))
        for rank,row in enumerate(rows):
            if not torch.equal(out['source_advantages'].reshape(8,8)[rank],row['advantages']):
                raise ValueError('Original global advantage failed exact reconstruction')
            if not torch.equal(out['source_mask'].reshape(8,8)[rank],row['adv_mask']):
                raise ValueError('Original PG mask mismatch')
        counts.update(out['counts']);mass.update(out['mass'])
        key=f'updates_{((collection-1)//1250)*5000:05}_{((collection-1)//1250+1)*5000:05}'
        phase.setdefault(key,Counter()).update(out['counts'])
        if out['advantage_cosine'] is not None:cosines.append(out['advantage_cosine'])
        stds.append([out['source_std'],out['canonical_std']])
    result=dict(status='COMPLETE_READONLY_TRAIN_REWARD_COUNTERFACTUAL',schema=1,
        run_identity=identity['sha256'],source_sha256=file_sha(__file__),
        selection='collections 1,11,...,4991; all original ranks/candidates; fixed before outcomes',
        collections=len(chosen),unique_scenes=len(unique),payload_digest=fingerprint(digests),
        exact_original_advantage_reconstruction=True,counts=dict(counts),advantage_absolute_mass=dict(mass),
        phases={k:dict(v) for k,v in phase.items()},
        advantage_cosine=dict(mean=float(np.mean(cosines)),min=float(min(cosines)),p05=float(np.quantile(cosines,.05))),
        raw_global_std_mean=np.mean(stds,axis=0).tolist(),wall_seconds=time.monotonic()-start,
        new_candidates=0,optimizer_updates=0,new_scoring=0,
        limits=['Systematic 10% collection sample, not the full training population.',
                'Advantage mass/cosine are not parameter gradient norms or policy performance.',
                'Reweighted same behavior; cannot predict a separately trained canonical policy.',
                'Argmax index changes are literal and may include ties.'])
    args.output.parent.mkdir(parents=True,exist_ok=True);atomic_json(args.output,result)
    print(json.dumps(dict(status=result['status'],counts=result['counts'],output=str(args.output))))


if __name__=='__main__':main()
