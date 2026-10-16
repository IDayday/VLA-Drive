"""Read committed X1 / SAFETY25 prefixes without matching different scene streams.

Only existing journals and unsafe-positive saved groups are read. Conditional
means and fixed-stratum reweighting describe training, never deployed efficacy.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from scripts.analysis.simwam_recipe.analyze_training_safety_signal import (
    aggregate, candidate_signal, safety_type, validate_saved, MEANS)
from scripts.analysis.simwam_recipe.summarize_training import summarize_window
from starVLA.rl.simwam_recipe.contracts import file_sha, fingerprint, require_dispatch
from starVLA.rl.simwam_recipe.data_stream import make_scene_stream
from starVLA.rl.simwam_recipe.invocation import validate_rows
from starVLA.rl.flow_grpo.transactions import atomic_json


def locked_prefix(run, identity, stream, updates):
    checkpoint = run / 'checkpoints' / f'update{updates:06d}_inner0'
    seal = json.loads((checkpoint / 'COMPLETE').read_text())
    if (seal['updates'], seal['phase'], seal['world'], seal['next_inner']) != (updates, 'READY', 8, 0):
        raise ValueError('Need an already committed complete collection boundary')
    if file_sha(checkpoint / 'files.json') != seal['files_sha256']:
        raise ValueError('Changed checkpoint seal')
    files = json.loads((checkpoint / 'files.json').read_text())
    if file_sha(checkpoint / 'transaction.json') != files['transaction.json']:
        raise ValueError('Changed checkpoint transaction')
    transaction = json.loads((checkpoint / 'transaction.json').read_text())
    if len(transaction['ranks']) != 8:
        raise ValueError('Missing committed rank')
    for state in transaction['ranks']:
        if any(state[k] != value for k, value in dict(phase='READY', updates=updates,
                collections=updates//4, scene_cursor=updates*2, next_inner=0,
                run=identity['sha256'], pending=False).items()):
            raise ValueError('Unmatched transaction boundary')
    ranks = []
    for rank in range(8):
        with (run / f'metrics_rank{rank}.jsonl').open() as f:
            rows = [json.loads(line) for line in itertools.islice(f, updates)]
        validate_rows(rows, updates)
        if len(rows) != updates:
            raise ValueError('Missing committed journal prefix')
        for i in range(0, updates, 4):
            group = rows[i:i+4]
            token, _ = stream.at(i*2, rank)
            for inner, row in enumerate(group, 1):
                if (row['run'] != identity['sha256'] or row['token'] != token
                    or row['updates'] != i+inner or row['collection'] != i//4+1
                    or row['inner_completed'] != inner or row['raw_scene_exposures'] != (i//4+1)*8
                    or row['new_candidates'] != (i//4+1)*64
                    or row['behavior']['run'] != identity['sha256'] or row['behavior']['update'] != i):
                    raise ValueError('Data, update or behavior attribution mismatch')
                if any(row[k] != group[0][k] for k in ('chain_identity', 'behavior', 'exploration')):
                    raise ValueError('Four inner updates did not preserve the collection')
        ranks.append(rows)
    for i in range(0, updates, 4):
        if len({r[i]['behavior']['adapter_sha256'] for r in ranks}) != 1:
            raise ValueError('Ranks used different behavior')
        if len({r[i]['exploration']['global_reward_std'] for r in ranks}) != 1:
            raise ValueError('Ranks did not share raw reward std')
    return ranks, dict(checkpoint_files_sha256=seal['files_sha256'],
        transaction_sha256=file_sha(checkpoint/'transaction.json'),
        journal_prefix_sha256=[fingerprint(rows) for rows in ranks])


def analyze(registration, output):
    start = time.monotonic()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('This bounded audit uses no GPU')
    host = require_dispatch()
    registration, output = Path(registration), Path(output)
    if output.exists(): raise FileExistsError(output)
    reg = json.loads(registration.read_text()); updates = reg['optimizer_updates']
    if updates != 2500: raise ValueError('Only the preregistered 2500 prefix is admitted')
    for field in ('hard_manifest', 'population'):
        if file_sha(reg[field]['path']) != reg[field]['sha256']:
            raise ValueError('Changed fixed data identity')
    manifest = json.loads(Path(reg['hard_manifest']['path']).read_text())
    population = json.loads(Path(reg['population']['path']).read_text())
    hard = {r['token']: r for r in manifest['rows']}
    if len(hard) != len(manifest['rows']) or set(hard) != set(manifest['tokens']):
        raise ValueError('Incomplete SFT classification')
    held = set(population['dev_tokens']) | set(population['navtest_tokens'])
    if not set(hard) <= set(population['train_tokens']) or set(hard) & held:
        raise ValueError('Only the registered TRAIN hard population is admitted')
    if {population['token_logs'][t] for t in hard} & {population['token_logs'][t] for t in held}:
        raise ValueError('Held-out log overlap')
    types = {t:safety_type(row['metrics']) for t,row in hard.items()}
    cohort = {t:('safe' if k=='SFT_single_seed_safe' else 'unsafe') for t,k in types.items()}
    population_count = Counter(cohort.values()); type_count = Counter(types.values())
    identities = [json.loads((Path(r['run'])/'identity.json').read_text()) for r in reg['runs']]
    for key in ('SFT_sha256','hard_manifest_sha256','projected_condition_identity','score_protocol',
                'actual_optimizer','parameter_contract'):
        if identities[0][key] != identities[1][key]: raise ValueError('Unmatched shared contract: '+key)
    diffs = {k for k in set(identities[0]['recipe'])|set(identities[1]['recipe'])
             if identities[0]['recipe'].get(k) != identities[1]['recipe'].get(k)}
    if diffs != {'variant','data_order','scene_sampling'}: raise ValueError('Additional recipe difference')
    if [v['recipe']['variant'] for v in identities] != ['R_CANONICAL_X1','R_CANONICAL_X1_SAFETY25']:
        raise ValueError('Only the locked exposure comparison is admitted')
    output.mkdir(parents=True)
    summaries = []; initial = []; selected_total = 0; saved_bytes = 0
    for locked, identity in zip(reg['runs'], identities):
        run = Path(locked['run'])
        if identity['sha256'] != locked['run_identity'] or file_sha(run/'identity.json') != locked['identity_file_sha256']:
            raise ValueError('Changed registered run')
        stream = make_scene_stream(identity['recipe'],reg['hard_manifest']['path'],8)
        if hasattr(stream,'identity') and stream.identity != identity['data_stream_identity']:
            raise ValueError('Changed static exposure stream identity')
        ranks, prefix_identity = locked_prefix(run,identity,stream,updates)
        initial.append([rows[0]['behavior']['adapter_sha256'] for rows in ranks])
        fresh = [(rank,row) for rank,rows in enumerate(ranks) for row in rows[::4]]
        all_rows = [row for _,row in fresh]
        strata = {key:aggregate([r for r in all_rows if cohort[r['token']]==key],count,population['token_logs'])
                  for key,count in population_count.items()}
        by_type = {key:aggregate([r for r in all_rows if types[r['token']]==key],count,population['token_logs'])
                   for key,count in type_count.items()}
        if identity['recipe']['variant'].endswith('SAFETY25'):
            if strata['safe']['scene_exposures'] != 3750 or strata['unsafe']['scene_exposures'] != 1250:
                raise ValueError('Registered 6+2 exposure not observed')
        standardized = {k:sum(population_count[c]/len(hard)*strata[c]['stochastic_mean'][k]
                              for c in ('safe','unsafe')) for k in MEANS}
        selected = [(rank,row) for rank,row in fresh if row['exploration']['positive_still_unsafe']>0]
        if len(selected) > reg['maximum_selected_groups_per_run']:
            raise ValueError('Saved-chain read budget exceeded; no truncation allowed')
        counters=Counter(); violation_types=Counter(); gaps=[]
        private = output / (identity['recipe']['variant']+'_selected_private.jsonl')
        with private.open('x') as f:
            for rank,row in selected:
                p=run/'behavior'/f"collection{row['collection']:06d}"/f'rank{rank}.pt'
                saved=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
                chain_identity=validate_saved(saved,row,rank,identity,population,stream)
                signal=candidate_signal(saved['records'],saved['advantages'].numpy(),saved['adv_mask'].numpy())
                if signal['raw_positive_unsafe']!=row['exploration']['positive_still_unsafe']:
                    raise ValueError('Candidate and journal counts disagree')
                for k,v in signal.items():
                    if isinstance(v,(int,float)) and not isinstance(v,bool) and k!='score_comparison_tolerance':counters[k]+=v
                violation_types.update(signal['unsafe_positive_types']);gaps.extend(signal['unsafe_positive_minus_best_safe'])
                saved_bytes+=p.stat().st_size
                f.write(json.dumps(dict(rank=rank,collection=row['collection'],token=row['token'],
                    saved_sha256=file_sha(p),chain_identity=chain_identity,signal=signal),allow_nan=False)+'\n')
        overall=aggregate(all_rows,len(hard),population['token_logs'])
        if counters['active_positive_unsafe']!=overall['active_PG_positive_candidate_counts']['positive_still_unsafe']:
            raise ValueError('Selected group totals do not cover all unsafe positives')
        selected_total+=len(selected)
        raw_std=[rows['exploration']['global_reward_std'] for rows in ranks[0][::4]]
        summaries.append(dict(variant=identity['recipe']['variant'],run_identity=identity['sha256'],
            prefix_identity=prefix_identity,training=summarize_window(ranks,0,updates),overall=overall,
            by_fixed_SFT_stratum=strata,by_fixed_SFT_safety_type=by_type,
            original_hard_population_standardized_random_mean=standardized,
            raw_global_std_quantiles={str(q):float(np.quantile(raw_std,q)) for q in [0,.05,.5,.95,1]},
            selected_unsafe_positive=dict(groups=len(selected),candidate_records=len(selected)*8,
                all_original_chain_digests_verified=True,counters=dict(counters),violation_types=dict(violation_types),
                gap_to_best_safe_quantiles={str(q):float(np.quantile(gaps,q)) for q in [0,.05,.5,.95,1]} if gaps else {},
                private_evidence_sha256=file_sha(private))))
    if initial[0]!=initial[1] or len(set(initial[0]))!=1: raise ValueError('Different initial LoRA')
    source_paths=[Path(__file__),Path(__file__).with_name('analyze_training_safety_signal.py'),
                  Path(__file__).with_name('summarize_training.py'),Path('starVLA/rl/simwam_recipe/data_stream.py')]
    result=dict(schema='simwam_safety_exposure_prefix_v1',status='COMPLETE_READ_ONLY',
        captured_utc=datetime.now(timezone.utc).isoformat(),host=host,registration_sha256=file_sha(registration),
        analysis_sources={str(p):file_sha(p) for p in source_paths},optimizer_updates=updates,
        same_SFT_initial_LoRA_and_other_recipe_fields=True,same_scene_order=False,
        hard_population_count=dict(population_count),runs=summaries,
        cost=dict(new_rollout_candidates=0,new_scores=0,optimizer_updates=0,GPU_used=False,
            selected_saved_groups=selected_total,candidate_records=selected_total*8,
            saved_bytes_read_and_hashed=saved_bytes,analysis_body_seconds=time.monotonic()-start),
        limitations=['Different scene streams and evolving policies: random means are not a paired efficacy comparison.',
            'Fixed-stratum reweighting removes only the safe/unsafe mixture, not within-stratum scene, policy-age or repeat differences.',
            'Counts and advantage sums are not parameter gradient contributions; gradient_norm contains PG plus anchor.',
            'The global std is validated for saved/rank consistency, not independently reconstructed from every unselected candidate.',
            'No model selection, coefficient tuning, DEV/Navtest, new reward or training update. Fixed 5k/10k/15k/20k ODE gates unchanged.'])
    atomic_json(output/'summary.json',result)
    print(json.dumps(dict(status=result['status'],updates=updates,selected_groups=selected_total,
                         analysis_seconds=result['cost']['analysis_body_seconds'],new_scores=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--registration',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();analyze(a.registration,a.output)
