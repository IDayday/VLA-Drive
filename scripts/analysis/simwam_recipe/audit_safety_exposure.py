"""Verify actual fixed 6+2 exposure from committed journals, without new rollouts."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from starVLA.rl.simwam_recipe.contracts import (
    ROOT, SAFETY25_VARIANT, file_sha, fingerprint)
from starVLA.rl.simwam_recipe.data_stream import make_scene_stream
from starVLA.rl.simwam_recipe.invocation import validate_rows
from starVLA.rl.flow_grpo.transactions import atomic_json
from scripts.analysis.simwam_recipe.summarize_training import read_journal


def audit_rows(identity, stream, ranks, end):
    if identity['recipe']['variant'] != SAFETY25_VARIANT or identity['world'] != 8:
        raise ValueError('Only the registered eight-rank exposure control is supported')
    if stream.identity != identity['data_stream_identity']:
        raise ValueError('Changed manifest, SFT labels or static data contract')
    if end <= 0 or end % 4 or end > 20000 or len(ranks) != 8:
        raise ValueError('Require a finite complete four-inner window')
    counts=Counter();by_rank={str(rank):Counter() for rank in range(8)}
    safety_types=Counter();unique=set();sequence=[]
    metrics={r['token']:r['metrics'] for r in stream.manifest['rows']}
    for rank,rows in enumerate(ranks):
        validate_rows(rows,len(rows))
        if len(rows)<end:raise ValueError('Missing rank updates')
    for collection in range(end//4):
        labels=[]
        for rank,rows in enumerate(ranks):
            token,position=stream.at(collection*8,rank)
            four=rows[collection*4:collection*4+4]
            for inner,row in enumerate(four,1):
                if (row['run'] != identity['sha256'] or row['token'] != token
                    or row['updates'] != collection*4+inner
                    or row['collection'] != collection+1 or row['inner_completed'] != inner
                    or row['raw_scene_exposures'] != (collection+1)*8
                    or row['new_candidates'] != (collection+1)*64
                    or row['behavior']['run'] != identity['sha256']
                    or row['behavior']['update'] != collection*4):
                    raise ValueError('Actual scene/order/budget/behavior differs from registration')
                for key in ('behavior','chain_identity','exploration'):
                    if row[key] != four[0][key]:
                        raise ValueError('Fresh behavior material changed during inner reuse')
            label=position['stratum'];labels.append(label);counts[label]+=1
            by_rank[str(rank)][label]+=1;unique.add(token);sequence.append((token,position))
            m=metrics[token]
            kind='+'.join(name for name,key in (
                ('NC','no_at_fault_collisions'),('DAC','drivable_area_compliance'),
                ('TTC','time_to_collision_within_bound')) if m[key]<1) or 'SFT_single_seed_safe'
            safety_types[kind]+=1
        if Counter(labels)!=Counter(safe=6,unsafe=2):raise ValueError('Actual quota is not six plus two')
    if counts!=Counter(safe=end//4*6,unsafe=end//4*2):raise ValueError('Exposure totals differ')
    if end==20000 and unique!=set(stream.tokens):raise ValueError('Formal endpoint did not cover every hard scene')
    return dict(status='PASS',checked_optimizer_updates=end,fresh_global_rollouts=end//4,
        scene_exposures=end*2,new_candidates=end*16,actual_exposures=dict(counts),
        per_rank_exposures={k:dict(v) for k,v in by_rank.items()},
        SFT_single_seed_safety_types=dict(safety_types),unique_scenes=len(unique),
        all_hard_members_visited=(unique==set(stream.tokens)),
        actual_input_order_sha256=fingerprint(sequence),all_four_inner_identities_fixed=True,
        no_new_actor_rollout_or_scoring=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',required=True)
    p.add_argument('--manifest',default=ROOT+'/hard_manifest.json')
    p.add_argument('--output',required=True);p.add_argument('--limit-updates',type=int)
    a=p.parse_args();begin=time.monotonic();run=Path(a.run);out=Path(a.output)
    if out.exists():raise FileExistsError('Preserve previous audit')
    identity=json.loads((run/'identity.json').read_text())
    stream=make_scene_stream(identity['recipe'],a.manifest,identity['world'],
        qualification=identity['purpose']=='qualification_not_formal_initialization')
    ranks=[read_journal(run/f'metrics_rank{rank}.jsonl') for rank in range(8)]
    end=min(map(len,ranks))//4*4
    if a.limit_updates is not None:
        if a.limit_updates<=0 or a.limit_updates%4 or a.limit_updates>end:
            raise ValueError('Requested completed window is unavailable')
        end=a.limit_updates
    result=audit_rows(identity,stream,ranks,end)
    result.update(schema='simwam_observed_static_safety_exposure_v1',
        captured_utc=datetime.now(timezone.utc).isoformat(),run_identity=identity['sha256'],
        scientific_identity=identity['scientific_identity']['sha256'],
        manifest_sha256=file_sha(a.manifest),analysis_sha256=file_sha(__file__),
        checked_row_prefix_sha256=[fingerprint(rows[:end]) for rows in ranks],
        measured_analysis_body_seconds=time.monotonic()-begin,
        limits=['This audits actual data exposure and inner reuse, not driving efficacy.',
                'SFT safety labels use a fixed single seed; no current-policy rejection sampling.',
                'Counts reflect committed learning rows; physical failed work is in the invocation cost ledger.'])
    out.parent.mkdir(parents=True,exist_ok=True);atomic_json(out,result)
    print(json.dumps({k:result[k] for k in ('status','checked_optimizer_updates','actual_exposures','unique_scenes')}))


if __name__=='__main__':main()
