"""Verify the entire own-SFT TRAIN screen before fixing source-recipe membership.

Like the existing recipe builder, every score payload is hash checked. A partial
or qualification screen cannot publish a formal training population.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time
from starVLA.rl.fm_transfer.contracts import require_dispatch
from starVLA.rl.simwam_recipe.contracts import POPULATION, file_sha, fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json

SAFETY = ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound')


def build(screen, output, *, workers=8, population=None):
    if not 1 <= workers <= 8:raise ValueError('Bounded shared-filesystem verification')
    screen=Path(screen);output=Path(output)
    pop=population or json.loads(Path(POPULATION).read_text())
    identity=json.loads((screen/'manifest.json').read_text())
    body=dict(identity);seal=body.pop('sha256')
    if fingerprint(body)!=seal or identity['schema']!='fm_transfer_single_native_TRAIN_screen_v1':
        raise ValueError('Changed screen identity')
    parent=identity['new_SFT'];contract=identity['screen_contract'];tokens=pop['train_tokens']
    expected=dict(SFT_model_sha256=parent['model_sha256'],SFT_identity_sha256=parent['identity_sha256'],
        architecture=parent['config']['architecture'],seed=2101,grid='native',threshold=.9,
        statistic='one canonical v1.1 score per scene')
    if (identity['qualification_only'] or parent['qualification'] or
            parent['updates']!=parent['config']['max_updates'] or contract!=expected):
        raise ValueError('Needs this completed formal SFT and its registered single-native screen')
    if (identity['population_identity']!=pop['identity_sha256'] or
            identity['tokens_sha256']!=fingerprint(tokens) or identity['scene_count']!=len(tokens) or
            set(tokens)&(set(pop['dev_tokens'])|set(pop['navtest_tokens'])) or len(set(tokens))!=len(tokens)):
        raise ValueError('Changed/leaking TRAIN population')
    ranks=[json.loads((screen/f'rank{i}.json').read_text()) for i in range(identity['shards'])]
    if any(r['status']!='ASSIGNED_COMPLETE' or r['identity_sha256']!=seal for r in ranks):
        raise ValueError('Incomplete screen; missing scenes must not leave the denominator')
    def verify(token):
        path=screen/'scenes'/token[:2]/(token+'.json')
        row=json.loads(path.read_text())
        if (row['token']!=token or row['log']!=pop['token_logs'][token] or row['identity_sha256']!=seal or
                row['payload_sha256']!=file_sha(path.with_suffix('.pt'))):
            raise ValueError('Changed scene/score payload: '+token)
        metrics=row['metrics']
        if any(not math.isfinite(float(metrics[k])) or not 0<=metrics[k]<=1 for k in ('score',*SAFETY)):
            raise ValueError('Invalid canonical metric; no error-as-zero')
        return dict(token=token,score=metrics['score'],metrics=metrics,score_sha256=file_sha(path))
    rows=[];types=Counter();full_types=Counter();started=time.monotonic()
    progress=output.with_suffix('.verification.json')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for offset in range(0,len(tokens),256):
            for row in pool.map(verify,tokens[offset:offset+256]):
                labels=[k for k in SAFETY if row['metrics'][k]<1]
                key='+'.join(labels) or 'fully_safe'
                full_types[key]+=1
                if row['score']<.9:
                    types['+'.join(labels) or 'safe_lower_quality_or_progress']+=1;rows.append(row)
            atomic_json(progress,dict(status='VERIFYING',checked=min(offset+256,len(tokens)),
                total=len(tokens),workers=workers,elapsed=time.monotonic()-started))
    if not rows:raise ValueError('No hard scenes; no synthetic nonzero advantages or fallback population')
    result=dict(status='COMPLETE_FIXED_SFT_SINGLE_NATIVE_ODE_HARD_SUBSET',population_identity=pop['identity_sha256'],
        screen_identity=identity,screen_contract=contract,screened_scene_count=len(tokens),
        tokens=[r['token'] for r in rows],rows=rows,threshold=.9,
        log_count=len({pop['token_logs'][r['token']] for r in rows}),hard_types=dict(types),
        complete_TRAIN_safety_types=dict(full_types),full_TRAIN_scenes=len(tokens),no_DEV_or_Navtest=True)
    result['sha256']=fingerprint(result)
    if output.exists():
        if json.loads(output.read_text())!=result:raise ValueError('Frozen membership must not change')
    else:atomic_json(output,result)
    atomic_json(progress,dict(status='COMPLETE_ALL_PAYLOAD_HASHES_VERIFIED',checked=len(tokens),total=len(tokens),
        workers=workers,elapsed=time.monotonic()-started,hard_scenes=len(rows),manifest_sha256=file_sha(output)))
    return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--screen',required=True);p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=8);a=p.parse_args();require_dispatch()
    result=build(a.screen,a.output,workers=a.workers)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','tokens','screen_identity')}))


if __name__=='__main__':main()
