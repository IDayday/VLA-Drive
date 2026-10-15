"""Freeze the complete SFT single-native-ODE canonical score<0.9 TRAIN list."""
import argparse
import json
from pathlib import Path
from collections import Counter
from starVLA.rl.simwam_recipe.contracts import ROOT,POPULATION,SFT_SHA256,fingerprint,file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',default=ROOT);a=p.parse_args()
    root=Path(a.root);scores=root/'SFT_NATIVE_TRAIN_seed2101'
    pop=json.loads(Path(POPULATION).read_text());identity=json.loads((scores/'manifest.json').read_text())
    if identity['SFT_sha256']!=SFT_SHA256 or identity['scene_population']!=pop['identity_sha256'] or identity['seed']!=2101:raise ValueError('Wrong screen identity')
    rows=[];missing=[];types=Counter()
    for token in pop['train_tokens']:
        path=scores/'scenes'/token[:2]/(token+'.json')
        if not path.exists():missing.append(token);continue
        seal=json.loads(path.read_text())
        if seal['identity_sha256']!=identity['sha256']:raise ValueError('Mixed screen protocols')
        if seal['payload_sha256']!=file_sha(path.with_suffix('.pt')):raise ValueError('Modified score payload')
        m=seal['metrics']
        if m['score']<.9:
            labels=[k for k in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound') if m[k]<1]
            types['+'.join(labels) or 'safe_lower_quality_or_progress']+=1
            rows.append(dict(token=token,score=m['score'],metrics=m,score_sha256=file_sha(path)))
    if missing:
        print(json.dumps(dict(status='INCOMPLETE',scored=len(pop['train_tokens'])-len(missing),missing=len(missing),hard_so_far=len(rows))));return
    value=dict(status='COMPLETE_FIXED_SFT_SINGLE_NATIVE_ODE_HARD_SUBSET',population_identity=pop['identity_sha256'],
               screen_identity=identity,tokens=[r['token'] for r in rows],rows=rows,threshold=.9,
               log_count=len({pop['token_logs'][r['token']] for r in rows}),hard_types=dict(types),
               full_TRAIN_scenes=len(pop['train_tokens']),no_DEV_or_Navtest=True)
    value['sha256']=fingerprint(value)
    output=root/'hard_manifest.json'
    if output.exists():
        if json.loads(output.read_text())!=value:raise ValueError('Fixed hard list must not change')
    else:atomic_json(output,value)
    print(json.dumps({k:v for k,v in value.items() if k not in ('rows','tokens','screen_identity')}))


if __name__=='__main__':main()
