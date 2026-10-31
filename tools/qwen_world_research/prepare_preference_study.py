"""Seal a matched sampling-control / offline-preference-auxiliary study."""
from collections import Counter
from pathlib import Path
import copy
import json
import time
import yaml

from .campaign_owner import atomic_json, sha_file, fingerprint
from starVLA.rl.qwen_world_research.runtime import validate_recipe
from starVLA.rl.qwen_world_research_data.manifest import (
    read_sealed, write_stream_manifest, sealed,
)
from starVLA.rl.qwen_world_research_data.adaptive import stage_census, STAGE_SCHEMA

ROOT=Path('/mnt/project/qwen-world-rl-optimization-20261030')


def main():
    receipt=ROOT/'PREFERENCE_MIXTURE_STUDY_REGISTRATION.json'
    if receipt.exists():raise FileExistsError('Never replace a registered preference study')
    bank=ROOT/'data/preferences/TRAIN_METRIC_PREFERENCES_20261031/manifest.json'
    meta=json.loads(bank.read_text())
    pairs_path=Path(meta['artifacts']['pairs.jsonl']['path'])
    if sha_file(pairs_path)!=meta['artifacts']['pairs.jsonl']['sha256']:
        raise ValueError('Preference labels changed')
    pairs=[json.loads(line)for line in pairs_path.read_text().splitlines()]
    preferred={p['token']for p in pairs}
    full=read_sealed(ROOT/'data/manifests/FULL_103288.json')
    tokens=full['tokens']
    if len(preferred)!=910 or not preferred<=set(tokens) or len(pairs)!=5261:
        raise ValueError('Expected the reviewed TRAIN-only preference data')
    stages=ROOT/'preference_mixture_stages'
    manifest=ROOT/'data/manifests/PREFERENCE_MIXTURE_FULL80_PRIORITY20.json'
    if manifest.exists():raise FileExistsError(manifest)
    write_stream_manifest(manifest,data_root=ROOT/'data/NAVTRAIN',tokens=tokens,
        name='full80_pref20_static_v1',mode='adaptive',seed=42,
        stage_exposures=10000,stage_registry=stages)
    stream_sha=sha_file(manifest)
    weights=[.8/len(tokens)+(.2/len(preferred)if t in preferred else 0.)for t in tokens]
    census_receipts=[]
    for index in range(10):
        seed=int(fingerprint(dict(seed=42,stage_index=index,stream_sha256=stream_sha))[:16],16)
        census=stage_census(tokens,weights,10000,seed,world_size=8,adaptive=True)
        counts=Counter(census)
        stage=dict(schema=STAGE_SCHEMA,stream_manifest_sha256=stream_sha,
            stage_index=index,start_cursor=index*10000,scene_exposures=10000,
            ordered_tokens=census,mode='fixed_preference_mixture',world_size=8,seed=seed,
            unique_scenes=len(counts),max_scene_repeats=max(counts.values()),
            sources=[dict(path=str(bank),sha256=sha_file(bank),role='TRAIN preference scene priority only')],
            priority_diagnostics=dict(uniform_mass=.8,preference_mass=.2,
                preferred_scenes=len(preferred),eligible_scenes=len(tokens),
                actual_preference_scene_exposures=sum(counts[t]for t in preferred),
                expected_preference_scene_mass=.2+.8*len(preferred)/len(tokens)),
            selection='weighted draws across batches; distinct8 within batch; identical sealed stream for both arms',
            census_selected_before_optimizer_stage=True,NAVTEST_used_for_selection=False)
        path=stages/f'stage{index:06}.json';atomic_json(path,sealed(stage))
        census_receipts.append(dict(path=str(path),sha256=sha_file(path),
            unique_scenes=len(counts),preference_exposures=stage['priority_diagnostics']['actual_preference_scene_exposures']))
    configs=[]
    for arm in ('PPO_PREF_REPLAY','PPO_PREF_CFM'):
        cfg=yaml.safe_load((ROOT/'configs/PPO_FULL.yaml').read_text())
        cfg['data_order']='identical sealed full80-pref20 mixture, full103288 eligible; offline pair never enters PPO advantages'
        cfg['research'].update(arm=arm,data_policy='preference_mixture',
            manifest_sha256=sha_file(manifest),preference_distribution='full80_pref20_static_v1',
            preference_manifest=str(bank),preference_manifest_sha256=sha_file(bank))
        if arm=='PPO_PREF_CFM':
            cfg['research']['preference_auxiliary']=dict(schema='world_offline_cfm_preference_aux_v1',
                beta=.1,coefficient=.001,margin=0.,sigma_min=.2,sigma_max=.8,stateless_seed=42)
        validate_recipe(cfg)
        path=ROOT/'configs'/f'{arm}.yaml'
        if path.exists():raise FileExistsError(path)
        path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        configs.append(dict(arm=arm,path=str(path),sha256=sha_file(path),research_budget_updates=5000,
                            native_scheduler_max_updates=50000))
    atomic_json(receipt,dict(schema='world_matched_preference_mixture_study_v1',registered_unix=time.time(),
        bank=str(bank),bank_sha256=sha_file(bank),pairs=len(pairs),preference_scenes=len(preferred),
        eligible_scenes=len(tokens),sampler_manifest=str(manifest),sampler_sha256=stream_sha,
        stages=census_receipts,configs=configs,
        comparisons=['REPLAY vs FULL isolates the registered sampling change',
            'CFM vs REPLAY isolates the registered auxiliary preference loss',
            'CFM vs FULL evaluates the complete proposed method'],
        inference='unchanged single ODE10 shift5 seed2101',training_seed=42,
        IL_parent='unchanged pinned World IL70',
        labels='candidate-metric preferences; no claim of human traffic-rule/necessary-wait optimality',
        historical_pairs_in_onpolicy_advantages=False,extra_GPU_qualification=False,
        initial_reviews=[1000,5000],all_future_stages_presealed=True,
        min_full_population_probability=.8/len(tokens)))
    print(json.dumps(dict(registration=str(receipt),configs=configs)))


if __name__=='__main__':main()
