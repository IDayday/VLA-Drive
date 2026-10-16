"""Finite CPU acceptance of the static 6+2 TRAIN exposure control."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from omegaconf import OmegaConf
import torch

from starVLA.rl.simwam_recipe.contracts import (
    ROOT,POPULATION,SAFETY25_VARIANT,SAFETY25_DATA_ORDER,SAFETY25_SAMPLING,
    load_recipe,file_sha,fingerprint,registered_stochastic_steps)
from starVLA.rl.simwam_recipe.data_stream import SceneStream,FixedSafetyMixtureStream,make_scene_stream
from starVLA.rl.simwam_recipe.lineage import scientific_identity
from scripts.analysis.simwam_recipe.registered_followup import matched_control
from starVLA.rl.flow_grpo.transactions import atomic_json


def qualify(manifest):
    configs=Path('configs/flow_rl');checks=[]
    control=load_recipe(configs/'simwam_canonical_x1_ddp.yaml')
    candidate=load_recipe(configs/'simwam_canonical_x1_safety25_ddp.yaml')
    expected=copy.deepcopy(control)
    expected.update(variant=SAFETY25_VARIANT,data_order=SAFETY25_DATA_ORDER,
                    scene_sampling=copy.deepcopy(SAFETY25_SAMPLING))
    assert candidate==expected;checks.append('complete_recipe_only_exposure_change')
    assert registered_stochastic_steps(SAFETY25_VARIANT)==[7,8,9]
    checks.append('source_random_window_unchanged')
    state=torch.get_rng_state().clone()
    original=SceneStream(manifest,8);factory=make_scene_stream(control,manifest,8)
    # The original X1 stream is checked through the full registered exposure.
    old_order=[]
    for cursor in range(0,40000,8):
        for rank in range(8):
            a=original.at(cursor,rank);b=factory.at(cursor,rank)
            assert a==b;old_order.append(a)
    checks.append('original_X1_all40000_scene_positions_exact')
    stream=make_scene_stream(candidate,manifest,8)
    counts=Counter();unique=set();rank_counts={str(r):Counter() for r in range(8)}
    sequence=[];safety_categories=Counter()
    scores={r['token']:r['metrics'] for r in stream.manifest['rows']}
    for cursor in range(0,40000,8):
        labels=[]
        for rank in range(8):
            token,position=stream.at(cursor,rank);label=position['stratum'];labels.append(label)
            counts[label]+=1;rank_counts[str(rank)][label]+=1;unique.add(token);sequence.append((token,position))
            key='+'.join(k for k,name in [('NC','no_at_fault_collisions'),('DAC','drivable_area_compliance'),
                      ('TTC','time_to_collision_within_bound')] if scores[token][name]<1) or 'SFT_single_seed_safe'
            safety_categories[key]+=1
        assert Counter(labels)==Counter(safe=6,unsafe=2)
    assert counts==Counter(safe=30000,unsafe=10000)
    assert unique==set(stream.tokens)
    assert all(v==Counter(safe=3750,unsafe=1250) for v in rank_counts.values())
    checks+=['every_collection_has_exact6plus2','all_hard_scenes_visited',
             'all_ranks_have_identical_stratum_exposure','fixed40000_budget_exact']
    cursors={0,8,16,39992}
    for label,quota in [('safe',6),('unsafe',2)]:
        boundary=len(stream.strata[label])//quota
        cursors.update(8*c for c in range(max(0,boundary-1),boundary+3))
    for cursor in sorted(cursors):
        restored=make_scene_stream(candidate,manifest,8)
        for rank in reversed(range(8)):
            assert restored.at(cursor,rank)==stream.at(cursor,rank)
    checks.append('cursor_only_recovery_and_rank_call_order_at_cohort_boundaries')
    assert torch.equal(state,torch.get_rng_state());checks.append('data_sampling_does_not_consume_actor_rng')
    pop=json.loads(Path(POPULATION).read_text())
    held=set(pop['dev_tokens'])|set(pop['navtest_tokens'])
    assert not unique&held
    assert not {pop['token_logs'][t] for t in unique}&{pop['token_logs'][t] for t in held}
    checks.append('TRAIN_only_log_isolation')
    fixture_tokens=stream.strata['safe'][:24]+stream.strata['unsafe'][:8]
    by_token={r['token']:r for r in stream.manifest['rows']}
    fixture=dict(status='QUALIFICATION_ONLY_NOT_FORMAL_HARD_POPULATION',
        population_identity=pop['identity_sha256'],tokens=fixture_tokens,
        rows=[copy.deepcopy(by_token[t]) for t in fixture_tokens])
    with tempfile.TemporaryDirectory() as temporary:
        path=Path(temporary)/'fixture.json';cfgpath=Path(temporary)/'recipe.yaml'
        def reject(name,changed):
            path.write_text(json.dumps(changed))
            try:FixedSafetyMixtureStream(path,8,qualification=True)
            except (ValueError,KeyError):checks.append(name)
            else:raise AssertionError('Accepted '+name)
        changed=copy.deepcopy(fixture);changed['rows'].append(changed['rows'][0]);reject('duplicate_score_row_rejected',changed)
        changed=copy.deepcopy(fixture);changed['rows'][0]['metrics']['no_at_fault_collisions']=float('nan');reject('unknown_label_rejected',changed)
        changed=copy.deepcopy(fixture);changed['tokens'][0]=pop['dev_tokens'][0];changed['rows'][0]['token']=changed['tokens'][0];reject('DEV_member_rejected',changed)
        changed=copy.deepcopy(fixture)
        for r in changed['rows']:
            for k in SAFETY25_SAMPLING['reference_metrics']:r['metrics'][k]=1.
        reject('empty_unsafe_cohort_rejected',changed)
        changed=copy.deepcopy(fixture);changed['rows'][0]['metrics']['no_at_fault_collisions']=.5
        path.write_text(json.dumps(changed));half=FixedSafetyMixtureStream(path,8,qualification=True)
        assert len(half.strata['unsafe'])==9;checks.append('NC_half_is_not_fully_safe')
        path.write_text(json.dumps(fixture))
        try:FixedSafetyMixtureStream(path,2,qualification=True)
        except ValueError:checks.append('wrong_world_rejected')
        else:raise AssertionError('world2 accepted')
        changes=[]
        for key,value in [('learning_rate',1e-4),('max_steps',100),('seed',43),('gradient_clipping_effective',1.)]:
            x=copy.deepcopy(candidate);x[key]=value;changes.append((key,x))
        x=copy.deepcopy(candidate);x['scene_sampling']['unsafe_groups']=3;changes.append(('unregistered_quota',x))
        x=copy.deepcopy(candidate);x['grpo']['sample']['sde_exclude_last_step']=True;changes.append(('terminal_ODE_not_combined',x))
        x=copy.deepcopy(candidate);x['grpo']['train']['kl_dim_weights'][0]=.2;changes.append(('X1_anchor_unchanged',x))
        x=copy.deepcopy(candidate);x['variant']='R_CANONICAL_X1';changes.append(('historical_X1_contract_not_relaxed',x))
        for name,x in changes:
            OmegaConf.save(OmegaConf.create(x),cfgpath)
            try:load_recipe(cfgpath)
            except ValueError:checks.append(name+'_rejected')
            else:raise AssertionError(name)
    legacy=json.loads(Path('/mnt/project/simwam-x-anchor-20261015/runs/R_CANONICAL_X1_seed42_v1/identity.json').read_text())
    assert scientific_identity(legacy)==legacy['scientific_identity']
    newer=copy.deepcopy(legacy);newer['data_stream_identity']=stream.identity
    assert scientific_identity(newer)!=scientific_identity(legacy)
    checks+=['legacy_scientific_identity_exact','new_data_contract_changes_scientific_identity']
    assert matched_control(SAFETY25_VARIANT)==('R_CANONICAL_X1',Path('/mnt/project/simwam-x-anchor-20261015'))
    for name in ('R_CANONICAL_X1','R_CANONICAL_LAST_ODE'):
        assert matched_control(name)==('R_CANONICAL',Path(ROOT))
    checks.append('new_X1_comparator_and_old_comparators_correct')
    paths=['starVLA/rl/simwam_recipe/vendor/action_math.py','starVLA/rl/simwam_recipe/trainer_adapter.py',
           'starVLA/rl/simwam_recipe/source_advantages.py','starVLA/rl/simwam_recipe/reward_readouts.py',
           'starVLA/rl/simwam_recipe/collection.py','scripts/analysis/simwam_recipe/evaluate_policy.py']
    unchanged={}
    for path in paths:
        old=subprocess.check_output(['git','show','8bc2d08fd17e7892e8834797926079cd3dfc5810:'+path])
        assert old==Path(path).read_bytes();unchanged[path]=file_sha(path)
    checks.append('actor_sampler_advantage_reward_collection_and_evaluation_unchanged')
    return dict(status='PASS',checks=checks,registered_config_identity=fingerprint(candidate),
        control_config_identity=fingerprint(control),data_stream_identity=stream.identity,
        expected_formal_exposures=dict(counts),expected_formal_safety_types=dict(safety_categories),
        expected_rank_exposures={k:dict(v) for k,v in rank_counts.items()},
        unique_expected_formal_scenes=len(unique),expected_formal_stream_sha256=fingerprint(sequence),
        original_X1_stream_sha256=fingerprint(old_order),cursor_boundaries_tested=sorted(cursors),
        unchanged_files=unchanged,new_rollout_candidates=0,new_scores=0,optimizer_updates=0,
        note='These are predetermined future data exposures, not observed formal training counts or driving effects.')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--manifest',default=ROOT+'/hard_manifest.json')
    p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError('Preserve previous qualification')
    result=qualify(a.manifest);out.parent.mkdir(parents=True,exist_ok=True);atomic_json(out,result)
    print(json.dumps(dict(status=result['status'],checks=len(result['checks']),
        expected_exposures=result['expected_formal_exposures'],expected_scenes=result['unique_expected_formal_scenes'])))


if __name__=='__main__':main()
