"""Meaningful population-leakage, stage-integrity and cursor-recovery checks."""
import json
import pickle
import random
from pathlib import Path

import pytest
from PIL import Image

from starVLA.rl.qwen_world_research_data import manifest as mm
from starVLA.rl.qwen_world_research_data.data import ResearchSceneData, SceneStream
from starVLA.rl.qwen_world_research_data.adaptive import priority_weights, prepare_stage, stage_census
from starVLA.rl.qwen_world_research_data.panels import prepare_panels
from starVLA.rl.qwen_world_research_data.journal_analysis import analyze_journals,paired_regressions

@pytest.fixture
def population(tmp_path,monkeypatch):
    image=tmp_path/'current.jpg';Image.new('RGB',(5,4),(10,20,30)).save(image)
    tokens=[f't{i:02}' for i in range(24)]
    logs={t:f'2026.01.{i//4+1:02}.00.00.00_veh-1_00000_00100' for i,t in enumerate(tokens)}
    logs['navtest']='2026.02.01.00.00.00_veh-1_00000_00100'
    pop_path=tmp_path/'population.json'
    mm.atomic_json(pop_path,dict(train_tokens=tokens[:20],dev_tokens=tokens[20:],navtest_tokens=['navtest'],token_logs=logs))
    monkeypatch.setattr(mm,'POPULATION',pop_path)
    data=tmp_path/'data';data.mkdir()
    rows={t:dict(token=t,log=logs[t],images={c:str(image) for c in ['cam_l0','cam_f0','cam_r0']},
                 proprio=[1.,0.,0.,0.,0.,1.,0.,0.],navigation='go straight',requested_GT=[[1.,0.,0.]]*8,
                 teacher_frames='must never reach actor',objects='must never reach actor') for t in tokens}
    with (data/'records.pkl').open('wb') as f:pickle.dump(rows,f)
    body=dict(schema='world_rl_research_current_observations_v1',cohort='NAVTRAIN',tokens=tokens,count=len(tokens),
              source_population_path=str(pop_path),population_sha256=mm.file_sha(pop_path),
              records_sha256=mm.file_sha(data/'records.pkl'),old_DEV_tokens=tokens[20:],token_logs={t:logs[t] for t in tokens})
    mm.atomic_json(data/'manifest.json',body)
    mm.atomic_json(data/'COMPLETE',dict(status='COMPLETE',manifest_sha256=mm.file_sha(data/'manifest.json')))
    return dict(root=tmp_path,data=data,tokens=tokens,pop_path=pop_path,rows=rows)

def make_stream(p,mode='uniform',tokens=None):
    path=p['root']/f'{mode}.json'
    mm.write_stream_manifest(path,data_root=p['data'],tokens=tokens or p['tokens'],name='test',mode=mode,
                             stage_exposures=16 if mode=='adaptive' else None,
                             stage_registry=p['root']/'stages' if mode=='adaptive' else None)
    return path

def write_bank(p,name,scores,cohort='NAVTRAIN',inference=None):
    path=p['root']/f'{name}.json'
    m=mm.read_sealed(p['root']/'adaptive.json')
    mm.atomic_json(path,mm.sealed(dict(schema='world_rl_research_diagnostic_bank_v1',cohort=cohort,
        data_manifest_sha256=m['data_manifest_sha256'],score_units='0..1',policy_sha256=name,
        inference=inference or dict(sampler='ODE',steps=10,seed=2101),records={t:dict(score=s) for t,s in scores.items()})))
    return path

def test_current_observation_never_exposes_future_or_annotations(population):
    data=ResearchSceneData(population['data'])
    observation=data.observation('t00')
    assert set(observation)=={'images','proprio','navigation'}
    assert set(observation['images'])=={'cam_l0','cam_f0','cam_r0'}
    assert all(x.size==(5,4) for x in observation['images'].values())
    assert data.target('t00').shape==(8,3)

def test_uniform_stream_matches_source_order_and_restores_without_rng_state(population):
    from starVLA.rl.simwam_recipe.vendor.samplers import ResumableEpochSampler
    path=make_stream(population,tokens=population['tokens'][:21])
    stream=SceneStream(path,8)
    for epoch in [0,1,5]:
        source=ResumableEpochSampler(stream.tokens,42,1,8);source.set_epoch(epoch)
        expected=list(source);expected=(expected*2)[:24]
        observed=[stream.at(epoch*24+cursor,rank)[0] for cursor in [0,8,16] for rank in range(8)]
        assert observed==[stream.tokens[i] for i in expected]
    random.seed(999);random.random()
    restored=SceneStream(path,8)
    assert [stream.at(24*5+16,r) for r in range(8)]==[restored.at(24*5+16,r) for r in range(8)]
    assert stream.at(16,7)[1]['padding']

def test_tampered_data_or_navtest_tokens_are_rejected(population):
    path=make_stream(population)
    body=mm.read_sealed(path);body.pop('sha256');body['tokens'][0]='navtest'
    mm.atomic_json(path,mm.sealed(body))
    with pytest.raises(ValueError):SceneStream(path,8)
    path=make_stream(population)
    with (population['data']/'manifest.json').open('a') as f:f.write(' ')
    with pytest.raises(ValueError,match='identity changed'):SceneStream(path,8)

def test_priority_covers_unknown_and_both_low_and_regressed_scenes():
    tokens=['unknown','low','regressed','good']
    weights,diag=priority_weights(tokens,dict(low=.1,regressed=.5,good=.95),dict(low=.1,regressed=.99,good=.95))
    w=dict(zip(tokens,weights))
    assert w['unknown']>0 and w['low']>w['good'] and w['regressed']>w['good']
    assert diag['unknown_scenes']==1 and diag['observed_regressed_scenes']==1
    assert sum(weights)==pytest.approx(1.)
    with pytest.raises(ValueError):priority_weights(tokens,{'navtest':0.}, {})

def test_adaptive_stage_is_immutable_and_resumes_exactly(population):
    path=make_stream(population,mode='adaptive')
    initial=prepare_stage(path,0)
    assert initial['unique_scenes']==16
    stream=SceneStream(path,8)
    assert [stream.at(cursor,r)[0] for cursor in [0,8] for r in range(8)]==initial['ordered_tokens']
    with pytest.raises(FileNotFoundError):stream.at(16,0)
    current=write_bank(population,'current',{'t00':0.,'t01':.4,'t02':.9})
    baseline=write_bank(population,'baseline',{'t00':.2,'t01':.9,'t02':.9})
    stage=prepare_stage(path,1,current_bank=current,baseline_bank=baseline)
    assert stage['start_cursor']==16 and stage['unique_scenes']>=8
    assert all(len(set(stage['ordered_tokens'][i:i+8]))==8 for i in (0,8))
    restored=SceneStream(path,8)
    assert [stream.at(16,r) for r in range(8)]==[restored.at(16,r) for r in range(8)]
    with pytest.raises(FileExistsError):prepare_stage(path,1,current_bank=current,baseline_bank=baseline)

def test_small_diagnostic_panel_can_be_repeated_across_batches():
    tokens=[f't{i}' for i in range(1000)]
    scores={t:0. for t in tokens[:16]};baseline={t:.9 for t in tokens[:16]}
    weights,_=priority_weights(tokens,scores,baseline)
    census=stage_census(tokens,weights,1024,42,adaptive=True)
    # Focus can meaningfully exceed stage-wide no-replacement's 16/1024 cap.
    assert sum(t in scores for t in census)>256
    assert all(len(set(census[i:i+8]))==8 for i in range(0,1024,8))
    assert census==stage_census(tokens,weights,1024,42,adaptive=True)

def test_test_banks_and_mismatched_protocols_cannot_drive_selection(population):
    path=make_stream(population,mode='adaptive')
    baseline=write_bank(population,'baseline',{'t00':.8})
    bad=write_bank(population,'test_bank',{'t00':.2},cohort='NAVTEST')
    with pytest.raises(ValueError,match='legal TRAIN'):prepare_stage(path,0,current_bank=bad,baseline_bank=baseline)
    bad=write_bank(population,'foreign_token',{'navtest':.2})
    with pytest.raises(ValueError,match='held-out'):prepare_stage(path,0,current_bank=bad,baseline_bank=baseline)
    bad=write_bank(population,'different_seed',{'t00':.2},inference=dict(sampler='ODE',steps=10,seed=999))
    with pytest.raises(ValueError,match='same sampler'):prepare_stage(path,0,current_bank=bad,baseline_bank=baseline)

def test_panels_are_nested_score_independent_and_cover_physical_logs(population):
    result=prepare_panels(population['data'],population['root']/'panels',sizes=(8,16))
    small=mm.read_sealed(result[8]['path']);large=mm.read_sealed(result[16]['path'])
    assert set(small['tokens'])<set(large['tokens'])
    assert small['physical_logs']==6 and small['NAVTEST_token_overlap']==0
    assert small['train_feedback_allowed'] is True
    assert result==prepare_panels(population['data'],population['root']/'panels',sizes=(8,16))

def test_coverage_counts_fresh_groups_once_and_excludes_partial_active_tail(population):
    path=make_stream(population)
    run=population['root']/'run';run.mkdir()
    identity=mm.sealed(dict(world=2,purpose='formal',data_stream_identity=dict(manifest_sha256=mm.file_sha(path))))
    mm.atomic_json(run/'identity.json',identity)
    for rank in range(2):
        with (run/f'metrics_rank{rank}.jsonl').open('w') as f:
            for update in range(1,9):
                group=(update-1)//4+1
                token=f't{(group-1)*2+rank:02}'
                row=dict(updates=update,inner_completed=(update-1)%4+1,collection=group,token=token,
                         run=identity['sha256'],loss=.1,gradient_norm=.2,exploration=dict(
                             canonical_random_mean=.7,safe_fraction=.5,all_zero=False,zero_PG=False,NC=1.,DAC=.5,TTC=1.))
                f.write(json.dumps(row)+'\n')
            if rank==1:f.write('{"updates":9')
    result=analyze_journals(run,path,world_size=2)
    assert result['common_optimizer_update']==8
    assert result['fresh_global_collections']==2 and result['fresh_scene_groups']==4
    assert result['new_candidates']==32 and result['unique_seen_scenes']==4
    assert result['file_snapshots'][1]['partial_tail_excluded'] is True
    assert result['paired_ODE_regressions']['status']=='NOT_MEASURED'
    assert result['stochastic_means_are_not_paired_ODE_regressions'] is True

def test_paired_ODE_regression_requires_identical_census_and_inference(population):
    path=make_stream(population,'adaptive')
    current=write_bank(population,'current',dict(t00=.5,t01=.9))
    baseline=write_bank(population,'baseline',dict(t00=.9,t01=.5))
    result=paired_regressions(path,current,baseline)
    assert result['regressed_scenes']==1 and result['improved_scenes']==1
    assert result['mean_delta_PDMS']==pytest.approx(0)
    assert result['G_plus_PDMS']==pytest.approx(20)
    assert result['G_minus_PDMS']==pytest.approx(-20)
    wrong=write_bank(population,'wrong',dict(t00=.9,t01=.5),inference=dict(sampler='SDE',steps=10,seed=2101))
    with pytest.raises(ValueError,match='identical scene census'):paired_regressions(path,current,wrong)
