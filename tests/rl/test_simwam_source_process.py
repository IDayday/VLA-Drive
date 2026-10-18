import copy
import json
from pathlib import Path
import pytest
import torch
import yaml
from starVLA.rl.fm_transfer.source_process import expected_config, load_process_recipe, authenticate_manifest, SourceTokenStream
from starVLA.rl.simwam_recipe.contracts import load_recipe, fingerprint
from starVLA.rl.simwam_recipe.vendor.samplers import ResumableEpochSampler


def test_source_process_retains_every_actor_loss_optimizer_field():
    base=load_recipe('configs/flow_rl/simwam_source_ddp.yaml')
    for mode in ('UPSTREAM_DATA','OWN_DATA_CONTROL'):
        cfg=expected_config(mode);cfg.pop('source_process')
        assert cfg.pop('max_steps')==170000
        assert cfg.pop('evaluate_updates')==[0,5000,20000,50000,100000,170000]
        original=dict(base);original.pop('max_steps');original.pop('evaluate_updates')
        assert cfg==original
    a,b=[expected_config(mode) for mode in ('UPSTREAM_DATA','OWN_DATA_CONTROL')]
    a['source_process']['data_mode']=b['source_process']['data_mode']
    assert a==b


@pytest.mark.parametrize('field,value',[('learning_rate',1e-4),('max_steps',170004),('seed',43),('gradient_clipping_effective',1.)])
def test_mutated_effective_recipe_rejected(tmp_path,field,value):
    cfg=expected_config('UPSTREAM_DATA');cfg[field]=value
    path=tmp_path/'changed.yaml';path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):load_process_recipe(path)


def sealed(tokens,pop):
    x=dict(status='COMPLETE_UPSTREAM_DATA_TRAIN_ONLY',population_identity=pop['identity_sha256'],tokens=tokens,
        upstream_commit='1433ee7e06a62bd8b89da74455df0d82bda7b726',retained_scenes=len(tokens),
        source_scenes=len(tokens)+1,excluded_DEV_scenes=1,source_Navtest_overlap=0,missing_source_scenes=0,
        source_yaml_sha256='source',source_loader_sha256='loader')
    x['sha256']=fingerprint(x);return x


def test_source_membership_checks_DEV_and_population_hash(tmp_path):
    pop=dict(train_tokens=['a','b'],dev_tokens=['c'],navtest_tokens=['d'],identity_sha256='split')
    path=tmp_path/'manifest.json'
    for tokens in (['a','c'],['a','d'],['a','a']):
        path.write_text(json.dumps(sealed(tokens,pop)))
        with pytest.raises(ValueError):authenticate_manifest(path,pop)
    valid=sealed(['a','b'],pop);path.write_text(json.dumps(valid))
    assert authenticate_manifest(path,pop)[0]==valid
    valid['tokens'].reverse();path.write_text(json.dumps(valid))
    with pytest.raises(ValueError):authenticate_manifest(path,pop)


def test_source_epoch_permutation_padding_and_resume(monkeypatch,tmp_path):
    from starVLA.rl.fm_transfer import source_process as module
    tokens=[str(i) for i in range(11)]
    pop=dict(train_tokens=tokens,dev_tokens=['dev'],navtest_tokens=['test'],token_logs={t:'log' for t in tokens},identity_sha256='p')
    payload=sealed(tokens,pop);path=tmp_path/'manifest.json';path.write_text(json.dumps(payload))
    monkeypatch.setattr(module,'authenticate_manifest',lambda _: (payload,pop))
    stream=SourceTokenStream(path,8);reference=ResumableEpochSampler(tokens,seed=42,batch_size=1,num_processes=8)
    rng=torch.get_rng_state().clone()
    for epoch in range(3):
        reference.set_epoch(epoch);order=list(reference);padded=(order*2)[:16]
        actual=[stream.at(epoch*16+offset,rank)[0] for offset in (0,8) for rank in range(8)]
        assert actual==[tokens[i] for i in padded]
        restored=SourceTokenStream(path,8)
        assert [stream.at(epoch*16+8,r) for r in range(8)]==[restored.at(epoch*16+8,r) for r in range(8)]
    assert torch.equal(rng,torch.get_rng_state())


def test_170k_budget_is_optimizer_updates_not_fresh_rollouts():
    p=expected_config('UPSTREAM_DATA')['source_process']
    assert p['fresh_collections']*4==170000
    assert p['scene_exposures']==p['fresh_collections']*8
    assert p['new_candidates']==p['scene_exposures']*8


def test_fixed_milestone_queue_has_two_grids_in_each_unchanged_worker(tmp_path):
    from scripts.analysis.fm_transfer.source_process_controller import ProcessEvaluations
    jobs=list(ProcessEvaluations.jobs(tmp_path))
    assert len(jobs)==20
    assert len({j['output'] for j in jobs})==20
    assert {j['update'] for j in jobs}=={5000,20000,50000,100000,170000}
    assert {j['cohort'] for j in jobs}=={'DEV','TRAIN_DIAGNOSTIC'}
    assert all('B_SOURCE_ACTION_DIT_SFT_update100000' in ProcessEvaluations.reference_for(tmp_path,j) for j in jobs)


def test_navtest_lock_first_publication_creates_nested_parent(tmp_path):
    from scripts.analysis.fm_transfer.source_process_navtest import publish_lock
    path=tmp_path/'navtest/update020000/final_navtest/LOCK.json'
    value=dict(observed_update=20000,status='LOCKED_BEFORE_FINAL_NAVTEST')
    publish_lock(path,value)
    assert json.loads(path.read_text())==value
    assert list(path.parent.iterdir())==[path]


def test_navtest_lock_republication_preserves_existing_selection(tmp_path):
    from scripts.analysis.fm_transfer.source_process_navtest import publish_lock
    path=tmp_path/'LOCK.json';path.write_text('{"original": true}\n')
    original=path.read_bytes()
    with pytest.raises(FileExistsError):publish_lock(path,dict(original=False))
    assert path.read_bytes()==original
