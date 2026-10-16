import copy
import json
from pathlib import Path

import pytest

from scripts.analysis.fm_transfer import final_lock
from starVLA.rl.simwam_recipe.contracts import file_sha, fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json


def sealed(path, value):
    value = dict(value)
    value['sha256'] = fingerprint(value)
    atomic_json(path,value)
    return value


def summary():
    return dict(status='TRAINING_COMPLETE',run_identity='run',matched_completed_updates=20000,
        observed_updates_by_rank=[20000]*8,cost=dict(checkpoint_updates=20000),
        cumulative=dict(fresh_global_rollouts=5000,scene_exposures=40000,new_candidates=320000))


def test_budget_is_fresh_samples_not_four_inner_reuse():
    value = summary()
    final_lock.validate_rl_summary(value,dict(sha256='run'))
    value['cumulative']['new_candidates'] *= 4
    with pytest.raises(ValueError):
        final_lock.validate_rl_summary(value,dict(sha256='run'))


@pytest.mark.parametrize('key,value',[
    ('status','INTERIM_TRAINING_ONLY'),('matched_completed_updates',19996),
    ('observed_updates_by_rank',[20000]*7+[19999]),('run_identity','other_run')])
def test_summary_refuses_partial_or_foreign_endpoint(key,value):
    result = summary();result[key] = value
    with pytest.raises(ValueError):
        final_lock.validate_rl_summary(result,dict(sha256='run'))


def test_waiting_gate_cannot_touch_navtest_assets_or_publish_lock(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('No Navtest asset or score use before endpoint gate')
    monkeypatch.setattr(final_lock,'prepare_assets',forbidden)
    result = final_lock.readiness(tmp_path)
    assert result['status'] == 'WAITING_FOR_BOTH_ENDPOINTS_AND_DEV'
    assert result['Navtest_opened'] is False and result['new_candidates'] == 0
    assert any('B_SOURCE_ACTION_DIT_SFT_seed42_v1/checkpoints/update100000' in p for p in result['missing'])
    assert any('A_DDP_TIME_R_SOURCE_seed42_v1/checkpoints/update020000_inner0' in p for p in result['missing'])
    with pytest.raises(ValueError,match='Navtest remains closed'):
        final_lock.make_lock(tmp_path,tmp_path/'snapshot',tmp_path/'LOCK.json')
    assert not (tmp_path/'LOCK.json').exists()


def native(tmp_path):
    checkpoint = tmp_path/'update020000_inner0';checkpoint.mkdir()
    identity = sealed(checkpoint/'identity.json',dict(schema='test',execution_identity=dict(sha256='execution')))
    for rank in range(8):
        (checkpoint/f'rank{rank}.pt').write_bytes(b'native-state')
    (checkpoint/'optimizer.pt').write_bytes(b'optimizer-state')
    row = dict(phase='COMPLETE',updates=20000,collections=5000,scene_cursor=40000,
               next_inner=0,run=identity['sha256'],pending=False)
    atomic_json(checkpoint/'transaction.json',dict(ranks=[row]*8,execution_identity=identity['execution_identity']))
    atomic_json(checkpoint/'files.json',{p.name:file_sha(p) for p in checkpoint.iterdir()})
    seal = dict(phase='COMPLETE',world=8,updates=20000,next_inner=0,
        files_sha256=file_sha(checkpoint/'files.json'))
    atomic_json(checkpoint/'COMPLETE',seal)
    return checkpoint,identity,seal


@pytest.mark.parametrize('field,value',[
    ('phase','COLLECTING'),('phase','READY'),('next_inner',2),('updates',15000),('world',2)])
def test_complete_filename_does_not_override_native_boundary(tmp_path,field,value):
    checkpoint,identity,seal = native(tmp_path)
    final_lock.checkpoint_files(checkpoint,identity,20000,rl=True)
    seal[field] = value;atomic_json(checkpoint/'COMPLETE',seal)
    with pytest.raises(ValueError):
        final_lock.checkpoint_files(checkpoint,identity,20000,rl=True)


def test_changed_native_payload_is_not_a_valid_endpoint(tmp_path):
    checkpoint,identity,_ = native(tmp_path)
    (checkpoint/'rank0.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='payload'):
        final_lock.checkpoint_files(checkpoint,identity,20000,rl=True)


def test_missing_rank_or_divergent_transaction_rejected(tmp_path):
    checkpoint,identity,seal = native(tmp_path)
    transaction = json.loads((checkpoint/'transaction.json').read_text())
    transaction['ranks'][7]['updates'] = 19999
    atomic_json(checkpoint/'transaction.json',transaction)
    files = json.loads((checkpoint/'files.json').read_text())
    files['transaction.json'] = file_sha(checkpoint/'transaction.json')
    atomic_json(checkpoint/'files.json',files)
    seal['files_sha256'] = file_sha(checkpoint/'files.json');atomic_json(checkpoint/'COMPLETE',seal)
    with pytest.raises(ValueError):
        final_lock.checkpoint_files(checkpoint,identity,20000,rl=True)
    del files['rank7.pt'];atomic_json(checkpoint/'files.json',files)
    seal['files_sha256'] = file_sha(checkpoint/'files.json');atomic_json(checkpoint/'COMPLETE',seal)
    with pytest.raises(ValueError):
        final_lock.checkpoint_files(checkpoint,identity,20000,rl=True)


def test_sft_uses_actual_end_update_and_full_exposure_fields(tmp_path):
    checkpoint = tmp_path/'checkpoints/update020000'
    for rank in range(8):
        atomic_json(tmp_path/f'result_rank{rank}.json',dict(status='COMPLETE',end_update=20000,
            rank=rank,identity='own-sft',cursor=640000,qualification=False,checkpoint=str(checkpoint)))
    final_lock.validate_sft_results(tmp_path,dict(sha256='own-sft'),20000,checkpoint)
    value = json.loads((tmp_path/'result_rank7.json').read_text());value['end_update'] = 19999
    atomic_json(tmp_path/'result_rank7.json',value)
    with pytest.raises(ValueError):
        final_lock.validate_sft_results(tmp_path,dict(sha256='own-sft'),20000,checkpoint)


def test_own_sft_and_merged_export_are_bound_to_native_endpoint(tmp_path):
    export = tmp_path/'export';export.mkdir()
    (export/'pytorch_model.pt').write_bytes(b'merged-head')
    (export/'config.yaml').write_text('model: ddp\n')
    checkpoint = tmp_path/'update020000_inner0'
    own_sft = dict(identity_sha256='new-SFT',config=dict(architecture='ddp'))
    info = sealed(export/'export.json',dict(schema='fm_transfer_original_entry_export_v1',
        qualification=False,frozen_upstream_exact=True,stage='RL',updates=20000,
        checkpoint=str(checkpoint),run_identity='run',new_SFT=own_sft,architecture='ddp',
        weight_sha256=file_sha(export/'pytorch_model.pt'),config_sha256=file_sha(export/'config.yaml')))
    atomic_json(export/'COMPLETE',info)
    result = final_lock.validate_export(export,checkpoint,dict(sha256='run'),'RL',20000,own_sft)
    assert result['weight_sha256'] == info['weight_sha256']
    foreign = copy.deepcopy(own_sft);foreign['identity_sha256'] = 'old-F-SFT'
    with pytest.raises(ValueError):
        final_lock.validate_export(export,checkpoint,dict(sha256='run'),'RL',20000,foreign)
    with pytest.raises(ValueError):
        final_lock.validate_export(export,checkpoint,dict(sha256='run'),'SFT',20000,own_sft)
    (export/'pytorch_model.pt').write_bytes(b'other-weights')
    with pytest.raises(ValueError):
        final_lock.validate_export(export,checkpoint,dict(sha256='run'),'RL',20000,own_sft)


def test_positive_final_lock_has_four_own_endpoints_and_never_selects_a_grid(tmp_path,monkeypatch):
    root = tmp_path/'root';root.mkdir()
    snapshot = tmp_path/'snapshot';snapshot.mkdir()
    monkeypatch.setattr(final_lock,'EVALUATION_SOURCE',snapshot)
    atomic_json(snapshot/'SOURCE_COMPLETE.json',dict(git_archive_extraction_complete=True))
    for module in ['simwam_recipe','fm_transfer']:
        path = snapshot/'scripts/analysis'/module/'evaluate_policy.py'
        path.parent.mkdir(parents=True);path.write_text('# existing qualified worker\n')
    for path in final_lock.required_files(root):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    recipe = dict(variant='R_SOURCE',max_steps=20000)
    monkeypatch.setattr(final_lock,'load_recipe',lambda path:recipe)
    monkeypatch.setattr(final_lock,'validate_milestones',lambda root,**kwargs:dict(verified='milestone-evidence'))
    population = tmp_path/'population.json';atomic_json(population,dict(identity_sha256='population'))
    monkeypatch.setattr(final_lock,'POPULATION',str(population))
    asset_calls = []
    def assets(path):
        asset_calls.append(path)
        return dict(sha256='assets',data_root='inputs',navtest_datalist='tokens',v1_cache='canonical-cache')
    monkeypatch.setattr(final_lock,'prepare_assets',assets)
    for variant,cfg in final_lock.CONFIGS.items():
        sft_identity = None;own_sft = None
        for stage,update in [('SFT',cfg['max_updates']),('RL',20000)]:
            family = 'SFT' if stage=='SFT' else 'R_SOURCE'
            run = root/'runs'/f'{variant}_{family}_seed42_v1'
            checkpoint = run/'checkpoints'/(f'update{update:06d}'+('_inner0' if stage=='RL' else ''))
            if stage=='SFT':
                identity = sealed(run/'identity.json',dict(qualification=False,config=final_lock.sft_config(variant)))
            else:
                identity = sealed(run/'identity.json',dict(purpose='formal',world=8,
                    recipe=dict(**recipe,supervised_transfer=dict(SFT_identity=own_sft)),
                    execution_identity=dict(sha256='execution'),source_files=dict(trainer='same-locked-code')))
            atomic_json(checkpoint/'identity.json',identity)
            (checkpoint/'optimizer.pt').write_bytes(b'optimizer')
            for rank in range(8):
                (checkpoint/f'rank{rank}.pt').write_bytes(b'rank-state')
                result = dict(status='COMPLETE',rank=rank,qualification=False,checkpoint=str(checkpoint))
                if stage=='SFT':result.update(end_update=update,cursor=update*32,identity=identity['sha256'])
                else:result.update(updates=update,collections=5000,raw_scene_exposures=40000,new_candidates=320000)
                atomic_json(run/f'result_rank{rank}.json',result)
            if stage=='SFT':
                (checkpoint/'model.pt').write_bytes(variant.encode())
                own_sft = dict(checkpoint=str(checkpoint),model_sha256=file_sha(checkpoint/'model.pt'),
                    identity_sha256=identity['sha256'],config=identity['config'],updates=update,qualification=False)
            else:
                row = dict(phase='COMPLETE',updates=update,collections=5000,scene_cursor=40000,
                    next_inner=0,run=identity['sha256'],pending=False)
                atomic_json(checkpoint/'transaction.json',dict(ranks=[row]*8,execution_identity=identity['execution_identity']))
                result = summary();result['run_identity'] = identity['sha256']
                atomic_json(root/'analysis'/f'{variant}_RL_complete_training.json',result)
            files = {p.name:file_sha(p) for p in checkpoint.iterdir() if p.name!='COMPLETE'}
            atomic_json(checkpoint/'files.json',files)
            atomic_json(checkpoint/'COMPLETE',dict(phase='READY' if stage=='SFT' else 'COMPLETE',world=8,
                updates=update,next_inner=0,files_sha256=file_sha(checkpoint/'files.json')))
            export = root/'exports'/variant/f'{stage}_update{update:06d}'
            (export/'pytorch_model.pt').write_bytes((variant+stage).encode())
            (export/'config.yaml').write_text('config: unchanged\n')
            info = sealed(export/'export.json',dict(schema='fm_transfer_original_entry_export_v1',
                stage=stage,updates=update,architecture=cfg['architecture'],new_SFT=own_sft,qualification=False,
                frozen_upstream_exact=True,checkpoint=str(checkpoint),run_identity=identity['sha256'],
                weight_sha256=file_sha(export/'pytorch_model.pt'),config_sha256=file_sha(export/'config.yaml')))
            atomic_json(export/'COMPLETE',info)
    lock = final_lock.make_lock(root,snapshot,tmp_path/'LOCK.json')
    assert set(lock['models']) == {'A_DDP_TIME_SFT','A_DDP_TIME_RL','B_SOURCE_ACTION_DIT_SFT','B_SOURCE_ACTION_DIT_RL'}
    assert lock['inference_seeds']==[2101] and lock['modes']==['E_NATIVE','E_SHIFT5']
    assert lock['Navtest_scores_read_for_selection'] is False and len(asset_calls)==1
    for variant in final_lock.CONFIGS:
        assert lock['models'][variant+'_SFT']['new_SFT']==lock['models'][variant+'_RL']['new_SFT']
    assert final_lock.authenticated(tmp_path/'LOCK.json')==lock
    with pytest.raises(FileExistsError):
        final_lock.make_lock(root,snapshot,tmp_path/'LOCK.json')
