"""Synthetic evidence exercises the actual gate; it grants no real GPU release."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from omegaconf import OmegaConf
from starVLA.rl.flow_grpo.config import config_hash
from starVLA.rl.flow_grpo.data import keys_for_positions
from starVLA.rl.flow_grpo.full_epoch import epoch_budget
from starVLA.rl.flow_grpo.acceptance import enforce_training_budget
from starVLA.rl.flow_grpo.loading import file_sha
from tests.flow_grpo.test_research_budget import fixture as pilot_fixture


def test_one_epoch_covers_every_scene_and_only_pads_last_batch():
    budget = epoch_budget(103288)
    assert budget['optimizer_updates'] == 12912
    assert budget['behavior_batches'] == 6456 and budget['padding_repeats'] == 8
    assert budget['candidate_trajectories'] == 1652736
    # The real SceneStream key allocator, over two shard layouts.
    tokens = [str(i) for i in range(23)]
    expected = list(keys_for_positions(tokens,42,list(range(32))))
    shard = {rank:list(keys_for_positions(tokens,42,list(range(rank,32,8)))) for rank in range(8)}
    assert [shard[i%8][i//8] for i in range(32)] == expected
    assert {key[0] for key in expected[:23]} == set(tokens)
    assert len({key[0] for key in expected[:23]}) == 23


def fixture(tmp_path, monkeypatch):
    _, _, record, rows, put = pilot_fixture(tmp_path,monkeypatch)
    cfg = OmegaConf.to_container(OmegaConf.load('configs/flow_grpo/frozen_full_navtrain_epoch1.yaml'))
    cfg['sampling']['num_steps'] = 10
    cfg['runtime']['epoch_evidence'] = str(tmp_path/'record.json')
    tokens = [str(i) for i in range(103288)]
    # Isolate corpus I/O only; all training-budget and proof validation is real.
    monkeypatch.setattr('starVLA.rl.flow_grpo.full_epoch.split_tokens',lambda c:(tokens,[]))
    monkeypatch.setattr('starVLA.rl.flow_grpo.full_epoch.OmegaConf.load',lambda p:SimpleNamespace(tokens=tokens))
    cfg['paths']['split_manifest'] = str(tmp_path/'split.json')
    Path(cfg['paths']['split_manifest']).write_text(json.dumps(tokens))
    ctx={'executable_sha256':'synthetic-test-only','checkpoint_sha256':cfg['checkpoint_contract']['sha256'],
         'world_size':8,'resume_identity':{'synthetic':True},'config_sha256':config_hash(cfg)}
    for row in rows:
        row['scheduler_completed_updates'] = row['update']
        row['stability'] = {'controller':{'updates':row['update']}}
        row['advantage_nonzero_group_fraction'] = .5
    (tmp_path/'pilot').mkdir();(tmp_path/'resume').mkdir()
    record.update(status='AUTHORIZED_FULL_DATA_EXPERIMENT',budget=epoch_budget(103288),context=ctx,
                  split_sha256=file_sha(cfg['paths']['split_manifest']),
                  pilot_config=put('config.json',cfg),
                  pilot_context=put('pilot/execution_context.json',ctx),
                  resume_context=put('resume/execution_context.json',ctx),
                  pilot_training=put('training.jsonl',rows,True),
                  resume_control=put('resume_control.json',{'status':'PASS','exit_codes':[0]}))
    names=['scheduler.bin','custom_checkpoint_0.pkl','pytorch_model/mp_rank_00_model_states.pt']
    names += [f'rank_{r}.pt' for r in range(8)]
    names += [f'random_states_{r}.pkl' for r in range(8)]
    names += [f'pytorch_model/bf16_zero_pp_rank_{r}_mp_rank_00_optim_states.pt' for r in range(8)]
    comparison={'status':'PASS','files':{n:{'status':'PASS','values':{'test_only':{'allclose':True}}} for n in names},
                'boundaries':{'continuous':str(tmp_path/'pilot/checkpoints/update_000002'),
                              'resumed':str(tmp_path/'resume/checkpoints/update_000002'),
                              'config_hash':config_hash(cfg),'trainer_state':{'update':2,'world_size':8}}}
    record['exact_resume']=put('comparison.json',comparison)
    return cfg,ctx,record,rows,put


def save(cfg,record):Path(cfg['runtime']['epoch_evidence']).write_text(json.dumps(record))


def test_gate_accepts_matching_evidence_and_preserves_old_diagnostic_limit(tmp_path,monkeypatch):
    cfg,ctx,record,_,_=fixture(tmp_path,monkeypatch);save(cfg,record)
    enforce_training_budget(cfg,ctx)
    cfg['runtime']['max_updates']=12913
    with pytest.raises(ValueError,match='budget'):enforce_training_budget(cfg,ctx)
    cfg['runtime'].update(run_mode='diagnostic',max_updates=9)
    with pytest.raises(ValueError,match='8 optimizer'):enforce_training_budget(cfg,ctx)


@pytest.mark.parametrize('change',['cpu','fail','world','chunk','u','source','all_equal','resume_fail',
                                  'schedule','missing_kl_state','changed_split','missing_evidence'])
def test_full_epoch_does_not_bypass_gpu_proofs(tmp_path,monkeypatch,change):
    cfg,ctx,record,rows,put=fixture(tmp_path,monkeypatch)
    if change=='cpu':rows[0]['ranks'][0]['device']['type']='cpu'
    if change=='fail':record['pilot_control']=put('control.json',{'status':'FAIL','exit_codes':[1]})
    if change=='world':monkeypatch.setenv('WORLD_SIZE','16')
    if change=='chunk':cfg['sampling']['candidate_chunk_size']=2
    if change=='u':cfg['checkpoint_contract']['variant']='unfrozen_visual'
    if change=='source':ctx=copy.deepcopy(ctx);ctx['executable_sha256']='changed'
    if change=='all_equal':rows[0]['advantage_nonzero_group_fraction']=0
    if change=='schedule':rows[1]['scheduler_completed_updates']=16
    if change=='missing_kl_state':rows[1].pop('stability')
    if change=='resume_fail':record['exact_resume']=put('comparison.json',{'status':'FAIL','files':{}})
    if change=='changed_split':Path(cfg['paths']['split_manifest']).write_text('[]')
    record['pilot_training']=put('training.jsonl',rows,True)
    if change!='missing_evidence':save(cfg,record)
    with pytest.raises(ValueError):enforce_training_budget(cfg,ctx)


def test_world16_requires_its_own_complete_gpu_and_resume_evidence(tmp_path,monkeypatch):
    cfg,ctx,record,rows,put=fixture(tmp_path,monkeypatch)
    monkeypatch.setenv('WORLD_SIZE','16');cfg['runtime']['accumulation_steps']=1
    save(cfg,record)
    # Old world8 artifacts never release the expanded topology.
    with pytest.raises(ValueError):enforce_training_budget(cfg,ctx)
    ctx['world_size']=16;ctx['config_sha256']=config_hash(cfg)
    record.update(context=ctx,pilot_context=put('pilot/execution_context.json',ctx),
                  resume_context=put('resume/execution_context.json',ctx),
                  pilot_config=put('config.json',cfg),
                  pilot_control=put('control16.json',{'status':'PASS','exit_codes':[0,0]}),
                  resume_control=put('resume16.json',{'status':'PASS','exit_codes':[0,0]}))
    for row in rows:
        row['ranks'] += [dict(copy.deepcopy(r),rank=r['rank']+8) for r in row['ranks']]
    record['pilot_training']=put('training16.jsonl',rows,True)
    record['pilot_dtype']=put('dtype16.json',{'ranks':rows[-1]['ranks']})
    for r in range(8,16):
        proof=json.loads(Path(record['pilot_immutable'][r-8]['path']).read_text())
        record['pilot_immutable'].append(put(f'immutable_update000002_rank{r}.json',proof))
    comparison=json.loads(Path(record['exact_resume']['path']).read_text())
    comparison['boundaries']['trainer_state']['world_size']=16
    comparison['boundaries']['config_hash']=config_hash(cfg)
    for r in range(8,16):
        for name in (f'rank_{r}.pt',f'random_states_{r}.pkl',f'pytorch_model/bf16_zero_pp_rank_{r}_mp_rank_00_optim_states.pt'):
            comparison['files'][name]={'status':'PASS','values':{'test_only':{'allclose':True}}}
    record['exact_resume']=put('compare16.json',comparison)
    save(cfg,record);enforce_training_budget(cfg,ctx)
    # A missing rank or failed peer cannot be hidden behind the summary.
    rows[1]['ranks'].pop()
    record['pilot_training']=put('missing_rank.jsonl',rows,True);save(cfg,record)
    with pytest.raises(ValueError,match='rank coverage'):enforce_training_budget(cfg,ctx)


def test_graph_needs_semantic_real_chain_evidence(tmp_path,monkeypatch):
    from starVLA.rl.flow_grpo.full_epoch import validate_velocity_graph_evidence
    cfg,ctx,record,rows,put=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError):validate_velocity_graph_evidence(None,cfg)
    report={'status':'PASS','device_type':'cuda','checkpoint_sha256':cfg['checkpoint_contract']['sha256'],
            'kernel_sha256':file_sha('starVLA/rl/flow_grpo/velocity_graph.py'),
            'script_sha256':file_sha('scripts/analysis/cuda_velocity_probe.py'),
            'live_weight_equal':True,'capture_rng_unchanged':True,
            'scenes':[{'tokens':[str(i)],'chain_equal':True,'old_equal':True,
                       'checks':[{k:{'equal':True,'max_abs':0.0} for k in ('velocity','mean','std','elementwise_logprob')} for _ in range(4)]} for i in range(4)]}
    validate_velocity_graph_evidence(put('graph.json',report),cfg)
    for change in ('status','device_type','kernel_sha256','live_weight_equal','capture_rng_unchanged'):
        bad=copy.deepcopy(report);bad[change]='BAD'
        with pytest.raises(ValueError):validate_velocity_graph_evidence(put('bad.json',bad),cfg)
    report['scenes'][2]['checks'][1]['velocity']['max_abs']=1e-8
    with pytest.raises(ValueError):validate_velocity_graph_evidence(put('bad.json',report),cfg)
