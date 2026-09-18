"""Synthetic temporary evidence tests the real budget gate, not model acceptance."""
import copy
import json
from pathlib import Path
import pytest
from omegaconf import OmegaConf
from starVLA.rl.flow_grpo.acceptance import enforce_training_budget
from starVLA.rl.flow_grpo.loading import file_sha


def fixture(tmp_path, monkeypatch):
    monkeypatch.setenv('WORLD_SIZE','8')
    cfg=OmegaConf.to_container(OmegaConf.load('configs/flow_grpo/frozen_research_g16_flow_credit.yaml'))
    cfg['sampling']['num_steps']=10
    cfg['runtime'].update(run_mode='bounded_research',max_updates=64,research_evidence=str(tmp_path/'record.json'))
    context={'executable_sha256':'synthetic-test-only','checkpoint_sha256':cfg['checkpoint_contract']['sha256'],
             'world_size':8,'resume_identity':{'synthetic':True},'config_sha256':'test'}
    def put(name,data,lines=False):
        path=tmp_path/name
        path.write_text('\n'.join(json.dumps(x) for x in data) if lines else json.dumps(data))
        return {'path':str(path),'sha256':file_sha(path)}
    ranks=[{'rank':r,'device':{'type':'cuda'},'gradient_tensors':672,
      'behavior_sha256':['fixed'],'scene_tokens':['a'],'replay_tokens':['b'],
      'dtype':{'parameters':{'torch.bfloat16':1},'accumulation_dtype':'torch.float32',
        'communication_dtype':'torch.float32','master_weights':{'torch.float32':1},
        'communication_buffers':{'torch.float32':1},'optimizer_states':{'torch.float32':1}},
      'dtype_before_step':{'partition_buffers':['torch.float32']},'activation_dtypes':{'a':'torch.bfloat16'}} for r in range(8)]
    rows=[{'update':i+1,'inner_epoch':i,'scene_count':16,'candidate_count':256,
           'pre_update_ratio_min':1.,'pre_update_ratio_max':1.,'ranks':copy.deepcopy(ranks)} for i in range(2)]
    record={'schema_version':1,'status':'RESEARCH_ONLY','budget':64,'contexts':[context],
      'pilot_control':put('control.json',{'status':'PASS','exit_codes':[0]}),
      'pilot_context':put('context.json',context),'pilot_config':put('config.json',cfg),
      'pilot_training':put('training.jsonl',rows,True),'pilot_dtype':put('dtype.json',{'ranks':ranks}),
      'pilot_immutable':[put(f'immutable_update000002_rank{r}.json',{'status':'TESTED','end_update':2,
        'changed':{'frozen':[],'reference':[]},'before':{'frozen':{'weight':'hash'}},'after':{'frozen':{'weight':'hash'}}}) for r in range(8)]}
    return cfg,context,record,rows,put


def save(cfg,record):Path(cfg['runtime']['research_evidence']).write_text(json.dumps(record))


def test_real_budget_function_accepts_consistent_temporary_semantics(tmp_path,monkeypatch):
    cfg,ctx,record,_,_=fixture(tmp_path,monkeypatch);save(cfg,record)
    enforce_training_budget(cfg,ctx)
    cfg['runtime']['max_updates']=65
    with pytest.raises(ValueError,match='64'):enforce_training_budget(cfg,ctx)


@pytest.mark.parametrize('change',['cpu','missing_rank','chain','inner','dtype','source','fail','u','chunk','missing_proof'])
def test_false_research_evidence_rejected(tmp_path,monkeypatch,change):
    cfg,ctx,record,rows,put=fixture(tmp_path,monkeypatch)
    if change=='cpu':rows[1]['ranks'][0]['device']['type']='cpu'
    if change=='missing_rank':rows[1]['ranks'].pop()
    if change=='chain':rows[1]['ranks'][0]['behavior_sha256']=['changed']
    if change=='inner':rows[1]['inner_epoch']=0
    if change=='dtype':rows[1]['ranks'][0]['dtype']['accumulation_dtype']='torch.bfloat16'
    if change=='source':ctx=copy.deepcopy(ctx);ctx['executable_sha256']='changed'
    if change=='fail':record['pilot_control']=put('control.json',{'status':'FAIL','exit_codes':[1]})
    if change=='u':cfg['checkpoint_contract']['variant']='unfrozen_visual'
    if change=='chunk':cfg['sampling']['candidate_chunk_size']=2
    if change=='missing_proof':record['pilot_immutable'].pop()
    record['pilot_training']=put('training.jsonl',rows,True)
    record['pilot_dtype']=put('dtype.json',{'ranks':rows[-1]['ranks']})
    save(cfg,record)
    with pytest.raises(ValueError):enforce_training_budget(cfg,ctx)


def test_no_short_pilot_and_no_implicit_fast_profile_release(tmp_path,monkeypatch):
    cfg,ctx,record,rows,put=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='evidence'):enforce_training_budget(cfg,ctx)
    cfg['runtime']['activation_checkpointing']=False
    save(cfg,record)
    with pytest.raises((ValueError,KeyError)):enforce_training_budget(cfg,ctx)


def test_evidence_checksum_is_required(tmp_path,monkeypatch):
    cfg,ctx,record,_,_=fixture(tmp_path,monkeypatch);save(cfg,record)
    Path(record['pilot_control']['path']).write_text('{}')
    with pytest.raises(ValueError,match='changed'):enforce_training_budget(cfg,ctx)
