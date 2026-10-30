"""Research evaluation must preserve scene identity and feedback boundaries."""
import json
from pathlib import Path

import pytest

from tools.qwen_world_research.evaluate import (
    INFERENCE,SCHEMA,diagnostic_bank,load_evaluation,sealed,validate_manifest,
    verify_export_binding,
)
from starVLA.rl.qwen_world_postcot.contracts import fingerprint


@pytest.fixture
def population():
    return dict(train_tokens=['a','b'],dev_tokens=['dev'],navtest_tokens=['test'],
                token_logs=dict(a='trainA_000_010',b='trainB_000_010',dev='devA_000_010',test='testA_000_010'))


def check(manifest,population,cohort='TRAIN_DIAGNOSTIC'):
    return validate_manifest(manifest,cohort=cohort,data_sha256='data',
        data_tokens=list(population['token_logs']),token_logs=population['token_logs'],population=population)


def test_train_diagnostic_accepts_full_navtrain_but_marks_old_dev(population):
    result=check(sealed(dict(tokens=['a','b','dev'],count=3,data_manifest_sha256='data')),population)
    assert result['train_feedback_allowed'] is True
    assert result['old_DEV_is_train_diagnostic'] is True


@pytest.mark.parametrize('tokens',[['a','a'],['test'],['a','test'],[]])
def test_training_feedback_rejects_duplicates_or_navtest(population,tokens):
    with pytest.raises(ValueError):check(dict(tokens=tokens),population)


def test_training_feedback_rejects_same_physical_log_as_navtest(population):
    population['token_logs']['a']='testA_011_020'
    with pytest.raises(ValueError):check(dict(tokens=['a']),population)


def test_partial_navtest_never_accepted(population):
    with pytest.raises(ValueError):check(dict(tokens=['test']),population,'NAVTEST')


def test_evaluation_only_snapshot_cannot_be_promoted_to_train_feedback(population):
    with pytest.raises(ValueError):check(dict(tokens=['a'],used_for_actor_update=False),population)
    with pytest.raises(ValueError):check(dict(tokens=['a'],train_feedback_allowed=False),population)


def test_export_requires_exact_formal_scope_and_parent(tmp_path):
    parent=tmp_path/'parent';rl=tmp_path/'rl'
    meta=dict(qualification=False,IL_parent=dict(path=str(parent),identity={'qualification':False}),
        RL_checkpoint=str(rl),RL_parent=dict(schema='world_postcot_cleanx_scope_RL_v1',purpose='formal'))
    verify_export_binding(meta,parent_checkpoint=parent,rl_checkpoint=rl)
    with pytest.raises(ValueError):verify_export_binding(meta,parent_checkpoint=parent,rl_checkpoint=tmp_path/'other')
    with pytest.raises(ValueError):verify_export_binding(meta,parent_checkpoint=parent)


def make_output(tmp_path,*,complete_rank=True,extra=False,missing=False):
    output=tmp_path/'eval';(output/'scenes').mkdir(parents=True)
    identity=sealed(dict(schema=SCHEMA,qualification=False,cohort='TRAIN_DIAGNOSTIC',
        train_feedback_allowed=True,tokens=['a','b'],count=2,token_sha256=fingerprint(['a','b']),
        token_logs={'a':'logA_000_010','b':'logB_000_010'},inference=INFERENCE,seed=2101,
        protocol={'sha256':'protocol'},model_sha256='policy',data_sha256='data'))
    (output/'identity.json').write_text(json.dumps(identity))
    (output/'layout.json').write_text(json.dumps(dict(world=1)))
    (output/'rank0.json').write_text(json.dumps(dict(status='COMPLETE' if complete_rank else 'RUNNING',
        rank=0,world=1,identity=identity['sha256'])))
    for token in (['a'] if missing else ['a','b']):
        (output/'scenes'/f'{token}.json').write_text(json.dumps(dict(token=token)))
    if extra:(output/'scenes'/'foreign.json').write_text('{}')
    return output,identity


@pytest.mark.parametrize('options',[dict(missing=True),dict(extra=True),dict(complete_rank=False)])
def test_aggregation_requires_exact_population_and_completed_shards(tmp_path,options):
    output,_=make_output(tmp_path,**options)
    with pytest.raises(ValueError):load_evaluation(output,row_validator=lambda row,**kw:row)


def test_complete_census_retains_all_rows(tmp_path):
    output,_=make_output(tmp_path)
    _,rows=load_evaluation(output,row_validator=lambda row,**kw:row)
    assert set(rows)=={'a','b'}


def test_navtest_cannot_emit_adaptive_training_bank(tmp_path):
    _,identity=make_output(tmp_path)
    identity.update(cohort='NAVTEST',train_feedback_allowed=False)
    with pytest.raises(ValueError):diagnostic_bank(identity,{'a':{},'b':{}})


def test_train_bank_policy_separate_from_inference_protocol(tmp_path):
    _,identity=make_output(tmp_path)
    rows={t:dict(log=identity['token_logs'][t],feedback={'metrics':{'score':.75}}) for t in ['a','b']}
    bank=diagnostic_bank(identity,rows)
    assert bank['policy_sha256']=='policy'
    assert 'policy_sha256' not in bank['inference']
    assert bank['cohort']=='RL_TRAIN_DIAGNOSTIC'
    assert bank['records']['a']['score']==.75
    assert bank['score_units']=='0..1'
