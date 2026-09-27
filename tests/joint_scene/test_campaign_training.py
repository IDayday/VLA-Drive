import json
import pytest
from tools.joint_local_scene_v3.runtime import evaluate,EvaluationBudgetPause
from tools.joint_local_scene_v3.train_mechanism import parser,validate_arguments
from test_queries_and_config import corpus
from test_joint_scene import model


def test_interrupted_evaluation_retains_rows_and_fixed_denominator(tmp_path):
    calls=0
    def budget():
        nonlocal calls
        calls+=1
        if calls==3:raise RuntimeError('Test budget exhausted')
    with pytest.raises(EvaluationBudgetPause):evaluate(model(),corpus(),device='cpu',sampling_steps=1,output=tmp_path,budget_check=budget,evaluation_identity={'checkpoint':'fixed'})
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert summary['expected_queries']==3 and summary['evaluated_queries']==1
    assert len(summary['missing_query_ids'])==2 and not summary['aggregate_valid']
    assert (tmp_path/'query_progress.jsonl').read_text().count('\n')==1
    assert (tmp_path/'all_hidden_targets.csv').exists()


def test_diagnostic_eval_requires_fixed_subset_and_save_interval(tmp_path):
    a=parser().parse_args(['--train','train','--holdout','hold','--config','cfg','--output',str(tmp_path/'out'),'--ledger','ledger','--run-id','run','--mode','all','--updates','4','--schedule-updates','4','--eval-train'])
    with pytest.raises(ValueError,match='diagnostic'):validate_arguments(a)
    a.diagnostic_subset='fixed64';a.save_every=0
    with pytest.raises(ValueError,match='save_every'):validate_arguments(a)
    a.save_every=100;assert validate_arguments(a)==tmp_path/'out'


def test_batched_fixed_queries_preserve_scalar_protocol():
    from tools.joint_local_scene_v3.batched_evaluation import evaluate_batched
    m=model();data=corpus();scalar=evaluate(m,data,device='cpu',seed=81,sampling_steps=2)
    batched=evaluate_batched(m,data,device='cpu',seed=81,sampling_steps=2,batch_size=2)
    assert scalar['query_manifest']==batched['query_manifest']
    for a,b in zip(scalar['query_rows'],batched['query_rows']):
        assert a['query_id']==b['query_id'] and a['valid_xy_points']==b['valid_xy_points']
        for k in ('all_hidden_xy_ADE_m','conditional_xy_ADE_m'):assert abs(a[k]-b[k])<1e-4


def test_relation_ablation_selection_does_not_read_hidden_future_values():
    from tools.joint_local_scene_v3.endpoint_diagnostics import relation_queries
    from tools.joint_local_scene_v3.runtime import build_queries
    data=corpus();q=build_queries(data);first=relation_queries(data,q)
    data[0].future.fill_(987.)
    assert relation_queries(data,q)==first
    assert first['queries'][0]['status']=='applicable'
    for row in first['queries']:
        if row['status']=='applicable':
            assert row['strong_actor']!=row['weak_actor'] and row['strong_actor']!=row['slot'] and row['weak_actor']!=row['slot']
