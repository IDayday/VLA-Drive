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
