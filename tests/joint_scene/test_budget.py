import json
import pytest
from tools.joint_local_scene_v3.budget import BudgetRun,initialize_ledger


def test_review_ledger_forbids_real_updates_and_duplicate_runs(tmp_path):
    p=tmp_path/'budget.json';initialize_ledger(p)
    identity={'output':str(tmp_path/'run'),'source':'unit-test-no-optimizer'}
    with BudgetRun(p,'unique',identity) as run:
        with pytest.raises(RuntimeError,match='real'):run.require_updates('real')
        with pytest.raises(ValueError):
            with BudgetRun(p,'unique',identity):pass
        with pytest.raises(ValueError):
            with BudgetRun(p,'another',identity):pass
    data=json.loads(p.read_text());assert data['optimizer_updates']=={'synthetic':0,'real':0}
    assert len(data['runs'])==1 and data['runs'][0]['status']=='complete'


def test_budget_failure_is_recorded_and_different_resume_identity_rejected(tmp_path):
    p=tmp_path/'budget.json';initialize_ledger(p)
    with pytest.raises(RuntimeError,match='intentional'):
        with BudgetRun(p,'failed',{'source':'one'}):raise RuntimeError('intentional')
    data=json.loads(p.read_text());assert data['runs'][0]['status']=='failed' and data['runs'][0]['wall_seconds']>=0
    with pytest.raises(ValueError):
        with BudgetRun(p,'failed',{'source':'two'},resume=True):pass
