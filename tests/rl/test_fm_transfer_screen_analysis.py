import json
import pytest
from test_fm_transfer_screen import fixture_screen
from scripts.analysis.fm_transfer.build_hard_manifest import build
from scripts.analysis.fm_transfer import analyze_screen
from starVLA.rl.flow_grpo.transactions import atomic_json


def prepared(tmp_path,monkeypatch):
    pop,identity=fixture_screen(tmp_path)
    for t in pop['train_tokens']:
        p=tmp_path/'scenes'/t/(t+'.json');d=json.loads(p.read_text())
        d['metrics'].update(ego_progress=.5,comfort=1.,driving_direction_compliance=1.);atomic_json(p,d)
    population=tmp_path/'population.json';atomic_json(population,pop);monkeypatch.setattr(analyze_screen,'POPULATION',population)
    hard=tmp_path/'hard.json';build(tmp_path,hard,workers=1,population=pop)
    inv=tmp_path/'invocations/good/rank0.json';inv.parent.mkdir(parents=True)
    record=dict(status='ASSIGNED_COMPLETE',identity_sha256=identity['sha256'],host='test',rank=0,
        start_unix=10.,end_unix=13.,new=3,reused=0,actual_new_inferences=3,model_load_seconds=.4,
        generation_seconds=1.,scoring_wait_seconds=.1,worker_scoring_seconds=.2,total_worker_wall_seconds=3.,
        unpublished_scoring_requests=[])
    atomic_json(inv,record)
    return hard,inv,record


def test_fixed_single_ODE_population_and_actual_cost(tmp_path,monkeypatch):
    hard,_,_=prepared(tmp_path,monkeypatch);d=analyze_screen.summarize(tmp_path,hard)
    assert d['scenes']==3 and d['hard_scenes']==2
    assert d['counts']['score_exact_one']==1 and d['counts']['score_exact_zero']==1
    assert d['counts']['hard_safe']==1 and d['counts']['hard_not_fully_safe']==1
    assert d['cost']['recorded_physical_predictions']==3 and d['cost']['job_wall_seconds']==3.


def test_failed_unpublished_work_is_not_free(tmp_path,monkeypatch):
    hard,_,record=prepared(tmp_path,monkeypatch)
    p=tmp_path/'invocations/failed/rank0.json';p.parent.mkdir(parents=True)
    atomic_json(p,dict(record,status='FAILED',start_unix=9.,end_unix=10.,new=0,
        actual_new_inferences=1,total_worker_wall_seconds=1.,unpublished_scoring_requests=['aa']))
    d=analyze_screen.summarize(tmp_path,hard)['cost']
    assert d['recorded_new_committed_scenes']==3 and d['recorded_physical_predictions']==4
    assert d['cumulative_worker_wall_seconds']==4. and d['scoring_time_may_be_incomplete']


def test_nonterminal_invocation_leaves_cost_unknown(tmp_path,monkeypatch):
    hard,p,r=prepared(tmp_path,monkeypatch);r.pop('end_unix');atomic_json(p,r)
    d=analyze_screen.summarize(tmp_path,hard)['cost']
    assert not d['complete_duration_ledger'] and d['cumulative_worker_wall_seconds'] is None
    assert d['job_wall_seconds'] is None and d['physical_count_is_lower_bound_if_missing_invocations']


def test_complete_ledger_cannot_silently_lose_new_scenes(tmp_path,monkeypatch):
    hard,p,r=prepared(tmp_path,monkeypatch);r['new']=2;atomic_json(p,r)
    with pytest.raises(ValueError):analyze_screen.summarize(tmp_path,hard)
