"""Dispatcher behavior with explicit stub children, never real training evidence."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
import torch
import yaml
from tools.action_effect import run_campaign as c
from tools.action_effect.common import artifact,digest,read_json,write_json


def args(**kw):
    values=dict(phase='all',arm=None,g1_matrix=False,include_score_refiner=False,effect_checkpoint=None,
        config='fixture',max_samples=1024,evaluation_max_samples=None,evaluation_role='validation',context_manifest=None,
        evaluation_context_manifest=None,evaluation_binding=None,seed=42,device='cpu',max_updates=2,stop_after=None,
        max_total_updates=12,max_simulation_budget=100,resume=False,checkpoint=None,max_seconds=10.)
    return SimpleNamespace(**{**values,**kw})


def test_default_plan_and_explicit_single_arm_and_own_g1_matrix():
    plan=c.task_plan(args());training=[x['arm'] for x in plan if x['phase'] in ('g1','route-a','route-b')]
    assert training==['delta_feedback','A0','B0','B1-trajectory','A1','B1-effect']
    single=c.task_plan(args(arm='B0'))
    assert [x['arm'] for x in single if x['phase'].startswith('route')]==['B0']
    assert not any(x['phase']=='g1' for x in single)
    assert [x['arm'] for x in single if x['phase']=='evaluation']==['S0','B0']
    matrix=c.task_plan(args(phase='g1',g1_matrix=True))
    assert [x['arm'] for x in matrix]==list(c.G1_MATRIX) and len(matrix)==6
    assert not any('privileged' in x['arm'] for x in matrix)
    teacher=c.task_plan(args(arm='A1',effect_checkpoint='/own/explicit.pt'))
    assert not any(x['phase']=='g1' for x in teacher)
    assert any(x['arm']=='ScoreRefiner' for x in c.task_plan(args(include_score_refiner=True)))


def test_score_refiner_missing_own_absolute_teacher_stays_waiting(tmp_path,monkeypatch):
    d=c.Dispatcher({'artifact_root':str(tmp_path)},args(include_score_refiner=True),tmp_path/'dispatch')
    write_json(d.context,{'fixture':'not real'});write_json(d.repairs,{'fixture':'not real'})
    monkeypatch.setattr(d,'call',lambda *a,**kw:pytest.fail('missing own teacher must not train'))
    with pytest.raises(c.Blocked,match='WAITING_INPUTS'):
        d.train_arm('ScoreRefiner')
    assert not (d.folder/'UPDATE_BUDGET.json').exists()


def make_checkpoint(output,step):
    output.mkdir(parents=True,exist_ok=True);identity={'unit':'explicit stub child','arm':output.parent.name}
    write_json(output/'IDENTITY.json',identity)
    path=output/f'step_{step:06d}.pt';torch.save({'run_identity':digest(identity),'optimizer_step':step},path)
    write_json(path.with_suffix('.COMPLETE.json'),{'run_identity':digest(identity),'optimizer_step':step,'checkpoint':artifact(path)})


def test_reservation_and_existing_run_refuse_duplicate_or_silent_budget_growth(tmp_path):
    d=object.__new__(c.Dispatcher);d.a=args(max_total_updates=2);d.folder=tmp_path/'dispatch';d.folder.mkdir()
    run=tmp_path/'models/population_001024/route_b/B0/seed_42'
    assert d.reserve_updates(run,'B0')==2
    with pytest.raises(c.Blocked,match='OPTIMIZER_BUDGET'):d.reserve_updates(tmp_path/'another','A0')
    make_checkpoint(run,2)
    with pytest.raises(c.Blocked,match='EXPLICIT_RESUME'):d.reserve_updates(run,'B0')
    d.a.resume=True
    assert d.reserve_updates(run,'B0')==0
    assert read_json(d.folder/'UPDATE_BUDGET.json')['reserved_new_updates']==2


def test_simulation_budget_includes_other_prefixes_arms_and_seeds(tmp_path):
    d=c.Dispatcher({'artifact_root':str(tmp_path)},args(),tmp_path/'dispatch')
    write_json(tmp_path/'g0/population_000256/calls/COUNTER.json',{'reserved_calls':30})
    write_json(tmp_path/'g0/population_000512/calls/COUNTER.json',{'reserved_calls':20})
    write_json(tmp_path/'evaluation/other/seed_43/dev_report/population_001696/score/calls/COUNTER.json',{'reserved_calls':10})
    assert c.used_calls(d.call_roots())==60
    assert d.budget(tmp_path/'new_score')==40
    d.a.max_simulation_budget=60
    with pytest.raises(c.Blocked,match='exhausted'):d.budget(tmp_path/'new_score')


def test_all_requires_explicit_aggregate_ceiling_before_any_child(monkeypatch,capsys):
    monkeypatch.setattr(c.subprocess,'run',lambda *a,**kw:pytest.fail('budget validation dispatched a child'))
    assert c.main(['--config','configs/action_effect/shared_budget_v2.yaml','--phase','all','--execute-ready-stages'])==1
    assert 'max-total-updates' in capsys.readouterr().out


def test_g1_failure_continues_into_actual_B0_child_command(tmp_path,monkeypatch,capsys):
    cfg=yaml.safe_load(Path('configs/action_effect/shared_budget_v2.yaml').read_text());cfg['artifact_root']=str(tmp_path/'artifacts')
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(cfg));root=Path(cfg['artifact_root'])
    context=root/'contexts/MANIFEST_001024.json';write_json(context,{'fixture':'not real context'})
    g0=root/'g0/population_001024'
    for name in ('COMPLETE.json','PAIR_DATASET.json','REPAIR_DATASET.json','G0_RESULTS.json'):write_json(g0/name,{'fixture':'explicit stub'})
    monkeypatch.setattr(c,'audit',lambda *a:{'status':'PASS','scope':'unit fixture only'})
    commands=[]
    def child(command,**kw):
        commands.append(command);module=command[2]
        def option(name):return command[command.index(name)+1]
        if module.endswith('train_effect_model'):
            assert option('--arm')=='delta_feedback'
            return SimpleNamespace(returncode=1) # Actual Dispatcher.call sees a failed child.
        if module.endswith(('train_refiner','train_effect_flow')):
            output=Path(option('--output'));make_checkpoint(output,2)
            write_json(output/'RESULT.json',{'status':'COMPLETE','optimizer_step':2})
        elif '--phase' in command and option('--phase')=='predict':
            write_json(Path(option('--output'))/'PREDICTIONS.json',{'fixture':'not real predictions'})
        elif '--phase' in command and option('--phase')=='score':
            write_json(Path(option('--output'))/'RESULT.json',{'status':'COMPLETE','summary':{'fixture':True}})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(c.subprocess,'run',child)
    # check_output uses subprocess.run internally; pin source query separately.
    monkeypatch.setattr(c.subprocess,'check_output',lambda *a,**kw:'fixed-unit-source\n')
    argv=['--config',str(path),'--phase','all','--execute-ready-stages','--continue-independent-on-failure',
          '--max-samples','1024','--max-updates','2','--max-total-updates','12','--max-simulation-budget','100','--device','cpu']
    assert c.main(argv)==1
    receipt=json.loads(capsys.readouterr().out)
    actual_training=[(x[2],x[x.index('--arm')+1]) for x in commands if '--arm' in x]
    assert ('tools.action_effect.train_effect_flow','B0') in actual_training
    assert ('tools.action_effect.train_effect_flow','B1-trajectory') in actual_training
    assert ('tools.action_effect.train_refiner','A0') in actual_training
    assert not any(arm in ('A1','B1-effect') for _,arm in actual_training)
    assert not any('--effect-checkpoint' in x for x in commands)
    assert receipt['optimizer_dispatch_budget']['reserved_new_updates']==8
    assert receipt['status']=='INCONCLUSIVE' and any(x['arm']=='A1' and x['status']=='BLOCKED' for x in receipt['stages'])
    for command in commands:
        if '--output' in command and '--arm' in command:
            assert 'population_001024' in option_from(command,'--output') and 'seed_42' in option_from(command,'--output')
    before=sum(arm=='B0' for _,arm in actual_training)
    assert c.main(argv)==1
    capsys.readouterr()
    assert sum(x[2].endswith('train_effect_flow') and '--arm' in x and option_from(x,'--arm')=='B0' for x in commands)==before


def option_from(command,name):return command[command.index(name)+1]
