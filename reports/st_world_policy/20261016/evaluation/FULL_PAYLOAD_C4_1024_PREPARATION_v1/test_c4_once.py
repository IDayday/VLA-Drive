"""Small pinned metadata and mocked CLI tests; never run a real analysis."""
import ast
import copy
import importlib.util
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('c4_delta', Path(__file__).with_name('run_c4_once.py'))
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)


@pytest.fixture(scope='module')
def controls():
    module=r.original(); core,supervisor,observer=module.dependencies()
    original_pin=core.pinned; cache={}
    def cached(pin, *, decode=True):
        key=(pin['path'],pin['sha256'],decode)
        if key not in cache: cache[key]=original_pin(pin,decode=decode)
        return copy.deepcopy(cache[key])
    core.pinned=cached
    command,budget=r.fixed_command(module,core)
    r.verify_inputs(module,core,command,budget)  # exact actual small controls once
    return core,supervisor,observer,command,budget


def test_actual_pins_complete_inputs_and_only_fixed_c4(controls):
    core,_,_,command,budget=controls
    module=r.original()
    assert command['comparison']=='C4_minus_C0' and command['stage']=='DIAGNOSTIC_1024'
    assert budget['statistics_unchanged']['bootstrap_seed']==20261017
    assert budget['statistics_unchanged']['bootstrap_replicates']==2000
    assert module.option(command,'control-score').endswith('C0_step001024_dev1696_seed42_v1/official_score')
    assert module.option(command,'treatment-score').endswith('C4_step001024_dev1696_seed42_v1/official_score')


@pytest.mark.parametrize('key,value',[('treatment-score','/fake/C3'),('control-score','/fake/C0_step512'),('stage','COMMON_ENDPOINT'),('population','/fake/navtest'),('scope','SMOKE')])
def test_wrong_command_rejected(controls,key,value):
    core,*_=controls; module=r.original(); doc=core.pinned(r.COMMANDS)
    c=next(c for c in doc['paired_analyses'] if c['stage']=='DIAGNOSTIC_1024')
    c['argv'][c['argv'].index('--'+key)+1]=value
    stub=SimpleNamespace(pinned=lambda pin,**kw: copy.deepcopy(doc) if pin==r.COMMANDS else core.pinned(pin,**kw))
    with pytest.raises(ValueError): r.fixed_command(module,stub)


@pytest.mark.parametrize('field,value',[('status','PARTIAL'),('valid_scored',1695),('invalid',1)])
def test_partial_official_input_rejected(controls,field,value):
    core,_,_,command,budget=controls; module=r.original()
    target=budget['runs']['C4']['score_complete']; changed=core.pinned(target); changed[field]=value
    stub=SimpleNamespace(pinned=lambda pin,**kw: copy.deepcopy(changed) if pin==target else core.pinned(pin,**kw))
    with pytest.raises(ValueError): r.verify_inputs(module,stub,command,budget)


def test_existing_output_and_wrong_wrapper_rejected(controls,tmp_path,monkeypatch):
    core,_,_,command,budget=controls; module=r.original(); orig=module.option
    module.option=lambda c,k: str(tmp_path) if k=='output' else orig(c,k)
    with pytest.raises(ValueError,match='must be absent'): r.verify_inputs(module,core,command,budget)
    fake=tmp_path/'changed.py'; fake.write_text('# changed\n')
    monkeypatch.setattr(r,'WRAPPER',{'path':str(fake),'sha256':r.WRAPPER['sha256']})
    with pytest.raises(ValueError,match='changed'): r.original()


def test_child_body_identical_after_explicit_c4_selection():
    original=r.original()
    class Unqualify(ast.NodeTransformer):
        def visit_Attribute(self,node):
            node=self.generic_visit(node)
            if isinstance(node.value,ast.Name) and node.value.id=='module':
                return ast.Name(id=node.attr,ctx=node.ctx)
            return node
    def body(function,normalize=False):
        tree=ast.parse(inspect.getsource(function)).body[0]
        index=next(i for i,x in enumerate(tree.body) if isinstance(x,ast.Assign)
                   and any(isinstance(t,ast.Name) and t.id=='guard' for t in x.targets))
        result=ast.Module(body=tree.body[index:],type_ignores=[])
        if normalize: result=Unqualify().visit(result)
        return ast.dump(result,include_attributes=False)
    assert body(original.child)==body(r.child,True)


def test_child_routes_exact_c4_cli_without_real_import_or_payload(controls,tmp_path,monkeypatch,capsys):
    core,supervisor,observer,command,budget=controls; module=r.original()
    monkeypatch.setattr(r,'original',lambda:module)
    monkeypatch.setattr(module,'dependencies',lambda:(core,supervisor,observer))
    monkeypatch.setattr(module,'allowed_payloads',lambda *a:set())
    monkeypatch.setattr(module,'preload_analysis_imports',lambda:None)
    monkeypatch.setattr(module,'output_size',lambda *a:0)
    monkeypatch.setattr(r.sys,'addaudithook',lambda hook:None)
    monkeypatch.setattr(r.os,'getpgrp',r.os.getpid)
    calls=[]
    def fake_cli(name,**kwargs):
        calls.append((name,list(r.sys.argv)))
        assert r.sys.argv[1:]==command['argv'][3:]
        print(json.dumps({'fixture_only':True}))
        raise SystemExit(0)
    monkeypatch.setattr(r.runpy,'run_module',fake_cli)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    with (tmp_path/'fixture.lock').open('x') as lock:
        monkeypatch.setenv('ST_EVAL_CHILD_LOCK_FD',str(lock.fileno()))
        assert r.child()==0
    result=json.loads(capsys.readouterr().out)
    assert len(calls)==1 and result['cli_result']['exit_code']==0
    assert result['schema']=='st_full_future_child_result_v2'
    assert result['counters']['proc_self_io_final']['rchar']>0


@pytest.mark.parametrize('fail',[False,True])
def test_one_parent_attempt_no_c2_c3_calls(controls,tmp_path,monkeypatch,fail):
    core,supervisor,_,command,budget=controls; module=r.original()
    monkeypatch.setattr(r,'CONTROL',tmp_path/'new')
    monkeypatch.setattr(r,'original',lambda:module)
    observer=SimpleNamespace(proc_io=lambda:{'rchar':120})
    monkeypatch.setattr(module,'dependencies',lambda:(core,supervisor,observer))
    monkeypatch.setattr(module,'rss_monitor',lambda *a:None)
    monkeypatch.setattr(r.subprocess,'check_output',lambda argv,**kw:module.SOURCE if 'rev-parse' in argv else '')
    affinities=[]
    monkeypatch.setattr(r.os,'sched_setaffinity',lambda pid,cpus:affinities.append((pid,list(cpus))))
    calls=[]
    def fake_pair(selected,budget_,core_,supervisor_,observer_,metadata,lock,meter):
        assert selected==command and selected['comparison']=='C4_minus_C0'
        assert module.COMMANDS==r.COMMANDS and module.BUDGET==r.BINDINGS
        assert module.__file__==str(Path(r.__file__).absolute())
        assert module.PAIR_LIMITS['cpu_affinity']==[6,7]
        assert affinities[-1]==(0,[6,7])
        assert (r.CONTROL/'SESSION_STARTED.json').exists()
        calls.append(selected)
        if fail: raise RuntimeError('fixture only')
        meter.parent({'rchar':100}); return {'status':'COMPLETE','fixture_only':True}
    monkeypatch.setattr(module,'run_pair',fake_pair)
    if fail:
        with pytest.raises(RuntimeError,match='fixture only'): r.run()
    else: assert r.run()==0
    terminal=json.loads((r.CONTROL/'SESSION_TERMINAL.json').read_text())
    assert terminal['C2_C3_reruns'] is False and terminal['automatic_retry'] is False
    assert terminal['max_invocations']==1
    with pytest.raises(ValueError,match='Consumed session'): r.run()
    assert len(calls)==1


def test_original_limits_and_control_reserve():
    m=r.original()
    assert m.PAIR_LIMITS['cpu_affinity']==[6,7]
    assert m.TOTAL_READ==1<<30 and m.PAIR_LIMITS['rss_bytes']==2<<30
    assert m.PAIR_LIMITS['wall_seconds']==1800 and m.PAIR_LIMITS['cleanup_seconds']==40
    assert m.ANALYSIS_LIMIT+m.CONTROL_RESERVE+m.ADMISSION_RESERVE==32<<20
    assert 32*1024+2*16*1024<=m.ADMISSION_RESERVE
