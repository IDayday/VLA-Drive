"""Independent tiny CPU boundaries; no real analysis/payload opening."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import pytest

SOURCE=Path(__file__).resolve().parents[2]/'20261016/evaluation/FULL_PAYLOAD_C2_C3_1024_EXECUTION_v2/run_fixed_pairs.py'
# parents[2] is st_world_policy.
spec=importlib.util.spec_from_file_location('independent_fixed_eval',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
OUT=Path(__file__).parent

@pytest.mark.parametrize('name',['SESSION_STARTED.json','SESSION_TERMINAL.json','C2_minus_C0'])
def test_consumed_session_cannot_restart_any_budget(tmp_path,monkeypatch,name):
    monkeypatch.setattr(m,'CONTROL',tmp_path)
    commands=[{'comparison':'C2_minus_C0'},{'comparison':'C3_minus_C0'}]
    assert m.session_state(commands) is False
    target=tmp_path/name
    if name.endswith('.json'):target.write_text(json.dumps({'status':'COMPLETE','charged_whole_process_rchar_bytes':900}))
    else:target.mkdir()
    with pytest.raises(ValueError,match='Consumed session'):m.session_state(commands)


def test_shared_controls_fit_pair_and_session_serialized_budget():
    assert m.ANALYSIS_LIMIT+m.CONTROL_RESERVE+m.ADMISSION_RESERVE==32*(1<<20)
    assert m.ADMISSION_RESERVE//2+2*16*1024<=m.ADMISSION_RESERVE
    assert m.TOTAL_READ==1<<30
    assert m.PAIR_LIMITS['rss_bytes']==2<<30 and m.PAIR_LIMITS['wall_seconds']==1800


def test_read_guard_remains_active_while_import_write_limit_disabled(tmp_path):
    guard=m.OutputGuard(tmp_path,[]);guard.enforce_writes=False
    guard.audit('open',('/dev/null','w',os.O_WRONLY))
    for forbidden in ('/fake/image.jpg','/fake/model.pt','/dev/nvidia0'):
        with pytest.raises(ValueError):guard.audit('open',(forbidden,'r',os.O_RDONLY))
    guard.enforce_writes=True
    with pytest.raises(ValueError):guard.audit('open',('/dev/null','w',os.O_WRONLY))


def test_second_pair_receives_only_remaining_cumulative_allowance():
    core,_,_=m.dependencies();meter=m.ReadMeter(cap=1000)
    assert meter.parent({'rchar':100})==900
    account=core.terminal_read_accounting({'rchar':100},{'rchar':120},{'rchar':620},500,900,final_child_available=True)
    meter.reaped(core,account,{'rchar':650})
    assert meter.total==650
    assert meter.parent({'rchar':700})==300
    account=core.terminal_read_accounting({'rchar':700},{'rchar':710},{'rchar':990},280,300,final_child_available=True)
    meter.reaped(core,account,{'rchar':995})
    assert meter.total==995
    with pytest.raises(ValueError):meter.parent({'rchar':1001})


def test_first_session_reuses_already_complete_output_without_child(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'CONTROL',tmp_path/'controls')
    verification={'path':str(tmp_path/'ANALYSIS.json'),'sha256':'fixture','output_bytes':32}
    monkeypatch.setattr(m,'complete_output',lambda *a:verification)
    core=SimpleNamespace(monitor=lambda *a,**k:pytest.fail('No analysis child when complete'))
    meter=m.ReadMeter(cap=1000)
    result=m.run_pair({'comparison':'C2_minus_C0'},None,core,None,SimpleNamespace(proc_io=lambda:{'rchar':100}),None,None,meter)
    assert result=={'status':'ALREADY_COMPLETE_VERIFIED','analysis':verification}
    assert meter.total==100
    assert not (tmp_path/'controls').exists()


def test_actual_cold_import_and_help_under_live_guard(tmp_path):
    core,supervisor,observer=m.dependencies();metadata=m.rss_monitor(core)
    script=r'''import importlib.util,sys,json,io,runpy
from pathlib import Path
from contextlib import redirect_stdout
spec=importlib.util.spec_from_file_location('review_child',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
core,_,obs=m.dependencies();before=obs.proc_io()
root=Path(sys.argv[2]);root.mkdir()
guard=m.OutputGuard(root,[]);guard.enforce_writes=False
sys.addaudithook(guard.audit);sys.path.insert(0,str(m.RUNTIME));sys.dont_write_bytecode=True
m.preload_analysis_imports();guard.enforce_writes=True
sys.argv=['tools.st_world_policy.evaluate_campaign','--help'];output=io.StringIO()
with guard.writers(),redirect_stdout(output):
    try:runpy.run_module('tools.st_world_policy.evaluate_campaign',run_name='__main__',alter_sys=True)
    except SystemExit as exc:assert exc.code==0
    from tools.st_world_policy.base_binding import immutable_json
    immutable_json(root/'ANALYSIS.json',{'review':'tiny only; no analysis payload'})
assert guard.written==m.output_size(root)
final=obs.proc_io()
print(json.dumps({'schema':'st_full_future_child_result_v2','cli_result':{'exit_code':0,'payload_reads':0,'help_pass':True,'fixture_bytes':guard.written},'counters':{'proc_self_io_baseline':before,'proc_self_io_final':final,'proc_self_io_delta':{k:final[k]-before[k] for k in final}}}))
'''
    core.BOOT_IO=observer.proc_io();core.START=time.monotonic()
    with (tmp_path/'owned.lock').open('x') as lock:
        result=core.monitor([m.PYTHON,'-c',script,str(SOURCE),str(tmp_path/'out')],runtime=str(tmp_path),env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},fd=lock.fileno(),root=tmp_path,s=supervisor,m=metadata,obs=observer,budget={**m.PAIR_LIMITS,'wall_seconds':30,'read_bytes':256*m.MiB})
    (OUT/'INDEPENDENT_COLD_IMPORT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    assert result['failure'] is None and result['returncode']==0
    assert result['cleanup']['live_owned_remaining']==0
    assert result['read_accounting']['status']=='COMPLETE'
    assert result['child_accounting']['cli_result']['help_pass'] is True
    assert result['peak_aggregate_rss_bytes']<2<<30
