"""Independent pre-return Popen signal cases; controlled sleeping CPU children."""
import importlib.util,os,signal,subprocess,sys,json
from pathlib import Path
import pytest
OWNER=Path(os.environ['ST_ENDPOINT_IMPLEMENTATION_ROOT'])
spec=importlib.util.spec_from_file_location('signal_endpoint_owner_fixtures',OWNER/'tests/st_world/test_endpoint_supervisor.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
s=t.s

@pytest.mark.parametrize('first,second',[(signal.SIGTERM,signal.SIGINT),(signal.SIGINT,signal.SIGTERM)])
def test_both_signals_before_Popen_return_preserve_first_and_clean_once(tmp_path,monkeypatch,first,second):
 monkeypatch.setattr(s,'OUTPUT',tmp_path/'job');monkeypatch.setattr(s,'ANALYSIS',tmp_path/'analysis')
 monkeypatch.setattr(s,'immutable',t.immutable)
 (s.OUTPUT/'control').mkdir(parents=True)
 binding,documents,_=t.endpoint_fixture(tmp_path/'source',monkeypatch)
 route={'gpus':[0,1,2,3],'gpu_uuids':['fixture0','fixture1','fixture2','fixture3']}
 monkeypatch.setattr(s,'mature_helpers',lambda:{'check_route_time_v3':lambda *args:None,'fresh_gpu_v3':lambda *args:None})
 monkeypatch.setattr(s,'route_checks',lambda *args:route)
 monkeypatch.setattr(s,'gpu_monitor',lambda *args:{})
 monkeypatch.setattr(s,'argv_for',lambda *args:[sys.executable,'-c','import time;time.sleep(60)'])
 handlers={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
 original=subprocess.Popen;children=[]
 def before_return(*args,**kwargs):
  child=original(*args,**kwargs);children.append(child)
  os.kill(os.getpid(),first);os.kill(os.getpid(),second)
  return child
 monkeypatch.setattr(s.subprocess,'Popen',before_return)
 try:
  result=s.execute('export',{'synthetic':'explicit test receipt'},binding,documents,resume=False)
  assert result['status']=='PARTIAL_OR_FAILED'
  assert result['stop_reason']=='RuntimeError: OPERATOR_SIGNAL_'+str(int(first))
  assert result['invocations_started']==1 and len(children)==1
  assert children[0].poll() is not None
  attempts=list((s.OUTPUT/'control/invocations').glob('*/*'))
  assert len(attempts)==1
  assert (attempts[0]/'ADMISSION.json').is_file()
  assert json.loads((attempts[0]/'EXIT.json').read_text())['owned_cleanup_complete']
  assert all(signal.getsignal(sig)==old for sig,old in handlers.items())
 finally:
  for child in children:
   if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
