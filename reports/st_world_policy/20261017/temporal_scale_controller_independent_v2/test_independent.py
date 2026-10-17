"""Controller-only checks; synthetic CPU process, no measurement/model assets."""
import importlib.util,json,os,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path('/mnt/project/VLA-Drive-st-loss-scale-real4-v2-20261017/reports/st_world_policy/20261017')
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path); obj=importlib.util.module_from_spec(spec); spec.loader.exec_module(obj); return obj
c=load('review_calibration_controller',ROOT/'temporal_scale_calibration_v1/run_once.py')

@pytest.mark.parametrize('field',['control_expiry','checkpoint_expiry'])
def test_nonfinite_expiry_is_rejected(monkeypatch,tmp_path,field):
 cp=dict(schema='st_checkpoint_storage_commitments_v1',status='CURRENT',remaining_bytes=2303<<30,free_floor_bytes=1<<40,valid_until_unix=time.time()+5000)
 control={'checkpoint_receipt':{},'valid_until_unix':time.time()+4000}
 if field=='control_expiry': control['valid_until_unix']=float('nan')
 else: cp['valid_until_unix']=float('nan')
 monkeypatch.setattr(c,'ROOT',tmp_path)
 monkeypatch.setattr(c,'pin_read',lambda _: cp)
 monkeypatch.setattr(c.os,'statvfs',lambda _:SimpleNamespace(f_bavail=1<<50,f_frsize=1))
 with pytest.raises(ValueError,match='commitments|lease|finite|window'):
  c.storage_guard(control)


def test_fifo_control_refused_before_blocking_read(tmp_path):
 fifo=tmp_path/'CONTROL.json'; os.mkfifo(fifo)
 code="import importlib.util;from pathlib import Path;s=importlib.util.spec_from_file_location('c',%r);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.pin_read({'path':%r,'sha256':'0'*64})"%(str(Path(c.__file__)),str(fifo))
 p=subprocess.Popen([sys.executable,'-c',code],stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
 blocked=False
 try:
  try: out,err=p.communicate(timeout=.35)
  except subprocess.TimeoutExpired:
   blocked=True; p.terminate(); out,err=p.communicate(timeout=2)
 finally:
  if p.returncode is None: p.kill(); p.wait(timeout=2)
 assert not blocked,'FIFO accepted by pin_read; parent blocks before any owned monitor/deadline starts'
 assert p.returncode!=0 and (b'Canonical' in err or b'Small control/source only' in err)


def test_real_existing_core_accounts_setup_child_and_wait_once(tmp_path):
 sys.path.insert(0,str(ROOT/'spatial_probe_actual_preparation_v1'))
 import supervise_once as sup
 import observed_cli as obs
 from tools.st_world_policy import metadata_storage
 core=load('review_actual_calibration_monitor',ROOT/'full_future_metadata_preparation_v1/run_build_once.py')
 script=tmp_path/'child.py'
 script.write_text('''import json,os,sys,time\nfrom pathlib import Path\ndef io():\n return {k:int(v) for k,v in (x.split(':') for x in Path('/proc/self/io').read_text().splitlines())}\na=io(); Path(sys.argv[1]).read_bytes(); time.sleep(.2); b=io()\nprint(json.dumps({'schema':'st_full_future_child_result_v2','cli_result':{'exit_code':0},'error':None,'counters':{'proc_self_io_baseline':a,'proc_self_io_final':b,'proc_self_io_delta':{k:b[k]-a[k] for k in b}}}))\n''')
 payload=tmp_path/'synthetic.bin';payload.write_bytes(b'x'*(1<<20))
 before=obs.proc_io();core.BOOT_IO=before
 payload.read_bytes() # setup work before entering monitor must still be charged
 checks=[]
 budget={'wall_seconds':2,'read_bytes':32<<20,'rss_bytes':1<<30}
 with (tmp_path/'lock').open('a+b') as f:
  result=core.monitor([sys.executable,str(script),str(payload)],runtime=str(tmp_path),env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),fd=f.fileno(),root=tmp_path,s=sup,m=metadata_storage,obs=obs,budget=budget,cpu_affinity=(6,7),storage_guard=lambda:checks.append(time.monotonic()),lease_valid_until=time.time()+100)
 after=obs.proc_io()
 assert result['returncode']==0 and result['failure'] is None
 assert result['cleanup']['live_owned_remaining']==0 and result['cleanup']['verified_processes']==1
 assert result['read_accounting']['status']=='COMPLETE'
 charged=core.charge_after_reap(result['read_accounting'],after)
 assert charged==after['rchar']-before['rchar']
 assert charged>=2<<20 and charged<32<<20
 assert checks and result['peak_aggregate_rss_bytes']>0
 (tmp_path/'core_result.json').write_text(json.dumps(result,indent=2))


def test_source_inventory_is_exact_five_and_cli_no_update_contract():
 import ast
 node=ast.parse(Path(c.__file__).read_text())
 assert c.LIMITS['optimizer_updates']==0 and c.LIMITS['new_tensor_bytes']==0 and c.LIMITS['max_invocations']==1
 assert c.LIMITS['max_disk_bytes']==16<<20 and c.LIMITS['cpu_affinity']==[2,3]
 # Specific child's measure entry is inherited, not an alternate trainer.
 assert "'--config', admission['config']['path']" in Path(c.__file__).read_text()
 assert 'len(cfg[\'batch\'][\'scene_indices\']) != 4' in Path(c.__file__).read_text()
 assert 'core.BOOT_IO = BOOT_IO' in Path(c.__file__).read_text()

@pytest.mark.parametrize('field',['issued_at_unix','latest_start_unix','valid_until_unix'])
def test_parent_rejects_nonfinite_window_before_imports(monkeypatch,field):
 control=dict(schema='st_loss_scale_owned_control_v1',limits=c.LIMITS,host=c.socket.gethostname(),runtime=str(c.RUNTIME),output_root=str(c.ROOT),issued_at_unix=0,latest_start_unix=1e20,valid_until_unix=1e20)
 control[field]=float('nan')
 monkeypatch.setattr(c,'pin_read',lambda pin:control)
 with pytest.raises(ValueError,match='Finite absolute resource times'):
  c.controls(SimpleNamespace(control='/fixture',control_sha256='0'*64,child=False))
