import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
import pytest
HERE=Path(__file__).parent

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
p=module('history_pilot_test',HERE/'run_pilot_once.py')
core=module('history_core_test',HERE.parent/'full_future_metadata_preparation_v1/run_build_once.py')
helper=HERE.parent/'spatial_probe_actual_preparation_v1';sys.path.insert(0,str(helper.resolve()))
import supervise_once as s
import observed_cli as obs
from tools.st_world_policy import metadata_storage as m


def admission():
    return dict(schema='st_history64_owned_admission_v1',status='APPROVED_HISTORY64_PILOT_ONLY',limits=p.LIMITS.copy(),reviewer='/independent',hostname=p.socket.gethostname(),checkpoint_receipt=p.CP.copy(),output_root=str(p.ROOT),runtime=str(p.RUNTIME),control_root=str(p.CONTROL_ROOT),control_reserved_bytes=p.CONTROL_CAP,issued_at_unix=0,latest_start_unix=100,valid_until_unix=1000)

@pytest.mark.parametrize('key,value',[('reviewer','/root/st_temporal_readiness_inventory'),('valid_until_unix',100),('output_root','/tmp/wrong'),('status','PROPOSED')])
def test_admission_rejects(key,value):
    a=admission();a[key]=value
    with pytest.raises(ValueError):p.validate(a,50)


def test_setup_expiry_prevents_spawn(tmp_path):
    with patch.object(core.subprocess,'Popen',side_effect=AssertionError('must not spawn')):
        with pytest.raises(ValueError,match='actual-start lease'):
            core.monitor([],runtime=str(tmp_path),env={},fd=0,root=tmp_path,s=s,m=m,obs=obs,budget={'wall_seconds':900},cpu_affinity=(6,7),lease_valid_until=time.time()+2)


def test_real_tiny_owned_child_accounting_cpu_and_cleanup(tmp_path):
    code="""import os,json
before={k:int(v) for k,v in (s.split(':') for s in open('/proc/self/io'))}
after={k:int(v) for k,v in (s.split(':') for s in open('/proc/self/io'))}
print(json.dumps({'schema':'st_full_future_child_result_v2','cli_result':{'affinity':sorted(os.sched_getaffinity(0))},'counters':{'proc_self_io_baseline':before,'proc_self_io_final':after,'proc_self_io_delta':{k:after[k]-before[k] for k in after}}}))
"""
    checks=[]
    with (tmp_path/'owned').open('w') as f:
        r=core.monitor([sys.executable,'-c',code],runtime=str(tmp_path),env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),fd=f.fileno(),root=tmp_path,s=s,m=m,obs=obs,budget={'wall_seconds':20,'read_bytes':1<<28,'rss_bytes':4<<30},cpu_affinity=(6,7),storage_guard=lambda:checks.append(True),lease_valid_until=time.time()+120)
    assert r['returncode']==0 and r['failure'] is None
    assert r['cleanup']['live_owned_remaining']==0
    assert r['child_accounting']['cli_result']['affinity']==[6,7]
    assert r['read_accounting']['status']=='COMPLETE' and checks
