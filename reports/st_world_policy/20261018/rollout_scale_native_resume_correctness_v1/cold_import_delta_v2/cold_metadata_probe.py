"""Isolated CPU metadata probe; never proceeds to mount, GPU, model or trainer."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

controller=Path(sys.argv[1]);reference=Path(sys.argv[2])
assert not any(name=='tools' or name.startswith('tools.') for name in sys.modules)
assert '/mnt/project/VLA-Drive-st-rollout-scale-runtime-v2-20261018' not in sys.path
spec=importlib.util.spec_from_file_location('cold_reviewed_controller',controller)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
# Exercise the exact mature helper responsible for the observed failure.
helper_path=Path('/mnt/project/st-world-policy-20261016/resources/launch_formal_dev_export_v5.py')
tree=ast.parse(helper_path.read_text())
node=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='fresh_environment_v3')
ns={'Path':Path,'json':json,'sha':lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(helper_path),'exec'),ns)
seen=[]
class MetadataComplete(Exception):pass

def fresh(ref):
    result=ns['fresh_environment_v3'](ref);seen.append('real_CPU_environment_import_and_compare');return result

def stop():raise MetadataComplete('Stop before mount/GPU/model/trainer checks')
policy={'allowed_hosts':{'training-vlawm-tsinghua-zt6':{'aliases':['training-vlawm-tsinghua-zt6-worker-0']}},
    'allowed_aliases':['training-vlawm-tsinghua-zt6-worker-0'],'global_dispatch_prohibitions':{}}
auth={'allowed_hosts':['training-vlawm-tsinghua-zt6']}
helpers={'check_route_time_v3':lambda *a:None,'cpu_capacity_v3':lambda:{'quota_us':1600,'period_us':100},
    'fresh_environment_v3':fresh,'shared_lock_mount_v3':stop}
ep=SimpleNamespace(DISPATCH_POLICY='/synthetic-policy',AUTHORIZATION='/synthetic-auth',mature_helpers=lambda:helpers)
m.helper=lambda:ep;m.pinned=lambda ref:policy if ref['path']=='/synthetic-policy' else auth
m.socket.gethostname=lambda:'training-vlawm-tsinghua-zt6-worker-0'
m.os.sched_getaffinity=lambda pid:set(range(32))
binding={'dispatch_policy':{'path':'/synthetic-policy'},'authorization':{'path':'/synthetic-auth'},
    'environment':{'path':str(reference),'sha256':hashlib.sha256(reference.read_bytes()).hexdigest()},
    'route':{'host':'training-vlawm-tsinghua-zt6','hostname':'training-vlawm-tsinghua-zt6-worker-0',
        'gpus':list(range(8)),'gpu_uuids':['explicit-fixture-'+str(i) for i in range(8)]}}
try:m.route_checks(binding,initial=True)
except MetadataComplete:
    assert seen==['real_CPU_environment_import_and_compare']
    loaded=Path(sys.modules['tools.st_world_policy.training_config'].__file__).resolve()
    assert loaded==m.RUNTIME/'tools/st_world_policy/training_config.py'
    print(json.dumps({'status':'PASS_COLD_METADATA_IMPORT','module':str(loaded),'GPU_calls':0,'trainer_calls':0,'model_loads':0}))
except ModuleNotFoundError as error:
    print(json.dumps({'status':'REPRODUCED_OLD_COLD_IMPORT_FAILURE','error':str(error),'GPU_calls':0,'trainer_calls':0,'model_loads':0}))
    sys.exit(2)
else:raise AssertionError('Probe must stop at the CPU metadata boundary')
