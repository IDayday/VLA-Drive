"""Small metadata-only proposal checks; no provider/index/tensor creation."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
RUNTIME=Path('/mnt/project/VLA-Drive-st-indexed-profile-runtime-v1-20261017')
sys.path.insert(0,str(RUNTIME))
from tools.st_world_policy.teacher_provider import pin,pinned,validate_profile_recipe,PROFILE_SCHEMA
from tools.st_world_policy.training_config import validate_config
import torch

cfg=json.loads((HERE/'INPUT_PROFILE_CONFIG.json').read_text())
handoff=json.loads((HERE/'HANDOFF.json').read_text())
parent=pinned(handoff['parent_descriptor'])
admission={'schema':PROFILE_SCHEMA,'scope':'PROFILE','parent_descriptor':handoff['parent_descriptor'],
    'source_migration':pin(HERE/'SOURCE_MIGRATION_PROPOSAL.json'),
    'data_manifest':handoff['data_manifest'],'recipe':pin(HERE/'INPUT_PROFILE_CONFIG.json'),
    'reference_recipe':handoff['reference_config'],'max_scenes':256,'max_updates':256,
    'new_population_tensor_qualified':False,'formal_approved':False}
checks=[]
validate_config(cfg)
checks.append('actual original-policy PROFILE input schema validates without old SMOKE provider')
validate_profile_recipe(admission,cfg)
checks.append('original recipe, exact32, stop64, math/precision/PCA preserved')
for name,mutation in [
    ('learning_rate',lambda c:c['optimizer'].update(lr=2e-5)),
    ('larger_stop',lambda c:c['execution'].update(stop_after=65)),
    ('changed_population',lambda c:c['execution'].update(sample_limit=33)),
    ('precision',lambda c:c['space_time']['teacher'].update(compute_dtype='bfloat16'))]:
    bad=deepcopy(cfg);mutation(bad)
    try:validate_profile_recipe(admission,bad)
    except ValueError:checks.append('reject '+name)
    else:raise AssertionError('Wrong acceptance: '+name)
bad=deepcopy(cfg)
bad['data']['teacher_root']=None
bad['data']['teacher_provider']=pinned(handoff['reference_config'])['data']['teacher_provider']
try:validate_config(bad)
except ValueError:checks.append('old SMOKE provider cannot be promoted by PROFILE flag')
else:raise AssertionError('SMOKE descriptor promoted')
for file in ('prepare_handoff.py','relocate_config.py','validate_handoff.py'):
    ast.parse((HERE/file).read_text())
subprocess.run(['bash','-n',str(HERE/'PREPARE_COMMAND.sh')],check=True)
assert not torch.cuda.is_initialized()
checks.append('Python AST/bash syntax; CUDA remains uninitialized')
result={'status':'PASS','checks':checks,'check_count':len(checks),'index_opened':False,
    'image_or_tensor_or_weight_read':False,'cuda_initialized':False,'factory_started':False,
    'training_started':False,'profile_measurement_status':'NOT_RUN'}
path=HERE/'VALIDATION.json'
if path.exists():raise FileExistsError('Preserve validation evidence')
path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result))
