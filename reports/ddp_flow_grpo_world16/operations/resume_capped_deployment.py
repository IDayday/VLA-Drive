"""Hand over our finishing step100 evaluation to the qualified paired controller."""
import json
import os
from pathlib import Path
import sys
import time
root=Path('/mnt/project/DriveDreamer-Policy-paired')
os.chdir(root)
output=root/'runs/resource_reallocation_v3'
evaluation=root/'runs/paired_full_navtrain_world16/frozen_visual_evaluation/step100_rl_dev_seed42'
deadline=time.monotonic()+900
print('WAITING_FOR_OWN_FINISHING_STEP100_EVALUATION',flush=True)
while not (evaluation/'COMPLETE').is_file():
    if time.monotonic()>deadline:raise TimeoutError('own evaluation handover deadline')
    time.sleep(3)
print('RESUMING_QUALIFIED_CAPPED_PAIRED_TRAINING',flush=True)
os.environ.update(PYTHONPATH=f'{root}/navsim:{root}',OMP_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
command=[sys.executable,'-m','scripts.cluster_flow_grpo.paired','--plan',
         'configs/cluster_flow_grpo/paired_world16.json','--resume','--migrate-deployment']
(output/'formal_launch.json').write_text(json.dumps({'pid':os.getpid(),'time':time.time(),'command':command},indent=2))
os.execv(sys.executable,command)
