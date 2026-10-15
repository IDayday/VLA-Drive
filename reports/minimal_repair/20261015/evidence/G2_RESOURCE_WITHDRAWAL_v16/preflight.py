"""Read-only CPU environment/data qualification for the native G2 migration."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

OPS=Path(__file__).resolve().parent
SOURCE=Path('/mnt/project/VLA-Drive-structured-world-fgtr-io-formal-frozen-20261009')
ART=Path('/mnt/project/structured-world-fgtr-round1-artifacts/20261008')
sys.path.insert(0,str(SOURCE))
import torch
import transformers
import mmcv
from mmcv.ops import MultiScaleDeformableAttention
from starVLA.dataloader.structured_world.dataset import StructuredNAVSIMDataset

assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()=='0c54f1c49e7ed2a76b22826b654d6c72625f5b6d'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE,text=True).strip()
torch.set_num_threads(2)
start=time.monotonic()
data=StructuredNAVSIMDataset(ART/'navsim_train_full_label_v5_centered','/var/tmp/ddp-full-foresight-20260929/student_train_v1',
    dino_root=ART/'navsim_current_C1_only_v1',dino_index='/mnt/project/ddp-full-foresight-study-artifacts/20260929/dino_index_v1',
    expected_dino='bda28c73a00fa2d47df2720dd03007f09d006f34482e406b0418a2112ae6924e',
    image_root='/var/tmp/ddp-full-foresight-20260929/images',allow_debug=False)
samples=[]
for i in [0,127,len(data)//2,len(data)-1]:
    observation,targets=data[i]
    samples.append(dict(index=i,token=data.index[i]['token'],geometry_shape=list(observation['geometry_images'].shape),
                        geometry_sha256=hashlib.sha256(observation['geometry_images'].numpy().tobytes()).hexdigest(),
                        ego_sha256=hashlib.sha256(targets['ego'].numpy().tobytes()).hexdigest(),
                        dino_sha256=hashlib.sha256(targets['current_dino'].numpy().tobytes()).hexdigest()))
qwen=Path(transformers.__file__).parent/'models/qwen3_vl/modeling_qwen3_vl.py'
extensions=list(Path(mmcv.__file__).parent.glob('_ext*.so'));assert len(extensions)==1
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
record=dict(host=socket.gethostname(),versions={k:importlib.metadata.version(k) for k in ['torch','transformers','deepspeed','mmcv','diffusers']},
            cuda=torch.version.cuda,python=sys.version,python_prefix=sys.prefix,qwen_source_sha256=digest(qwen),
            mmcv_ext_sha256=digest(extensions[0]),dataset_identity=data.identity['identity'],current_identity=data.base.identity,
            scenes=len(data),samples=samples,read_seconds=time.monotonic()-start,
            source_sha='0c54f1c49e7ed2a76b22826b654d6c72625f5b6d',
            paths={str(p):str(p.resolve()) for p in [Path('/var/tmp/ddp-full-foresight-20260929/images'),Path('/var/tmp/ddp-full-foresight-20260929/student_train_v1')]})
p=OPS/('PREFLIGHT_'+socket.gethostname()+'.json');p.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['host','versions','cuda','scenes','read_seconds','paths']}),flush=True)
