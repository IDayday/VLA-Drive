"""Read-only conditional fit on the fixed4096 diagnostic intersection with B."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,sys,time
from pathlib import Path
import numpy as np
import torch
w=Path('/mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015');sys.path.insert(0,str(w))
from tools.decision_local.common import atomic_json,check_host,read,digest,identity
from tools.decision_local.train_small_model import build_model,forward
from starVLA.dataloader.recovery_plan_dataset import RecoveryPlanData
from starVLA.dataloader.foresight_dataset import decode_ego
from starVLA.model.modules.decision_local.minimal_repair_contract import CaseStatus
r=Path('/mnt/project/minimal-repair-artifacts/20261015');resources=check_host();started=time.monotonic()
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
data=RecoveryPlanData('/dev/shm/decision-local-minimal-repair-v5-20261015/anchor_train_full_v1',
    '/dev/shm/decision-local-repair-labels-v12-20261015/targets_foldB_v5',reader='mmap_tensors')
diagnostic=read(r/'TRAIN_DIAGNOSTIC_4096.json');tokens={x['token'] for x in diagnostic}
ids=[i for i,row in enumerate(data.index) if row['token'] in tokens]
assert 1000<len(ids)<3000
kinds=('LOG_NONE','LOG_CV','LOG_FUT','MIN_NONE','MIN_CV','MIN_FUT');models={};checkpoints={};stats={}
for kind in kinds:
    path=r/'stageA_v5_seed42'/kind/'epoch_24.pt';saved=torch.load(path,map_location='cpu',weights_only=False)
    assert saved['epoch']==24 and saved['offset']==0 and saved['completed']==9168
    model=build_model(kind,42,512).cuda().eval();model.load_state_dict(saved['model'],strict=True)
    models[kind]=model;checkpoints[kind]=digest(path);stats[kind]={}
with torch.no_grad():
    for epoch in range(8):
        for begin in range(0,len(ids),64):
            if time.monotonic()-started>600:raise TimeoutError('Finite conditional fit diagnostic budget')
            b=data.batch(ids[begin:begin+64],'cuda',epoch=epoch)
            pin=b['plan_in_physical'].float();valid=b['target_eligible'].bool()
            for kind,model in models.items():
                output,_,target=forward(model,kind,b)
                pred=decode_ego(output.float());truth=decode_ego(target.original_ego_GT if kind.startswith('LOG_') else target.repair_target)
                prediction_error=torch.linalg.norm(pred[...,:2]-truth[...,:2],dim=-1).mean(-1)
                input_error=torch.linalg.norm(pin[...,:2]-truth[...,:2],dim=-1).mean(-1)
                modification=torch.linalg.norm(pred[...,:2]-pin[...,:2],dim=-1).mean(-1)
                heading=torch.atan2(torch.sin(pred[...,2]-truth[...,2]),torch.cos(pred[...,2]-truth[...,2])).abs().mean(-1)
                for state,name in ((int(CaseStatus.KEEP),'KEEP'),(int(CaseStatus.REPAIR),'REPAIR')):
                    for native,label in ((None,'all'),(True,'native'),(False,'perturbed')):
                        mask=valid&(b['case_status']==state)
                        if native is not None:mask &= (b['plan_index']==0)==native
                        # One packed scalar transfer; no change to executed inference.
                        values=torch.stack((mask.sum(),prediction_error[mask].sum(),input_error[mask].sum(),
                            modification[mask].sum(),heading[mask].sum(),(prediction_error[mask]<input_error[mask]-1e-5).sum())).double().cpu().tolist()
                        key=name+'/'+label;old=stats[kind].setdefault(key,np.zeros(6));old+=np.asarray(values)
        print(json.dumps({'epochs_observed':epoch+1,'original_scenes':len(ids),'elapsed_seconds':time.monotonic()-started}),flush=True)
result={}
for kind,groups in stats.items():
    result[kind]={}
    for name,v in groups.items():
        n=int(v[0]);result[kind][name]={'eligible_plan_exposures':n,
            **dict(zip(('prediction_target_ADE_m','input_target_ADE_m','mean_modification_m','heading_error_rad','fraction_closer_to_target'),[float(x/n) if n else None for x in v[1:]]))}
atomic_json(r/'TRAINING_CONDITIONAL_FIT_v13.json',dict(source_sha='0471af96e4f769eb2b31b787a68c57be24bebee0',
    script_sha256=digest(__file__),resources=resources,checkpoints=checkpoints,original_scenes=len(ids),
    index_sha256=identity([data.index[i] for i in ids]),label_identity=data.label_identity['identity'],
    observations='First8 registered plan-selection epochs on fixed4096 diagnostic intersection with B; frozen common24 heads',
    repeated_plans_are_not_independent_scenes=True,precision='FP32;TF32false',training_state_updated=False,
    GPU_process_hours=(time.monotonic()-started)/3600.,metrics=result))
