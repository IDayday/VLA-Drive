"""Real compact-token/local-label backward and target-leakage checks.

Explicit real REPAIR and local-only cases are selected for engineering tests,
not for formal sampling or reported performance. No base policy is updated.
"""
import argparse
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,check_host,digest,read,source_sha
from tools.decision_local.train_small_model import build_model,forward,tensor_hash,load_predictor
from starVLA.dataloader.natural_local_dataset import NaturalLocalData
from starVLA.dataloader.recovery_plan_dataset import repair_inputs_targets
from starVLA.model.modules.decision_local.minimal_repair_contract import execution_view
from starVLA.model.modules.decision_local.consequence_contract import predicted_relations


def fixed_batch(data, status, device):
    for i,row in enumerate(data.index):
        labels=data._scene_labels(row['token'])
        positions=torch.nonzero(labels['case_status']==status)
        if len(positions):
            anchor,plan=positions[0].tolist()
            old=data.selection
            try:
                data.selection=lambda *args,**kwargs:[(row['token'],anchor,plan,
                    bool(labels['target_eligible'][anchor,plan]),status)]
                return data.batch([i],device),dict(token=row['token'],anchor=anchor,plan=plan,status=status)
            finally:data.selection=old
    raise ValueError('Real engineering population lacks required case status '+str(status))


def camera(model,kind,batch,predictor):
    current,plan,queries,_=repair_inputs_targets(batch)
    values=predicted_relations(predictor(execution_view(current,plan),queries)) if predictor is not None else None
    if kind=='R_CURRENT':values=torch.cat((values[...,:3],values[...,1:3]),-1)
    return model.camera_only(current,plan,queries,values)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('features','p-labels','r-labels','abs-checkpoint','diff-checkpoint','output'):
        p.add_argument('--'+k,required=True,type=Path)
    a=p.parse_args();resources=check_host();started=time.monotonic()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    datasets=[NaturalLocalData(a.features,x) for x in (a.p_labels,a.r_labels)]
    p_logs={r['log'].rsplit('_',2)[0] for r in datasets[0].index}
    r_logs={r['log'].rsplit('_',2)[0] for r in datasets[1].index}
    if p_logs&r_logs:raise ValueError('Engineering P/R recording overlap')
    repair,case=fixed_batch(datasets[1],1,'cuda')
    unknown,unknown_case=fixed_batch(datasets[0],2,'cuda')
    results=[]
    for kind in ('P_ABS','P_DIFF'):
        model=build_model(kind,42).cuda().train()
        raw,loss,target=forward(model,kind,unknown)
        if target.full_repair_valid.any():raise ValueError('Expected local-only sample')
        loss['total'].backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
        if not norm>0:raise AssertionError('Reliable local facts were filtered by full-target eligibility')
        results.append(dict(kind=kind,case=unknown_case,local_valid=int(target.relation_valid.sum()),
            full_target_valid=False,loss=float(loss['total'].detach()),gradient_norm=norm))
        del model,raw,loss,target
    for kind in ('N_NONE','N_CV','N_FUT','R_CURRENT','R_ABS_FUTURE','R_DIFF_FUTURE'):
        model=build_model(kind,42).cuda().train();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.001)
        predictor=None
        if kind.startswith('R_'):
            path=a.diff_checkpoint if kind=='R_DIFF_FUTURE' else a.abs_checkpoint
            predictor,_=load_predictor(path,'P_DIFF' if kind=='R_DIFF_FUTURE' else 'P_ABS')
            frozen_before=tensor_hash(predictor.state_dict())
        history=[]
        for step in range(3):
            optimizer.zero_grad(set_to_none=True);output,loss,target=forward(model,kind,repair,predictor)
            loss['total'].backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
            if not norm>0:raise AssertionError('Real REPAIR produced no gradient')
            history.append(dict(step=step,loss=float(loss['total'].detach()),gradient_norm=norm));optimizer.step()
        model.eval();invariant=None
        if kind=='N_NONE' or kind.startswith('R_'):
            poisoned=dict(repair)
            for key in ('ego','repair_target','relations'):
                poisoned[key]=torch.full_like(repair[key],float('nan'))
            for key in ('common_valid','valid','target_eligible'):
                poisoned[key]=~repair[key]
            poisoned['case_status']=torch.full_like(repair['case_status'],2)
            with torch.no_grad():
                clean=camera(model,kind,repair,predictor);other=camera(model,kind,poisoned,predictor)
            torch.testing.assert_close(clean,other,rtol=0,atol=0);invariant=True
        if predictor is not None:
            if tensor_hash(predictor.state_dict())!=frozen_before or any(p.grad is not None for p in predictor.parameters()):
                raise AssertionError('Frozen predictor changed in R')
        unknown_r,_=fixed_batch(datasets[1],2,'cuda')
        model.zero_grad(set_to_none=True);_,empty,_=forward(model,kind,unknown_r,predictor);empty['total'].backward()
        if empty['total'].item()!=0 or any(p.grad is not None and bool(p.grad.any()) for p in model.parameters()):
            raise AssertionError('UNKNOWN received an invented full-target gradient')
        results.append(dict(kind=kind,case=case,repair_backward=history,camera_label_invariance=invariant,
            unknown_full_target_loss=0.,predictor_frozen=predictor is not None))
        del model,optimizer,predictor,output,loss,target
    atomic_json(a.output,dict(passed=True,scope='bounded engineering only; historical S0 saw these logs',
        formal_training_or_performance_result=False,source_sha=source_sha(ROOT),results=results,
        features=datasets[0].feature_identity['identity'],labels=[d.label_identity['identity'] for d in datasets],
        P_R_recordings_disjoint=True,real_new_base_updates=0,wall_seconds=time.monotonic()-started,
        GPU_process_hours=(time.monotonic()-started)/3600.,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        resources=resources))


if __name__=='__main__':main()
