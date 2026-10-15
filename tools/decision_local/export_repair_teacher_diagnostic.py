"""Frozen-label development diagnostic, never a deployable execution policy.

For eligible plans use the already fixed KEEP/REPAIR target. Else retain input
only to keep the entire evaluation denominator. No optimization, score-based
selection, model selection, or teacher recalibration occurs in this tool.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import read,digest,identity,atomic_json,check_host,source_sha
from tools.decision_local.minimal_repair_training import label_semantics
from starVLA.model.modules.decision_local.minimal_repair_contract import CaseStatus
from starVLA.dataloader.foresight_dataset import decode_ego


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('labels','acceptance','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--distribution',choices=('NATURAL','RECOVERY'),required=True)
    p.add_argument('--max-seconds',type=float,required=True);a=p.parse_args()
    resources=check_host();start=time.monotonic();cpu=time.process_time();torch.set_num_threads(1)
    meta=read(a.labels/'identity.json');acceptance=read(a.acceptance)
    if meta['split']!='dev' or len(meta['index'])!=1696 or not (a.labels/'COMPLETE.json').exists():
        raise ValueError('Only the complete fixed development population')
    if not acceptance['stage_a_training_allowed'] or acceptance['label_semantics']!=label_semantics(meta):
        raise ValueError('Teacher diagnostic must use the already accepted frozen training-side definition')
    if a.output.exists():raise FileExistsError('Preserve completed or failed diagnostic')
    a.output.mkdir(parents=True)
    for name in ('predictions','records'):(a.output/name).mkdir()
    contract={'schema':'minimal_repair_teacher_diagnostic_v1','index':meta['index'],
        'label_identity':meta['identity'],'acceptance_sha256':digest(a.acceptance),
        'source_sha':source_sha(ROOT),'tool_sha256':digest(__file__),'distribution':a.distribution,
        'sampling_seed':42,'mode':'offline_privileged_target_reference_NOT_camera_only',
        'unknown_handling':'input retained for denominator only; not a deployable GT gate',
        'interpretation':'Given frozen target generator; not a universal oracle upper bound',
        'target_or_method_selection_using_this_score':False}
    ident=identity(contract);atomic_json(a.output/'identity.json',{'identity':ident,**contract})
    states={status.name:0 for status in CaseStatus}
    for row in meta['index']:
        if time.monotonic()-start>a.max_seconds:raise TimeoutError('Finite read-only diagnostic budget')
        token=row['token'];receipt=read(a.labels/'records'/(token+'.json'));path=a.labels/'scenes'/(token+'.npz')
        if receipt['identity']!=meta['identity'] or digest(path)!=receipt['sha256']:raise ValueError('Frozen labels changed')
        i=0 if a.distribution=='NATURAL' else 1+int(hashlib.sha256((token+':fixed-recovery-v1').encode()).hexdigest()[:8],16)%4
        with np.load(path,allow_pickle=False) as data:
            qin=data['plan_in_physical'][0,i].copy();status=CaseStatus(int(data['case_status'][0,i]))
            eligible=bool(data['target_eligible'][0,i])
            if eligible!=(status in (CaseStatus.KEEP,CaseStatus.REPAIR)):raise ValueError('Invalid target eligibility')
            final=decode_ego(torch.from_numpy(data['repair_target'][0,i])).numpy() if status==CaseStatus.REPAIR else qin.copy()
        if not np.isfinite(final).all():raise ValueError('Nonfinite target cannot be hidden')
        dest=a.output/'predictions'/(token+'.npz')
        with dest.open('wb') as stream:np.savez_compressed(stream,q0=qin,q_final=final)
        atomic_json(a.output/'records'/(token+'.json'),{'identity':ident,'token':token,
            'sha256':digest(dest),'plan_index':i,'offline_case_status':status.name})
        states[status.name]+=1
    atomic_json(a.output/'COMPLETE.json',{'identity':ident,'scenes':len(meta['index']),
        'case_counts':states,'CPU_process_seconds':time.process_time()-cpu,
        'elapsed_seconds':time.monotonic()-start,'resources':resources,
        'no_new_model_or_optimization':True,'not_a_deployable_policy':True})
    print(json.dumps({'scenes':len(meta['index']),'case_counts':states}))


if __name__=='__main__':main()
