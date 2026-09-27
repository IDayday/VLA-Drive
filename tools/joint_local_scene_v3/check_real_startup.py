"""Actual width384 real-data4 vs2+2, plus zero-update state/gradient checks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
from tools.joint_local_scene_v3.budget import BudgetRun,atomic_json
from tools.joint_local_scene_v3.check_resume import equal
from tools.joint_local_scene_v3.campaign import subset
from tools.joint_local_scene_v3.data import AnnotatedCorpus
from tools.joint_local_scene_v3.runtime import batch_scenes,evaluate
from tools.joint_local_scene_v3.batched_evaluation import evaluate_batched
from tools.joint_local_scene_v3.train_mechanism import code_identity,deterministic_settings
from starVLA.model.modules.joint_scene.flow import JointSceneFlow,flow_loss_sums,normalized_loss,execution_from_joint


def main():
    p=argparse.ArgumentParser()
    for key in ('campaign','train','holdout','config'):p.add_argument('--'+key,required=True)
    p.add_argument('--tag',default='startup');a=p.parse_args();root=Path(a.campaign);out=root/a.tag
    if out.exists():raise FileExistsError('Startup already exists')
    out.mkdir();ledger=root/'budget_ledger.json';commands=[]
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    base=[sys.executable,'-m','tools.joint_local_scene_v3.train_mechanism','--train',a.train,'--train-subset',str(root/'train64.json'),'--holdout',a.holdout,'--config',a.config,'--mode','mask','--updates','4','--schedule-updates','4','--batch','16','--eval-milestones','4','--save-every','2','--ledger',str(ledger),'--device','cuda','--deterministic']
    def invoke(name,extra=(),expect_failure=False):
        cmd=base+['--output',str(out/name),'--run-id',a.tag+'_'+name]+list(extra);commands.append(cmd)
        with (out/(name+'.log')).open('a') as log:proc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        if (proc.returncode!=0)!=expect_failure:raise RuntimeError(f'Unexpected startup exit {proc.returncode}; inspect {name}.log')
    invoke('continuous');invoke('resumed',['--stop-after','2'])
    checkpoint=out/'resumed/checkpoint.pt';before=checkpoint.read_bytes();invoke('resumed',['--resume'],True)
    assert before==checkpoint.read_bytes()
    invoke('resumed',['--resume','--acknowledge-stop'])
    first=torch.load(out/'continuous/checkpoint.pt',map_location='cpu',weights_only=False);second=torch.load(checkpoint,map_location='cpu',weights_only=False)
    checks={k:equal(first[k],second[k]) for k in first}
    def log_values(name):return [{k:v for k,v in json.loads(row).items() if k not in ('seconds','peak_gpu_bytes')} for row in (out/name/'train.jsonl').read_text().splitlines()]
    logs=log_values('continuous');checks['learning_logs']=logs==log_values('resumed')
    initial=torch.load(out/'continuous/milestones/step_0.pt',map_location='cpu',weights_only=False)
    checks['real_parameter_update']=any(not torch.equal(first['model'][k],initial['model'][k]) for k in first['model'])
    checks['condition_frozen']=all(torch.equal(first['model'][k],initial['model'][k]) for k in first['model'] if k.startswith('condition.'))
    checks['neighbor_xy_supervised']=first['supervised']['all_hidden_neighbor_xy']>0 and first['supervised']['role_neighbor_xy']>0
    checks['neighbor_yaw_unsupervised']=first['supervised']['all_hidden_neighbor_yaw']==first['supervised']['role_neighbor_yaw']==0
    checks['roles_present']=first['task_totals']['neighbor_actual']>0
    cfg=json.loads(Path(a.config).read_text())
    with BudgetRun(ledger,a.tag+'_zero_update_checks',{'source':code_identity(),'kind':'actual-model real-data checks','output':str(out/'checks')},1) as budget:
        deterministic_settings();torch.manual_seed(18);torch.cuda.manual_seed_all(18)
        model=JointSceneFlow(**cfg['model']).cuda();model.load_state_dict(first['model'],strict=True);model.condition.requires_grad_(False)
        hook=model.register_forward_pre_hook(lambda *unused:budget.note('real',forwards=1))
        data=subset(AnnotatedCorpus(a.train),root/'train64.json');g,y,v=batch_scenes([data[i] for i in range(8)],'cuda');noise=torch.randn_like(y);time=torch.full((len(y),),.4,device='cuda')
        ego=v.clone();ego[:,1:]=False
        sums,counts=flow_loss_sums(model,y,ego,g.active_actor_mask,noise,time,g);normalized_loss(sums,counts).backward();budget.note('real',backwards=1)
        checks['ego_loss_interaction_gradient']=float(model.blocks[0].actor.in_proj_weight.grad.abs().sum())>0
        checks['gradients_finite']=all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
        with torch.no_grad():
            joint=model.sample(noise,g,sampling_steps=20);poison=noise.clone();poison[:,1:,:,2:]=float('nan')
            checks['unmodeled_yaw_invariance']=torch.equal(joint,model.sample(poison,g,sampling_steps=20)) and joint[:,1:,:,2:].count_nonzero()==0
            known=v.clone();known[:,0]=False
            completed,path=model.sample_conditional(noise,g,y,known,sampling_steps=20,return_path=True)
            checks['known_clamped_every_step']=all(torch.equal(z[known],y[known]) for z in path)
            checks['hidden_truth_no_leak']=torch.equal(completed,model.sample_conditional(noise,g,torch.where(known,y,float('nan')),known,sampling_steps=20))
            execution=execution_from_joint(joint);checks['execution_ego_joint_slot0']=torch.equal(execution['executed_ego_xyyaw'][...,:2],joint[:,0,:,:2])
        reference=evaluate(model,[data[i] for i in range(2)],seed=77,sampling_steps=20,device='cuda')
        batched=evaluate_batched(model,[data[i] for i in range(2)],seed=77,sampling_steps=20,device='cuda',batch_size=16)
        differences=[abs(x[k]-z[k]) for x,z in zip(reference['query_rows'],batched['query_rows']) for k in ('all_hidden_xy_ADE_m','conditional_xy_ADE_m')]
        checks['batched_evaluation_parity_atol_5e_4_m']=max(differences)<5e-4
        checks['same_query_manifest']=reference['query_manifest']==batched['query_manifest']
        atomic_json(out/'batch_parity.json',{'max_ADE_difference_m':max(differences),'declared_absolute_tolerance_m':5e-4,'width':cfg['model']['dim']})
        hook.remove()
    checks={k:bool(v) for k,v in checks.items()}
    report={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'real_optimizer_updates':8,'synthetic_optimizer_updates':0,'real_presentations':first['presentations']+second['presentations'],'model':cfg['model'],'source':code_identity(),'task_totals':first['task_totals'],'supervised':first['supervised'],'commands':commands}
    atomic_json(out/'summary.json',report);print(json.dumps(report,indent=2))
    if not all(checks.values()):raise AssertionError('Real startup verification failed')


if __name__=='__main__':main()
