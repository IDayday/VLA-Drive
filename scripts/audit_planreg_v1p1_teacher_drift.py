#!/usr/bin/env python3
"""Compare real trained teacher targets with the shared-initialized teacher."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from planreg_audit_runtime import load_formal_training_agent, collate_samples, to_device_non_paths
from navsim.planning.training.dataset import load_feature_target_from_pickle


def main():
    p=argparse.ArgumentParser()
    for name in ('config','checkpoint','runtime','shared-init','cache-root','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    _,agent,report=load_formal_training_agent(args.config,args.checkpoint,
                                              device=torch.device('cuda:0'),compute_dtype='bfloat16')
    runtime=json.loads(args.runtime.read_text())
    samples=[]
    for item in runtime['samples']:
        directory=args.cache_root/item['log']/item['token']
        samples.append((load_feature_target_from_pickle(directory/'internvl_feature.gz'),
                        load_feature_target_from_pickle(directory/'trajectory_target_planreg_wm_v1.gz')))
    features,targets=collate_samples(samples)
    features=to_device_non_paths(features,torch.device('cuda:0'))
    targets=to_device_non_paths(targets,torch.device('cuda:0'))
    teacher=agent.ema_register_target
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        trained_current,trained_future,_=agent._encode_ema_register_targets(features,targets,len(samples))
        shared=torch.load(args.shared_init,map_location='cpu',weights_only=True)['trainable_state_dict']
        copies=dict(teacher.named_parameters())
        master_sq=student_sq=0.
        student=teacher.student_parameters(agent.backbone)
        for name,master in teacher.master.tensors().items():
            key=('backbone.model.'+name if name.startswith('vision_model.') else 'backbone.'+name)
            initial=shared[key].to(master.device)
            master_sq+=float((master-initial).float().square().sum())
            student_sq+=float((master-student[name]).float().square().sum())
            torch.testing.assert_close(copies[name],master.to(copies[name].dtype),rtol=0,atol=0)
            # Audit-only in-memory forward copy reset; checkpoint and FP32 master
            # are never edited or saved after this counterfactual intervention.
            copies[name].copy_(initial)
        initial_current,initial_future,_=agent._encode_ema_register_targets(features,targets,len(samples))
    report.update(kind='real_teacher_online_drift',master_initial_l2=master_sq**.5,
                  master_student_l2=student_sq**.5,master_forward_copy_exact=True,
                  current_target_change_rms=float((trained_current-initial_current).float().square().mean().sqrt()),
                  future_target_change_rms=float((trained_future-initial_future).float().square().mean().sqrt()),
                  source_optimizer_step=int(agent._ema_optimizer_step),checkpoint_modified=False)
    assert report['master_initial_l2']>0 and report['future_target_change_rms']>0
    report['status']='PASS'
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
