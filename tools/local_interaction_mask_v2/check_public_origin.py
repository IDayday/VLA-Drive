"""Real current-image public-base initialization and original DiT forward/update benchmark."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from omegaconf import OmegaConf
from starVLA.model.modules.joint_world.public_baseline import PublicQwenBaseline
from starVLA.model.modules.structured_world.policy import StructuredWorldPolicy
from starVLA.model.modules.structured_world.contracts import WorldTargets
from starVLA.model.modules.structured_world.rehab import reference_loss_sums
from tools.local_interaction_mask_v2.budget import Run
from tools.local_interaction_mask_v2.data import current_metadata_from_training_pickle,current_example


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('public-qwen','config','provenance','observations','meta-root','targets','token','output','ledger','run-id'):
        p.add_argument('--'+k,required=True)
    p.add_argument('--updates',type=int,default=2)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    identity={'arguments':vars(a),'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'scope':'Public-only initialization + real one-scene training-cost check; not convergence or planning evidence'}
    with Run(a.ledger,a.run_id,identity,gpu_count=1,max_gpu_hours=.25) as run:
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        cfg=OmegaConf.load(a.config);provenance=json.loads(Path(a.provenance).read_text())
        base=PublicQwenBaseline(a.public_qwen,cfg,provenance,seed=42)
        wc={'enabled':True,'provider':'qwen','token_layout':'append_tail','head_location':'post_qwen','reference_heads':True,
            'agent_tokens':64,'scene_tokens':64,'reader_dim':256,'return_world_features':True,'historical_auxiliary_losses':False}
        torch.manual_seed(142);world=StructuredWorldPolicy(base,wc).cuda()
        record=current_metadata_from_training_pickle(Path(a.meta_root)/(a.token+'.pkl'),a.token)
        (out/'current_only.json').write_text(json.dumps(record,indent=2)+'\n')
        example,observation=current_example(Path(a.observations)/(a.token+'.npz'),record)
        # Only the following LABEL-side calls open future targets.
        target=WorldTargets(**torch.load(Path(a.targets)/'targets'/(a.token+'.pt'),map_location='cuda',weights_only=True))
        from tools.joint_world.planner_runtime import ego_action_target
        data_root=Path(a.meta_root).parent.parent
        actions=ego_action_target(a.token,data_root,act_norm=1)[None].cuda()
        world.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            native=base.native_conditions([example]);appended,pred=world.encode_conditions([example])
        parity=float((native.float()-appended.float()).abs().max())
        if parity>1e-5:raise AssertionError(f'Append-tail native condition changed: {parity}')
        parameters=[p for p in world.parameters() if p.requires_grad]
        opt=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=.01)
        updates=[]
        for step in range(1,a.updates+1):
            if not run.update(step-1):raise RuntimeError('Benchmark budget cap')
            opt.zero_grad(set_to_none=True);torch.cuda.synchronize();started=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):condition,pred=world.encode_conditions([example])
            with torch.autocast('cuda',enabled=False):
                ego=base.action_model(condition.float(),actions,None)
                sums,counts,_=reference_loss_sums(pred,[target])
                loss=ego+sum(sums[k]/max(counts[k],1) for k in ('cls','box'))
            loss.backward()
            gradients={name:float(sum((q.grad.float().square().sum() for q in module.parameters() if q.grad is not None),torch.zeros((),device='cuda')).sqrt())
                       for name,module in [('reader',world.reader),('current_heads',world.heads),('history',base.action_input_model),('DiT',base.action_model)]}
            if not torch.isfinite(loss):raise FloatingPointError('Public foundation nonfinite loss')
            torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize()
            updates.append({'step':step,'seconds':time.perf_counter()-started,'ego_FM':float(ego.detach()),'total_loss':float(loss.detach()),
                            'world_loss_sums':{k:float(v.detach()) for k,v in sums.items()},'world_counts':counts,'gradients':gradients})
            run.update(step);print(json.dumps(updates[-1]),flush=True)
        report={'identity':identity,'public_origin':base.public_origin,'native_append_tail_max_error':parity,
                'trainable_parameters':sum(p.numel() for p in parameters),'original_DiT_parameters':sum(p.numel() for p in base.action_model.parameters()),
                'public_Qwen_frozen':not any(p.requires_grad for p in base.qwen_vl_interface.parameters()),
                'updates':updates,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
                'timing_scope':'singleton real current3images, postQwen Reader/heads and originalDiT backprop+optimizer; initialization excluded',
                'private_driving_weights_loaded':False,'status':'PASS'}
        (out/'PUBLIC_ORIGIN_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
