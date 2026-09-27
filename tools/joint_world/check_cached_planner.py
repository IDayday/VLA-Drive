"""Real GPU original-DiT/ego-label/bridge parity, updates and independent restore."""
import argparse
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import torch

from tools.joint_world.planner_runtime import (CachedCurrentPlanner,ego_action_target,
    file_sha256,graph_noise,load_original_head,predict)
from tools.joint_world.train_graph import load_samples,current_batch
from tools.structured_world.runtime import load_dataset,seed_all
from tools.structured_world_v1p1.budget import start,record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['cache','targets','data-root','base-checkpoint','graph-checkpoint','output','ledger','run-id']:
        p.add_argument('--'+key,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    identity={'arguments':vars(a),'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    start(a.ledger,a.run_id,2,identity);completed=0
    try:
        samples,manifest,_=load_samples(a.cache,a.targets,a.data_root,8);sample=samples[0]
        head,config=load_original_head(a.base_checkpoint,manifest['identity']['baseline_checkpoint_sha256'])
        saved=torch.load(a.graph_checkpoint,map_location='cpu',weights_only=False);cfg=saved['identity']['config']
        seed_all(42);model=CachedCurrentPlanner(sample['cache']['context'].shape[-1],cfg).cuda().eval()
        model.graph.load_state_dict(saved['model'],strict=True);model.graph.requires_grad_(False)
        dataset=load_dataset(SimpleNamespace(model_config=config),manifest['identity']['arguments']['manifest'],a.data_root,1)
        raw=dataset[0];assert raw['token']==sample['cache']['token']
        labels=ego_action_target(raw['token'],a.data_root,int(config.datasets.vla_data.act_norm))
        checks={'released_dataset_action_exact':bool(np.array_equal(labels.numpy(),raw['action']))}
        original,_=predict(head,model,sample,disable_graph=True);zero,_=predict(head,model,sample)
        checks['gate0_original_DiT_exact']=bool(np.array_equal(original,zero))
        # Label data is not an argument of the current-only prediction helper.
        dirty=dict(sample,xy=torch.full_like(sample['xy'],float('nan')),valid=~sample['valid'],target=None)
        changed,_=predict(head,model,dirty)
        checks['poisoned_targets_prediction_exact']=bool(np.array_equal(changed,zero))
        if not all(checks.values()):raise AssertionError(checks)
        current=current_batch([sample]);native=sample['cache']['native_actions'].cuda()
        noise=graph_noise(1,current['actor_features'].shape[1],model.graph.steps,'cuda',cfg.get('sampling_seed',2037))
        params=[v for v in model.parameters() if v.requires_grad];opt=torch.optim.AdamW(params,lr=1e-4)
        gradients=[]
        for step in range(2):
            if not record(a.ledger,a.run_id,completed):raise RuntimeError('Budget exhausted')
            opt.zero_grad(set_to_none=True);condition,_=model.condition_from_frozen_graph(native,current,noise)
            loss=head(condition,labels[None].cuda(),None);loss.backward()
            row={'step':step+1,'loss':float(loss.detach()),'gate_grad':float(model.adapter.gate.grad),
                 'projection_grad':float(model.graph_to_world.weight.grad.norm()),
                 'frozen_graph_no_grad':all(v.grad is None for v in model.graph.parameters()),
                 'frozen_head_no_grad':all(v.grad is None for v in head.parameters())}
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();completed+=1;gradients.append(row)
            record(a.ledger,a.run_id,completed)
        checks['first_gate_gradient_nonzero']=gradients[0]['gate_grad']!=0
        checks['second_projection_gradient_nonzero']=gradients[1]['projection_grad']>0
        checks['frozen_modules_no_gradient']=all(r['frozen_graph_no_grad'] and r['frozen_head_no_grad'] for r in gradients)
        before,joint=predict(head,model,sample);torch.save(model.state_dict(),out/'bridge_state.pt')
        restored=CachedCurrentPlanner(native.shape[-1],cfg).cuda().eval()
        restored.load_state_dict(torch.load(out/'bridge_state.pt',map_location='cuda',weights_only=True),strict=True)
        after,other=predict(head,restored,sample)
        checks['independent_restore_action_exact']=bool(np.array_equal(before,after))
        checks['independent_restore_joint_exact']=bool(np.array_equal(joint,other))
        result={'identity':identity,'checks':checks,'gradients':gradients,'updates':completed,
                'original_head_parameters':sum(p.numel() for p in head.parameters()),
                'trainable_bridge_parameters':sum(p.numel() for p in params),'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
                'status':'PASS' if all(checks.values()) else 'FAIL','scope':'One real scene,2 updates; transfer runner engineering only'}
        (out/'CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
        if not all(checks.values()):raise AssertionError(checks)
        record(a.ledger,a.run_id,completed,'complete')
    except BaseException:record(a.ledger,a.run_id,completed,'failed');raise


if __name__=='__main__':main()
