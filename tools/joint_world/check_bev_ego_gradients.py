"""Measure original ego FM gradients into a trained BEV path, with no auxiliary loss/update."""
import argparse
import json
from pathlib import Path
import subprocess

import torch

from tools.joint_world.bev_cache import BEVFeatureStore, batch_bev
from tools.joint_world.planner_runtime import CachedCurrentPlanner, file_sha256, load_original_head, graph_noise, ego_action_target
from tools.joint_world.train_graph import current_batch
from tools.structured_world.runtime import seed_all
from tools.structured_world_v1p1.budget import start, record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['checkpoint','previous-checkpoint','cache','bev-index','base-checkpoint','data-root','output','ledger','run-id']:
        p.add_argument('--'+key,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    previous=torch.load(a.previous_checkpoint,map_location='cpu',weights_only=False)
    if not saved['identity']['bev_enabled'] or saved['identity']!=previous['identity'] or previous['step']>=saved['step']:
        raise ValueError('Need ordered checkpoints from the same BEV run')
    identity={'arguments':vars(a),'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'checkpoint_sha256':file_sha256(a.checkpoint),'previous_sha256':file_sha256(a.previous_checkpoint),
              'optimizer_updates':0,'loss':'original ego flow matching only; no world/BEV auxiliary labels loaded'}
    start(a.ledger,a.run_id,0,identity)
    try:
        head,cfg=load_original_head(a.base_checkpoint,saved['identity']['original_checkpoint_sha256'])
        manifest=json.loads((Path(a.cache)/'manifest.json').read_text())
        source=json.loads((Path(saved['identity']['arguments']['cache'])/'manifest.json').read_text())
        for key in ['world_checkpoint_sha256','baseline_checkpoint_sha256','world_config','sensor_contract']:
            if manifest['identity'][key]!=source['identity'][key]:raise ValueError('Changed current feature identity')
        samples=[];store=BEVFeatureStore(a.bev_index)
        for row in manifest['records'][:4]:
            path=Path(a.cache)/(row['token']+'.pt')
            if file_sha256(path)!=row['sha256']:raise ValueError('Changed current cache')
            c=torch.load(path,map_location='cpu',weights_only=True)
            samples.append({'cache':c,'bev':store[row['token']]})
        model=CachedCurrentPlanner(samples[0]['cache']['context'].shape[-1],saved['identity']['graph_config'],True).cuda().train()
        model.load_state_dict(saved['model'],strict=True);model.graph.requires_grad_(False).eval()
        current=current_batch(samples);bev=batch_bev(samples)
        native=torch.cat([s['cache']['native_actions'] for s in samples]).cuda()
        actions=torch.stack([ego_action_target(s['cache']['token'],a.data_root,int(cfg.datasets.vla_data.act_norm)) for s in samples]).cuda()
        current,_=model.fuse_bev(current,bev['features'],bev['coordinates'],bev['observation_support'])
        noise=graph_noise(len(samples),current['actor_features'].shape[1],model.graph.steps,native.device)
        conditions,_,_=model.rollout_condition(native,current,noise)
        seed_all(51);loss=head(conditions,actions,None);loss.backward()
        gradients={};updates={}
        for name in ['graph','graph_to_world','adapter','bev_encoder','bev_fusion','interaction_head']:
            params=list(getattr(model,name).parameters());values=[p.grad.square().sum() for p in params if p.grad is not None]
            gradients[name]=float(torch.stack(values).sum().sqrt()) if values else 0.
            updates[name]=max(float((value-previous['model'][key]).abs().max()) for key,value in saved['model'].items() if key.startswith(name+'.'))
        passed=all(gradients[n]>0 for n in ['graph_to_world','adapter','bev_encoder','bev_fusion'])
        passed=passed and gradients['graph']==0 and gradients['interaction_head']==0 and updates['graph']==0
        report={'identity':identity,'status':'PASS' if passed else 'FAIL','ego_FM_loss':float(loss.detach()),
                'ego_only_gradient_norms':gradients,'parameter_max_changes_between_checkpoints':updates,
                'tokens':[s['cache']['token'] for s in samples],'from_step':previous['step'],'to_step':saved['step'],
                'action_gate':float(model.adapter.gate),'bev_gate':float(model.bev_fusion.gate),
                'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
                'limitation':'Nonzero gradient and parameter changes show a live trainable route, not effective planning improvement.'}
        (out/'EGO_GRADIENTS.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
        if not passed:raise AssertionError('Expected BEV ego gradient/frozen state violated')
        record(a.ledger,a.run_id,0,'complete')
    except BaseException:record(a.ledger,a.run_id,0,'failed');raise


if __name__=='__main__':main()
