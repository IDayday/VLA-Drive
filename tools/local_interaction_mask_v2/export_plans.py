"""Multi-GPU current-only plan export; CPU scoring watches atomic files separately."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from tools.local_interaction_mask_v2.planner_runtime import CurrentOnlyCorpus,load_public_head,load_bridge,predict_payloads
from tools.local_interaction_mask_v2.train_foundation import atomic_json
from starVLA.model.modules.joint_world.public_baseline import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('cache','foundation','variants','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--seed',type=int,default=20260926)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--shard',type=int);p.add_argument('--shards',type=int)
    p.add_argument('--shard-offset',type=int,default=0,help='Offset local torchrun ranks when exporting across independent hosts')
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.shard is not None and a.shard_offset:raise ValueError('Use explicit shard or rank offset, not both')
    shard=int(os.environ.get('RANK',0))+a.shard_offset if a.shard is None else a.shard
    shards=int(os.environ.get('WORLD_SIZE',1)) if a.shards is None else a.shards
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)))
    if not 0<=shard<shards or not 1<=a.batch<=128:raise ValueError('Invalid deterministic shard/bounded batch')
    corpus=CurrentOnlyCorpus(a.cache);upstream=corpus.manifest['identity']
    head,config=load_public_head(a.foundation,upstream['foundation_sha256'])
    specs=json.loads(Path(a.variants).read_text())
    if not specs or any('/' in name or not name for name in specs):raise ValueError('Invalid variant names')
    variants={};out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    for name,path in specs.items():
        bridge=None
        if path:bridge,_=load_bridge(path,corpus.manifest['identity_sha256'],config.framework.qwenvl.vl_hidden_dim)
        variants[name]=bridge;bank=out/name;bank.mkdir(exist_ok=True);(bank/'predictions').mkdir(exist_ok=True)
        identity={'schema_version':2,'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'current_identity':corpus.manifest['identity_sha256'],'current_manifest_sha256':sha256(Path(a.cache)/'manifest.json'),
            'foundation_sha256':upstream['foundation_sha256'],'bridge_sha256':sha256(path) if path else None,
            'variant':name,'seed':a.seed,'shard':shard,'shards':shards,'batch':a.batch,'expected_total':len(corpus),
            'candidates':1,'FM_steps':10,'scorer':None,'future_labels_opened':False,
            'precision':'native Qwen conditions/BF16 noise draws converted to original FP32 DiT',
            'latency_scope':'cached current conditions + optional graph + DiT; upstream feature extraction reported separately'}
        ident=bank/f'identity_{shard}.json'
        if ident.exists():
            if not a.resume or json.loads(ident.read_text())!=identity:raise ValueError('Export identity changed; use fresh directory')
        else:atomic_json(ident,identity)
    # Use the repository's original decoder, including its yaw wrapping.
    from starVLA.model.modules.action_model.navsim_decode import deal_action_1225
    rows={name:[] for name in variants};complete=True
    indices=list(range(shard,len(corpus),shards))
    for offset in range(0,len(indices),a.batch):
        if (out/'STOP_REQUESTED').exists():complete=False;break
        batch_indices=indices[offset:offset+a.batch];payloads={};load_errors={}
        for i in batch_indices:
            try:payloads[i]=corpus[i]
            except Exception as error:load_errors[i]=repr(error)
        for name,bridge in variants.items():
            bank=out/name;todo=[]
            for i in batch_indices:
                token=corpus.records[i]['token'];dest=bank/'predictions'/(token+'.npz');record=dest.with_suffix('.json')
                if record.exists():
                    row=json.loads(record.read_text())
                    if not a.resume:raise FileExistsError('Immutable proposal already exists')
                    if row['status']=='ok' and (not dest.exists() or sha256(dest)!=row['proposal_sha256']):raise ValueError('Previously exported proposal changed')
                    rows[name].append(row)
                elif i in load_errors:
                    row={'token':token,'status':'failed','variant':name,'error':load_errors[i]}
                    atomic_json(record,row);rows[name].append(row)
                else:todo.append(i)
            if not todo:continue
            # Preserve original batch membership on partial resume; do not change
            # backend batch shapes/noise by dropping already completed members.
            live=[i for i in batch_indices if i in payloads];batch_error=None
            try:
                torch.cuda.synchronize();start=time.perf_counter()
                with torch.inference_mode():result=predict_payloads(head,bridge,[payloads[i] for i in live],upstream,a.seed)
                torch.cuda.synchronize();seconds=time.perf_counter()-start
                actions=result['normalized_actions'].cpu().numpy()
                trajectories=deal_action_1225(actions,act_norm=1)
            except Exception as error:batch_error=repr(error)
            for i in todo:
                token=corpus.records[i]['token'];dest=bank/'predictions'/(token+'.npz');record=dest.with_suffix('.json')
                row={'token':token,'status':'ok','variant':name}
                try:
                    if batch_error:raise RuntimeError(batch_error)
                    j=live.index(i);trajectory=trajectories[j];payload=payloads[i]
                    if trajectory.shape!=(8,3) or not np.isfinite(trajectory).all():raise ValueError('Invalid actual DiT plan')
                    row.update(cached_batch_wall_seconds=seconds,batch_scenes=len(live),amortized_policy_seconds=seconds/len(live))
                    values={'trajectory':trajectory,'normalized_actions':actions[j]}
                    if result['joint_trajectories_xy'] is not None:
                        values['graph_joint_xy']=result['joint_trajectories_xy'][j].cpu().numpy()
                        values['graph_source_slot_ids']=payload['local_graph']['source_slot_ids'][0].numpy()
                        values['graph_active_actor_mask']=payload['local_graph']['active_actor_mask'][0].numpy()
                    tmp=dest.with_suffix('.tmp')
                    with tmp.open('wb') as f:np.savez(f,**values)
                    tmp.replace(dest);row['proposal_sha256']=sha256(dest)
                except Exception as error:row.update(status='failed',error=repr(error))
                atomic_json(record,row);rows[name].append(row)
        if offset%(a.batch*4)==0:
            print(json.dumps({'shard':shard,'completed_per_variant':len(next(iter(rows.values()))),'assigned':len(range(shard,len(corpus),shards))}),flush=True)
    failures=0
    for name,records in rows.items():
        failed=sum(r['status']!='ok' for r in records);failures+=failed
        atomic_json(out/name/f'shard_{shard}.json',{'status':'complete' if complete else 'paused','completed':len(records),
            'failed':failed,'expected':len(range(shard,len(corpus),shards)),'peak_gpu_bytes':torch.cuda.max_memory_allocated()})
    if shards==1:atomic_json(out/'status.json',{'status':'failed' if failures else ('complete' if complete else 'paused'),'failed':failures})
    # The legacy original decoder import may initialize a distributed group.
    if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
    if failures:raise RuntimeError('Export failures retained; benchmark incomplete')


if __name__=='__main__':main()
