"""Export original DiT plans from immutable current-only features; never open future labels."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from tools.joint_world.extract_conditions import CACHE_FIELDS
from tools.joint_world.planner_runtime import CachedCurrentPlanner,file_sha256,load_original_head,predict
from tools.structured_world_v1p1.budget import start,record
from tools.structured_world_v1p1.reaudit_metrics import write_csv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['cache','base-checkpoint','output','ledger','run-id']:p.add_argument('--'+key,required=True)
    p.add_argument('--bridge-checkpoint');p.add_argument('--seed',type=int,default=20260926)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args()
    if not 0<=a.shard<a.shards:raise ValueError('Invalid shard')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/'predictions').mkdir(exist_ok=True)
    completion=out/f'manifest_{a.shard}.json'
    if completion.exists():raise FileExistsError('Use a fresh evaluation output/run ID')
    manifest=json.loads((Path(a.cache)/'manifest.json').read_text());upstream=manifest['identity']
    if not upstream['frozen_upstream'] or upstream['targets_loaded']:raise ValueError('Not a permitted current cache')
    identity={'arguments':vars(a),'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'cache_identity':manifest['identity_sha256'],'original_checkpoint_sha256':upstream['baseline_checkpoint_sha256'],
              'bridge_checkpoint_sha256':file_sha256(a.bridge_checkpoint) if a.bridge_checkpoint else None,
              'candidates':1,'sampling_steps':10,'scorer':None,'world_targets_opened':False,
              'latency_scope':'cached current features through joint graph and original DiT; excludes image/Qwen/provider cost'}
    start(a.ledger,a.run_id,0,identity);rows=[]
    try:
        from infer import deal_action_1225
        head,config=load_original_head(a.base_checkpoint,upstream['baseline_checkpoint_sha256'])
        planner=None
        if a.bridge_checkpoint:
            saved=torch.load(a.bridge_checkpoint,map_location='cpu',weights_only=False)
            source=json.loads((Path(saved['identity']['arguments']['cache'])/'manifest.json').read_text())['identity']
            for key in ['world_checkpoint_sha256','baseline_checkpoint_sha256','world_config','sensor_contract']:
                if source[key]!=upstream[key]:raise ValueError('Planner/current-feature identity mismatch: '+key)
            planner=CachedCurrentPlanner(config.framework.qwenvl.vl_hidden_dim,saved['identity']['graph_config']).cuda().eval()
            planner.load_state_dict(saved['model'],strict=True);planner.requires_grad_(False)
        records=manifest['records'][a.shard::a.shards]
        if len({r['token'] for r in manifest['records']})!=len(manifest['records']):raise ValueError('Duplicate scenes')
        for row in records:
            token=row['token'];result={'token':token,'status':'ok'}
            if not record(a.ledger,a.run_id,0):raise RuntimeError('Budget reached')
            try:
                path=Path(a.cache)/(token+'.pt');destination=out/'predictions'/(token+'.npz')
                if destination.exists():raise FileExistsError('Refusing to overwrite a trajectory')
                if file_sha256(path)!=row['sha256']:raise ValueError('Changed current features')
                c=torch.load(path,map_location='cpu',weights_only=True)
                if set(c)!=CACHE_FIELDS or c['token']!=token or c['identity_sha256']!=row.get('identity_sha256',manifest['identity_sha256']):
                    raise ValueError('Current-feature schema mismatch')
                torch.cuda.synchronize();begin=time.perf_counter()
                with torch.inference_mode():
                    if planner is not None:actions,joint=predict(head,planner,{'cache':c},a.seed)
                    else:
                        native=c['native_actions'].cuda()
                        value=int.from_bytes(hashlib.sha256(f'{a.seed}:{token}'.encode()).digest()[:4],'little')
                        gen=torch.Generator(device=native.device).manual_seed(value)
                        noise=torch.randn(1,head.config.action_horizon,head.config.action_dim,device=native.device,dtype=native.dtype,generator=gen)
                        with torch.autocast('cuda',dtype=torch.float32):actions=head.predict_action(native.float(),initial_noise=noise).cpu().numpy()
                        joint=None
                torch.cuda.synchronize();result['cached_policy_latency_seconds']=time.perf_counter()-begin
                trajectory=deal_action_1225(actions,act_norm=int(config.datasets.vla_data.act_norm))[0]
                if trajectory.shape!=(8,3) or not np.isfinite(trajectory).all():raise ValueError('Invalid ego plan')
                payload={'trajectory':trajectory,'normalized_actions':actions[0]}
                if joint is not None:payload['joint_xy']=joint[0]
                np.savez(destination,**payload);result['proposal_sha256']=file_sha256(destination)
            except Exception as error:
                result.update(status='failed',error=repr(error))
            rows.append(result)
            with (out/f'progress_{a.shard}.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            if len(rows)%64==0:print(json.dumps({'completed':len(rows),'requested':len(records),'failed':sum(r['status']!='ok' for r in rows)}),flush=True)
        write_csv(out/f'scenes_{a.shard}.csv',rows)
        summary=dict(identity,samples=len(rows),failed=sum(r['status']!='ok' for r in rows),peak_gpu_bytes=torch.cuda.max_memory_allocated())
        completion.write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
        if summary['failed']:raise RuntimeError('Retained failed scenes; incomplete evaluation')
        record(a.ledger,a.run_id,0,'complete')
    except BaseException:record(a.ledger,a.run_id,0,'failed');raise


if __name__=='__main__':main()
