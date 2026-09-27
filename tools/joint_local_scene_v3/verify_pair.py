"""Verify a completed common milestone from actual checkpoints and training logs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from tools.joint_local_scene_v3.budget import atomic_json


def parameters_hash(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        digest.update(name.encode());digest.update(value.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def verify(root,seed,epochs):
    registration=json.loads((root/'registration.json').read_text());n=registration['train_scenes'];batch=registration['batch'];updates=registration['updates_per_epoch']*epochs
    assert (n+batch-1)//batch==registration['updates_per_epoch']
    records=None;ends=[];logs=[];initial=[];result={'training_seed':seed,'epochs':epochs,'updates_per_arm':updates,'runs':{}}
    for mode in ('all','mask'):
        run=root/f'formal{seed}_{mode}';manifest=json.loads((run/'manifest.json').read_text())
        first=torch.load(run/'milestones/step_0.pt',map_location='cpu',weights_only=False)
        last=torch.load(run/f'milestones/step_{updates}.pt',map_location='cpu',weights_only=False)
        rows=[json.loads(x) for x in (run/'train.jsonl').read_text().splitlines()][:updates]
        assert len(rows)==updates and last['epoch']==epochs and last['offset']==0 and last['presentations']==n*epochs
        assert last['scheduler']['last_epoch']==updates and last['forward_count']==2*updates
        assert first['identity']==last['identity'] and first['step']==0
        assert last['identity']['training_seed']==seed and last['identity']['schedule_updates']==registration['scheduler_updates']
        assert last['identity']['train_identity']==registration['train_identity'] and last['identity']['holdout_identity']==registration['holdout_identity']
        assert not any(v for k,v in last['supervised'].items() if k.endswith('neighbor_yaw'))
        # Only current manifest tokens are needed to verify every epoch's exact order.
        if records is None:
            ledger=json.loads((root/'budget_ledger.json').read_text())
            entry=next(r for r in ledger['runs'] if r['id']==run.name)
            records=json.loads((Path(entry['identity']['train'])/'manifest.json').read_text())['records']
        expected=[]
        for epoch in range(epochs):
            ids=np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(n)
            assert len(set(ids.tolist()))==n
            for offset in range(0,n,batch):expected.append(hashlib.sha256(json.dumps([records[i]['token'] for i in ids[offset:offset+batch]]).encode()).hexdigest())
        assert expected==[row['data_order_sha256'] for row in rows]
        if mode=='all':assert last['exposures']=={'all_hidden':2*n*epochs,'role':0}
        else:
            assert last['exposures']=={'all_hidden':n*epochs,'role':n*epochs}
            tasks=last['task_totals'];assert tasks['neighbor_requested']*2==n*epochs
            assert tasks['neighbor_requested']==tasks['neighbor_actual']+tasks['ego_only_fallback']
            assert tasks['neighbor_valid_coordinates']>0
            assert all(min(row['role_tasks']['valid_hidden_coordinates'])>0 for row in rows)
        groups={}
        for name,start in first['model'].items():
            end=last['model'][name];assert torch.isfinite(end).all()
            group=name.split('.')[0];groups[group]=groups.get(group,0.)+float((end.double()-start.double()).square().sum())
        assert groups['condition']==0 and groups['blocks']>0 and groups['velocity']>0
        result['runs'][mode]={'source_sha':last['identity']['source']['git_sha'],'checkpoint_sha256':hashlib.sha256((run/f'milestones/step_{updates}.pt').read_bytes()).hexdigest(),'initial_parameters_sha256':parameters_hash(first['model']),'every_epoch_covered_once_in_registered_order':True,'tail_batch':n%batch or batch,'presentations':last['presentations'],'exposures':last['exposures'],'supervised_coordinates':last['supervised'],'task_totals':last['task_totals'],'full_parameter_change_L2':{k:v**.5 for k,v in groups.items()},'frozen_condition_exactly_unchanged':True}
        initial.append(result['runs'][mode]['initial_parameters_sha256']);ends.append(last);logs.append(rows)
    assert initial[0]==initial[1]
    assert ends[0]['identity']['source']==ends[1]['identity']['source']
    assert ends[0]['identity']['config']==ends[1]['identity']['config']
    assert torch.equal(ends[0]['noise_rng'],ends[1]['noise_rng']) and torch.equal(ends[0]['time_rng'],ends[1]['time_rng'])
    assert all(a['noise_time_sha256']==b['noise_time_sha256'] for a,b in zip(*logs))
    qa=json.loads((root/f'formal{seed}_all/holdout_queries.json').read_text());qb=json.loads((root/f'formal{seed}_mask/holdout_queries.json').read_text());assert qa==qb
    result.update(same_initialization=True,same_noise_time_stream=True,same_final_noise_time_rng_state=True,same_queries=True,all_checks_passed=True,optimizer_updates_by_this_verifier=0)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--epochs',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
    atomic_json(a.output,verify(Path(a.campaign),a.seed,a.epochs))


if __name__=='__main__':main()
