"""Actual matched source/canonical qualification: only the training readout differs."""
import argparse
import json
from pathlib import Path
import torch
from starVLA.rl.flow_grpo.transactions import atomic_json


def differences(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        if a.keys()!=b.keys():raise ValueError('Recipe keys differ: '+path)
        return sum((differences(a[k],b[k],path+'.'+k) for k in a),[])
    return [dict(field=path.removeprefix('.'),canonical=a,source=b)] if a!=b else []


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--canonical-run',required=True)
    p.add_argument('--source-run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(2);roots=[Path(a.canonical_run),Path(a.source_run)]
    identities=[json.loads((r/'identity.json').read_text()) for r in roots]
    for identity in identities:
        if identity['schema']!='native_qwen_sourceflow_RL_v1' or identity['purpose']=='formal':
            raise ValueError('Finite actual qualification only')
    for key in ('source_files','SFT_sha256','IL_parent','projected_condition_identity','hard_manifest_sha256','score_protocol'):
        if identities[0][key]!=identities[1][key]:raise ValueError('Unmatched qualification: '+key)
    delta=differences(*(v['recipe'] for v in identities))
    if {d['field'] for d in delta}!={'variant','grpo.reward.progress_weight'}:
        raise ValueError('More than the registered training reward differs: '+repr(delta))
    records=[]
    for rank in range(8):
        batches=[torch.load(r/'behavior/collection000001'/f'rank{rank}.pt',map_location='cpu',weights_only=False) for r in roots]
        for key in ('chain','timesteps','deltas','stoch_idx','old_logp'):
            if not torch.equal(batches[0][key],batches[1][key]):raise ValueError('Initial behavior changed: '+key)
        for batch,readout in zip(batches,('canonical_PDMS','train_reward_source')):
            if batch['qualification_sampling_check']['status']!='PASS':raise ValueError('Actual sampling kernel mismatch')
            target=torch.tensor([[row['readouts'][readout] for row in batch['records']]],dtype=batch['rewards'].dtype)
            if not torch.equal(batch['rewards'],target):raise ValueError('Wrong native training reward')
        for i in range(8):
            if batches[0]['records'][i]['readouts']!=batches[1]['records'][i]['readouts']:
                raise ValueError('Different official feedback for the same actual candidate')
        results=[json.loads((r/f'result_rank{rank}.json').read_text()) for r in roots]
        for result in results:
            if (result['updates']!=4 or result['collections']!=1 or
                result['frozen_SHA_before']!=result['frozen_SHA_after'] or
                result['frozen_observation_before']!=result['frozen_observation_after']):
                raise ValueError('Wrong update/frozen contract')
        records.append(dict(rank=rank,candidate_chain_equal=True,old_probability_equal=True,
            canonical_train_mean=float(batches[0]['rewards'].mean()),source_train_mean=float(batches[1]['rewards'].mean()),
            canonical_readout_mean=float(torch.tensor([x['readouts']['canonical_PDMS'] for x in batches[0]['records']]).mean())))
    record=dict(status='PASS',qualification_only=True,recipe_delta=delta,world=8,groups=8,
        candidates_per_configuration=64,inner_updates_per_configuration=4,ranks=records,
        meaning='Actual same initial native policy/chain/canonical feedback; only selected training reward differs. No driving efficacy conclusion.')
    atomic_json(a.output,record);print(json.dumps(dict(status='PASS',delta=delta)))


if __name__=='__main__':main()
