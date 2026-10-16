"""Full source-RL native state and real-scored chain continuation comparison."""
import argparse
import json
from pathlib import Path
import torch
from scripts.analysis.fm_transfer.compare_resume import equal
from starVLA.rl.simwam_recipe.contracts import file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    inv=json.loads((root/'invocation.json').read_text())
    if inv['status']!='EXECUTION_COMPLETE_REQUIRES_COMPARISON':raise ValueError('Qualification execution not complete')
    out=root/'validation.json'
    if out.exists():raise FileExistsError(out)
    paths=[root/k/'checkpoints/update000004_inner0' for k in ('continuous','resumed')]
    world=json.loads((paths[0]/'COMPLETE').read_text())['world']
    if world not in (2,8):raise ValueError('Only bounded two-rank or actual eight-rank qualification')
    for path in paths:
        seal=json.loads((path/'COMPLETE').read_text());files=json.loads((path/'files.json').read_text())
        if seal['files_sha256']!=file_sha(path/'files.json') or seal['world']!=world:raise ValueError('Wrong qualification seal')
        for name,h in files.items():
            if file_sha(path/name)!=h:raise ValueError('Modified qualification state')
    left,right=[torch.load(p/'optimizer.pt',map_location='cpu',weights_only=False) for p in paths]
    equal(left,right,'all adapters, Adam and scheduler')
    ranks=[]
    for rank in range(world):
        left,right=[torch.load(p/f'rank{rank}.pt',map_location='cpu',weights_only=False) for p in paths]
        equal(left['rng'],right['rng'],'RNG')
        equal(left['rollout'],right['rollout'],'rollout')
        for key,value in left['state'].items():
            if not key.startswith('seconds_'):equal(value,right['state'][key],'state/'+key)
        behaviors=[torch.load(root/k/'behavior/collection000001'/f'rank{rank}.pt',map_location='cpu',weights_only=False)
                   for k in ('continuous','resumed')]
        for key in ('cond_bg','chain','old_logp','advantages','adv_mask','global_std','timesteps','deltas','stoch_idx','rewards','behavior_version'):
            equal(behaviors[0][key],behaviors[1][key],'behavior/'+key)
        for a,b in zip(behaviors[0]['records'],behaviors[1]['records']):
            equal(a['metrics'],b['metrics'],'real canonical feedback');equal(a['readouts'],b['readouts'],'readouts')
        result=json.loads((root/'continuous'/f'result_rank{rank}.json').read_text())
        if result['frozen_SHA_before']!=result['frozen_SHA_after'] or not all(result['changed_adapters'].values()):
            raise ValueError('Frozen weights changed or adapters failed to update')
        ranks.append(dict(rank=rank,RL_updates=result['RL_updates_local'],
            anchor_only_updates=result['anchor_only_updates_local'],
            canonical_rewards=[r['metrics']['score'] for r in behaviors[0]['records']],
            all_adapters_changed=True,frozen_head_and_reference_unchanged=True))
    value=dict(status='PASS',scope=f'{world}-GPU real-scored source update; continuous4 versus split2/resume4',world=world,
        full_adapter_Adam_scheduler_RNG_equal=True,actual_chains_probabilities_advantages_scores_equal=True,
        four_optimizer_updates_per_fresh_collection=True,ranks=ranks,
        logical_candidates_per_continuous_trial=world*8,physical_qualification_candidates=world*16,
        qualification_only=True,driving_effect_claim=False,
        note='A tied local rank retains anchor; gradients still synchronize from the other rank. Endpoint formal RL acceptance is separate.')
    atomic_json(out,value);print(json.dumps(value))


if __name__=='__main__':main()
