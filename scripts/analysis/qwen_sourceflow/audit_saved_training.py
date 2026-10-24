"""Read-only audit of real native chains and a committed optimizer boundary.

No model forward, new rollout, scoring call, parameter update or RNG resampling.
Gaussian arithmetic is checked independently with torch.distributions.Normal;
the source relative clamp remains a proxy, not an exact trajectory likelihood.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import torch
from starVLA.rl.qwen_sourceflow.contracts import fingerprint,file_sha
from starVLA.rl.qwen_sourceflow.recipe import load_recipe
from starVLA.rl.simwam_recipe.source_advantages import advantages
from starVLA.rl.simwam_recipe.recovery_state import validate_boundary,validate_ranks,rollout_digest
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--source',required=True)
    parser.add_argument('--updates',nargs='+',type=int,default=[5000,10000,20000])
    parser.add_argument('--output',required=True);args=parser.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError('Preserve earlier audit evidence')
    torch.set_num_threads(4);run=Path(args.run);source=Path(args.source)
    identity=json.loads((run/'identity.json').read_text());body=dict(identity);seal=body.pop('sha256')
    assert fingerprint(body)==seal and identity['purpose']=='formal' and identity['world']==8
    cfg=load_recipe(run/'resolved_recipe.yaml');assert cfg==identity['recipe']
    assert len(identity['parameter_contract']['trainable_names'])==480
    assert all(file_sha(source/name)==sha for name,sha in identity['source_files'].items())
    assert all(update>0 and update%4==0 for update in args.updates)
    collections=[1]+[u//4 for u in args.updates];collections=sorted(set(collections))
    chosen_updates={4*(c-1)+i for c in collections for i in range(1,5)}
    journals={}
    for rank in range(8):
        with (run/f'metrics_rank{rank}.jsonl').open() as handle:
            journals[rank]={i:json.loads(line) for i,line in enumerate(itertools.islice(handle,max(chosen_updates)),1) if i in chosen_updates}
        assert set(journals[rank])==chosen_updates
    reports=[]
    for collection in collections:
        batches=[torch.load(run/'behavior'/f'collection{collection:06d}'/f'rank{rank}.pt',map_location='cpu',weights_only=False) for rank in range(8)]
        global_rewards=torch.cat([b['rewards'] for b in batches],0)
        maxima=dict(raw_logprob=0.,clamped_logprob=0.,used_mean_logprob=0.,advantages=0.,global_std=0.)
        for rank,b in enumerate(batches):
            assert tuple(b['chain'].shape)==(8,11,8,3) and b['stoch_idx'].tolist()==[7,8,9]
            assert len(b['records'])==8 and all(r['status']=='OK' for r in b['records'])
            assert not any(b[k].requires_grad for k in ['chain','old_logp','rewards','advantages','adv_mask'])
            assert b['behavior_version']['run']==identity['sha256'] and b['behavior_version']['update']==4*(collection-1)
            assert b['chain_identity']==rollout_digest(b)
            assert b['protocol_sha256']==identity['score_protocol']['sha256']
            for inner in range(1,5):
                row=journals[rank][4*(collection-1)+inner]
                assert row['chain_identity']==b['chain_identity'] and row['behavior']==b['behavior_version']
                assert row['inner_completed']==inner and row['collection']==collection
            key='train_reward_source' if cfg['variant']=='R_SOURCE' else 'canonical_PDMS'
            expected_reward=torch.tensor([[r['readouts'][key] for r in b['records']]],dtype=torch.float32)
            torch.testing.assert_close(expected_reward,b['rewards'],atol=0,rtol=0)
            stats=advantages(b['rewards'],all_rewards=global_rewards)
            torch.testing.assert_close(stats['adv_mask'],b['adv_mask'],atol=0,rtol=0)
            torch.testing.assert_close(stats['advantages'],b['advantages'],atol=2e-7,rtol=1e-6)
            assert abs(float(stats['global_std']-b['global_std']))<1e-12
            maxima['advantages']=max(maxima['advantages'],float((stats['advantages']-b['advantages']).abs().max()))
            maxima['global_std']=max(maxima['global_std'],abs(float(stats['global_std']-b['global_std'])))
            old=b['old_probability'];std=old['std'].reshape(1,3,1,1)
            assert bool((std>0).all()) and torch.isfinite(b['chain']).all()
            next_state=b['chain'].index_select(1,b['stoch_idx']+1)
            raw=torch.distributions.Normal(old['mean'],std).log_prob(next_state)
            peak=-std.log()-.5*math.log(2*math.pi)
            used=torch.minimum(torch.maximum(raw,peak-7.),peak)
            for name,current,stored in [('raw_logprob',raw,old['raw_gaussian_logprob']),
                    ('clamped_logprob',used,old['source_clamped_logprob']),
                    ('used_mean_logprob',used.mean((-1,-2)),b['old_logp'])]:
                torch.testing.assert_close(current,stored,atol=2e-5,rtol=2e-6)
                maxima[name]=max(maxima[name],float((current-stored).abs().max()))
        reports.append(dict(collection=collection,behavior_update=4*(collection-1),groups=8,candidates=64,
            global_raw_reward_std=float(global_rewards.double().std(unbiased=False)),max_abs_errors=maxima,
            four_inner_chain_old_probability_advantage_identities_fixed=True))
    update=max(args.updates);checkpoint=run/'checkpoints'/f'update{update:06d}_inner0'
    complete=json.loads((checkpoint/'COMPLETE').read_text());files=json.loads((checkpoint/'files.json').read_text())
    assert file_sha(checkpoint/'files.json')==complete['files_sha256']
    assert complete['phase']=='READY' and complete['updates']==update and complete['world']==8
    for name,sha in files.items():assert file_sha(checkpoint/name)==sha
    assert json.loads((checkpoint/'identity.json').read_text())==identity
    saved=torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=False);states=saved['optimizer']['state']
    assert len(states)==480 and all(float(s['step'])==update for s in states.values())
    assert saved['scheduler']['last_epoch']==update
    assert all(s['exp_avg'].dtype==s['exp_avg_sq'].dtype==torch.float32 and torch.isfinite(s['exp_avg']).all() and torch.isfinite(s['exp_avg_sq']).all() for s in states.values())
    summaries=[]
    for rank in range(8):
        own=torch.load(checkpoint/f'rank{rank}.pt',map_location='cpu',weights_only=False)
        summaries.append(validate_boundary(own['state'],own['rollout'],identity,saved['optimizer'],saved['scheduler']))
    validate_ranks(summaries)
    result=dict(schema='native_saved_training_audit_v1',status='PASS',analysis_sha256=file_sha(__file__),
        run=str(run),run_identity=identity['sha256'],source=str(source),source_files_sha256=fingerprint(identity['source_files']),
        checked_checkpoint=str(checkpoint),checkpoint_files_sha256=complete['files_sha256'],
        actual_Adam_updates=update,actual_scheduler_updates=update,FP32_finite_optimizer_states=480,
        checked_scene_groups=len(reports)*8,checked_real_candidates=len(reports)*64,
        source_probability_semantics='FP32 per-element Gaussian; source relative clamp then H/D mean; not joint trajectory ratio',
        CPU_vs_CUDA_tolerance=dict(atol=2e-5,rtol=2e-6),collections=reports,
        new_model_forwards=0,new_scoring_calls=0,parameter_updates=0,algorithm_or_run_changed=False)
    atomic_json(output,result);print(json.dumps({k:result[k] for k in ['status','actual_Adam_updates','checked_scene_groups','checked_real_candidates','new_model_forwards']}))


if __name__=='__main__':main()
