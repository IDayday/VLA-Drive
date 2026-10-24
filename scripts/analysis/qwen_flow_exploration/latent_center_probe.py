"""CPU-only, saved-data qualification of one proposed TRAIN mean centering.

No generator forward, scorer, fresh samples or formal run. The asset is only a
TRAIN512 feature mean; qualification policy weights are never exported.
"""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import time
import torch
from starVLA.rl.qwen_flow_exploration.protocol import read_protocol, ROOT
from starVLA.rl.qwen_flow_exploration.checkpoint import load_state
from starVLA.rl.qwen_flow_exploration.latent.policy import LatentNoisePolicy
from starVLA.rl.qwen_flow_exploration.latent.gaussian import joint_log_prob
from starVLA.rl.qwen_flow_exploration.latent.training import train_collected_batch
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest
from starVLA.rl.simwam_recipe.checkpoint import atomic_torch
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_sourceflow.contracts import fingerprint, file_sha


def signal_stats(policy, features):
    with torch.no_grad():
        normalized=policy.network[0](features)
        pre=policy.network[1](normalized)
        act=policy.network[2](pre)
        q=policy(features)
        return dict(first_preactivation_RMS=float(pre.square().mean().sqrt()),
                    first_preactivation_between_scene_RMS=float(pre.std(0,unbiased=False).square().mean().sqrt()),
                    first_tanh_saturation_gt099_fraction=float((act.abs()>.99).float().mean()),
                    first_tanh_between_scene_RMS=float(act.std(0,unbiased=False).square().mean().sqrt()),
                    root_mean_total_RMS=float(q.mean.square().mean().sqrt()),
                    root_mean_between_scene_RMS=float(q.mean.std(0,unbiased=False).square().mean().sqrt()),
                    root_logstd_between_scene_RMS=float(q.log_std.std(0,unbiased=False).square().mean().sqrt()),
                    prior_KL_joint=float((.5*(q.mean.square()+torch.exp(2*q.log_std)-1-2*q.log_std)).flatten(1).sum(1).mean()))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--protocol',default=str(ROOT/'common_protocol.json'))
    parser.add_argument('--run',default=str(ROOT/'runs/L1_seed42'))
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if torch.cuda.is_initialized():
        raise RuntimeError('This qualification is CPU-only; do not initialize CUDA')
    torch.set_num_threads(2)
    start=time.monotonic();protocol=read_protocol(args.protocol);run=Path(args.run);output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    identity=json.loads((run/'identity.json').read_text())
    if identity['arm']!='L1' or identity['qualification'] or identity['protocol_sha256']!=protocol['sha256']:
        raise ValueError('Use the registered completed L1 behavior evidence')
    _,init=load_state(next((run/'checkpoints').glob('round0000_update000000_inner0_*')),protocol['sha256'],restore_rng=False)
    features={};visits={};repeat_error=0.;first_batch=None;source_batches=[]
    for collection in range(1,65):
        path=next((run/'checkpoints').glob(f'round{collection:04d}_update{4*(collection-1):06d}_inner0_*'))
        state,payload=load_state(path,protocol['sha256'],restore_rng=False)
        if state['phase']!='ROLLOUT_COMMITTED' or state['next_inner']!=0:
            raise ValueError('Expected the actual pre-inner collection boundary')
        batch=payload['batch'];record_path=Path(payload['batch_records_path'])
        if file_sha(record_path)!=payload['batch_records_sha256']:
            raise ValueError('Behavior record changed before mean calibration')
        rows=json.loads(record_path.read_text());tokens=[rows[i]['scene_token'] for i in range(0,len(rows),16)]
        if len(tokens)!=16 or len(rows)!=256 or len(batch.features)!=16 or len(set(tokens))!=16:
            raise ValueError('Invalid saved scene/feature correspondence')
        for i,token in enumerate(tokens):
            if any(row['scene_token']!=token or row['split']!='TRAIN' for row in rows[16*i:16*(i+1)]):
                raise ValueError('Saved features include foreign scene/split')
            x=batch.features[i].float().cpu()
            if token in features:
                error=float((features[token]-x).abs().max());repeat_error=max(repeat_error,error)
                if not torch.equal(features[token],x):
                    raise ValueError('Frozen condition differs between the two TRAIN visits: '+str(error))
            else:
                features[token]=x.clone()
            visits[token]=visits.get(token,0)+1
        source_batches.append(dict(collection=collection,payload_sha256=file_sha(path/'payload.pt'),records_sha256=file_sha(record_path)))
        if collection==1:first_batch=copy.deepcopy(batch)
    if set(features)!=set(protocol['splits']['TRAIN']) or len(features)!=512 or set(visits.values())!={2}:
        raise ValueError('Mean must cover exactly the registered TRAIN512, once per scene')
    x=torch.stack([features[token] for token in protocol['splits']['TRAIN']])
    mean=x.double().mean(0).float()
    normalization=dict(schema='TRAIN512_mean_center_then_existing_LayerNorm_v1',
                       protocol_sha256=protocol['sha256'],base_model_sha256=protocol['base_model_sha256'],
                       TRAIN_tokens_sha256=fingerprint(protocol['splits']['TRAIN']),scenes=512,
                       feature_dim=x.shape[1],fit='FP64 mean of one verified frozen feature vector per unique TRAIN scene, storedFP32',
                       mean_tensor_sha256=tensor_digest({'mean':mean}),std_whitening=False,
                       immutable_mean=True,existing_LayerNorm_eps=1e-5,
                       no_DEV_or_test_statistics=True,qualification_only=True)
    normalization['sha256']=fingerprint(normalization)
    atomic_torch(output/'TRAIN512_feature_mean.pt',dict(mean=mean,identity=normalization))
    atomic_json(output/'normalizer_identity.json',normalization)
    original_digest=first_batch.digest()
    stats={};source_density_difference=0.
    for name,center in [('unchanged',False),('CENTER_qualification_only',True)]:
        policy=LatentNoisePolicy(x.shape[1]).cpu().eval();policy.load_state_dict(init['policy'],strict=True)
        batch=copy.deepcopy(first_batch)
        if center:batch=replace(batch,features=(batch.features-mean).detach())
        # CPU qualification uses the same mathematical N(0,I) density evaluated
        # on CPU. Preserve and report any original GPU reduction rounding gap.
        cpu_old=joint_log_prob(batch.old_parameters,batch.roots).detach()
        source_density_difference=max(source_density_difference,float((cpu_old-batch.old_joint_logp).abs().max()))
        batch=replace(batch,old_joint_logp=cpu_old,behavior_digest=tensor_digest(policy.state_dict()))
        current=policy(batch.features)
        assert torch.equal(current.mean,batch.old_parameters.mean) and torch.equal(current.log_std,batch.old_parameters.log_std)
        assert torch.equal(joint_log_prob(current,batch.roots),batch.old_joint_logp)
        roots=batch.roots.clone();rewards=batch.rewards.clone();advantages=batch.advantages.clone()
        optimizer=torch.optim.Adam(policy.parameters(),lr=1e-4)
        schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda index:min(1.,(index+1)/8))
        before=signal_stats(policy,x-mean if center else x)
        history=train_collected_batch(policy,optimizer,batch,inner_epochs=4,clip_range=.2,prior_coefficient=.01,
                                      checkpoint_callback=lambda *_:schedule.step())
        assert torch.equal(roots,batch.roots) and torch.equal(rewards,batch.rewards) and torch.equal(advantages,batch.advantages)
        assert history[0]['ratio_min']==history[0]['ratio_max']==1.
        assert len(history)==4 and schedule.last_epoch==4
        stats[name]=dict(initial_root_distribution_exactly_N0I=True,first_inner_ratio_exactly1=True,
                         old_roots_rewards_advantages_unchanged=True,optimizer_updates=4,
                         before=before,after=signal_stats(policy,x-mean if center else x),
                         gradient_norms=[row['gradient_norm'] for row in history],
                         actual_learning_rates=[row['learning_rates_used'] for row in history],
                         final_parameter_gradient_norms=history[-1]['parameter_gradient_norms'])
    assert first_batch.digest()==original_digest
    result=dict(schema='latent_CENTER_saved_data_feasibility_v1',status='PASS',qualification_only=True,
                new_rollouts=0,new_scoring_calls=0,new_GPU_tasks=0,policy_weights_exported=False,
                actual_training_source_identity=identity['sha256'],normalizer=normalization,
                normalizer_asset_sha256=file_sha(output/'TRAIN512_feature_mean.pt'),
                exact_repeat_feature_max_error=repeat_error,verified_unique_scenes=512,verified_scene_visits=1024,
                source_batch_identity_sha256=fingerprint(source_batches),
                CPU_vs_saved_GPU_initial_joint_logprob_max_abs=source_density_difference,
                same_input_four_inner_probe=stats,wall_seconds=time.monotonic()-start,
                interpretation='Bounded feasibility and conditioning arithmetic only; no claim CENTER improves PDMS or remains nonsaturated after a full run',
                proposed_newrun='L1_CENTER_V1 only if deployment evidence justifies it; fresh q0 and Adam, unchanged64round/256update budget, all other L1 settings fixed; no launch authorization consumed here')
    result['sha256']=fingerprint(result);atomic_json(output/'qualification.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
