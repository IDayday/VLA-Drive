"""L0/scaled-prior/learned-root single-output evaluation; never best-of-K."""
import argparse
import fcntl
import json
from pathlib import Path
import time
import torch
from starVLA.rl.qwen_flow_exploration.native import NativeEnvironment
from starVLA.rl.qwen_flow_exploration.protocol import read_protocol, require_scene
from starVLA.rl.qwen_flow_exploration.checkpoint import load_state
from starVLA.rl.qwen_flow_exploration.latent.gaussian import sample_roots, joint_kl, joint_entropy
from starVLA.rl.qwen_flow_exploration.latent.policy import LatentNoisePolicy, condition_features
from starVLA.rl.qwen_flow_exploration.latent.normalization import CENTER_ARM,policy_from_identity,assert_center_unchanged
from starVLA.rl.qwen_sourceflow.contracts import fingerprint, file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import seed_for
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--arm', choices=['L0','L0-S','L1','L2','L3',CENTER_ARM], required=True)
    parser.add_argument('--seed', type=int, default=42, help='Training seed identity; inference seeds are in common protocol')
    parser.add_argument('--checkpoint', help='Complete native latent run or explicit committed checkpoint')
    parser.add_argument('--split', choices=['TRAIN','DEV_SELECT','DEV_AUDIT','NAVTEST'], default='DEV_SELECT')
    parser.add_argument('--readout', choices=['sample','mean'], default='sample')
    parser.add_argument('--scale', type=float, default=1.)
    parser.add_argument('--full-split', action='store_true')
    parser.add_argument('--qualification', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--scorer-workers', type=int, default=1)
    args = parser.parse_args()
    protocol = read_protocol(args.protocol)
    learned = args.arm in ('L1','L2','L3',CENTER_ARM)
    if learned != bool(args.checkpoint):
        raise ValueError('Learned arms require a checkpoint; fixed priors must not load one')
    if args.scale not in (.7,1.,1.3) or (args.arm != 'L0-S' and args.scale != 1):
        raise ValueError('Only predeclared fixed noise scales belong to L0-S')
    if args.limit and not args.qualification:
        raise ValueError('No post-result scene subset selection')
    tokens = (protocol['splits'][args.split] if args.full_split or args.split not in protocol['diagnostic']
              else protocol['diagnostic'][args.split])
    if args.limit:
        tokens = tokens[:args.limit]
    if not tokens:
        raise ValueError('Empty evaluation population')
    for token in tokens:
        require_scene(protocol,token,args.split)
    # The mean root is deterministic. Repeating identical output under five
    # nominal seed labels adds no uncertainty or discovery evidence.
    seeds = protocol['evaluation_seeds'] if args.readout == 'sample' else protocol['evaluation_seeds'][:1]
    state, payload = (load_state(args.checkpoint,protocol['sha256'],restore_rng=False) if learned else (None,None))
    if learned and (state['arm'] != args.arm or state['seed'] != args.seed or
                    (payload['identity']['qualification'] and not args.qualification)):
        raise ValueError('Wrong arm/seed or qualification checkpoint in formal evaluation')
    root = Path(args.output).resolve()
    root.mkdir(parents=True,exist_ok=True)
    lock = (root/'owner.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = dict(schema='frozen_root_single_deployment_evaluation_v1',protocol_sha256=protocol['sha256'],
        base_model_sha256=protocol['base_model_sha256'],arm=args.arm,training_seed=args.seed,
        checkpoint_state=state,root_policy_sha256=tensor_digest(payload['policy']) if learned else None,
        split=args.split,tokens_sha256=fingerprint(tokens),scenes=len(tokens),evaluation_seeds=seeds,
        readout='q_sampled_single' if args.readout == 'sample' else 'q_mean_root_single',
        root_scale=args.scale,selection='no oracle/GT/scorer selection',
        root_innovations='original native_qwen_sourceflow_ODE CPU scene-bound namespace',
        qualification=args.qualification)
    if args.arm==CENTER_ARM:
        identity.update(center_normalizer=payload['identity']['center_normalizer'],
                        correction_budget_lock_sha256=payload['identity']['correction_budget_lock_sha256'])
    identity['sha256']=fingerprint(identity)
    if (root/'identity.json').exists():
        if json.loads((root/'identity.json').read_text()) != identity:
            raise ValueError('Changed evaluation scientific identity')
    else:
        atomic_json(root/'identity.json',identity)
    if (root/'COMPLETE.json').exists():
        value=json.loads((root/'COMPLETE.json').read_text())
        if value['identity_sha256'] != identity['sha256']:
            raise ValueError('Wrong completed evaluation')
        print(json.dumps(dict(status='REUSED_COMPLETE',PDMS=value['PDMS'],scenes=value['scenes'])))
        return
    invocation=root/'invocations'/str(time.time_ns())
    invocation.mkdir(parents=True,exist_ok=False)
    env=NativeEnvironment(protocol,trainable='frozen',scorer_workers=args.scorer_workers,
        ledger_path=invocation/'ledger.json',experiment_id=args.arm+'_eval_seed'+str(args.seed))
    rows, reused = [],0
    try:
        policy=None
        if learned:
            policy=policy_from_identity(payload['identity'],protocol,env.device)
            policy.load_state_dict(payload['policy'],strict=True)
            assert_center_unchanged(policy,payload['identity'])
            policy.eval().requires_grad_(False)
        env.set_behavior_identity(dict(experiment=env.experiment_id,root_policy_sha256=identity['root_policy_sha256'],
            checkpoint_state=state,arm=args.arm,scale=args.scale,readout=args.readout))
        for token in tokens:
            required=[(seed,root/'scenes'/f'{token}_seed{seed}.json') for seed in seeds]
            if all(path.exists() for _,path in required):
                for _,path in required:
                    row=json.loads(path.read_text());body=dict(row)
                    if body.pop('sha256')!=fingerprint(body) or row['evaluation_identity']!=identity['sha256']:
                        raise ValueError('Changed reused single-output record')
                    rows.append(row);reused+=1
                continue
            condition=env.condition(token,args.split)
            params=policy(condition_features(condition)).detached() if learned else None
            for seed,path in required:
                if path.exists():
                    row=json.loads(path.read_text());body=dict(row)
                    if body.pop('sha256')!=fingerprint(body) or row['evaluation_identity']!=identity['sha256']:
                        raise ValueError('Changed reused single-output record')
                    rows.append(row);reused+=1;continue
                rng=torch.Generator(device='cpu').manual_seed(seed_for('native_qwen_sourceflow_ODE',seed,token))
                if learned:
                    noise=sample_roots(params,1,rng)[:,0] if args.readout=='sample' else params.mean
                else:
                    noise=(torch.randn((1,8,3),generator=rng)*args.scale if args.readout=='sample' else torch.zeros(1,8,3)).to(env.device)
                action=env.sample_ode(condition,noise)
                metadata=dict(root=noise[0].cpu().tolist(),readout=identity['readout'],single_candidate=True,
                              root_scale=args.scale,root_policy_sha256=identity['root_policy_sha256'])
                if params is not None:
                    metadata.update(mean=params.mean[0].cpu().tolist(),log_std=params.log_std[0].cpu().tolist(),
                                    prior_KL_joint=float(joint_kl(params,params.standard_prior())[0]),
                                    entropy_joint=float(joint_entropy(params)[0]))
                    env.ledger.add(latent_network_scene_forwards=1/len(seeds))
                row=env.score_actions(token,action,args.split,proposal_source=args.arm+'_'+args.readout,
                    latent_or_proposal_parameters=[metadata],seed=seed)[0]
                row['evaluation_identity']=identity['sha256'];row['sha256']=fingerprint(row)
                path.parent.mkdir(exist_ok=True)
                atomic_json(path,row);rows.append(row)
            atomic_json(root/'progress.json',dict(scene_seed_records=len(rows),expected=len(tokens)*len(seeds)))
        by_scene={token:[] for token in tokens}
        for row in rows:
            by_scene[row['scene_token']].append(row)
        if any(len(value)!=len(seeds) for value in by_scene.values()):
            raise ValueError('Missing seeds/scenes cannot vanish from the denominator')
        means={key:sum(sum(row['native_subscores'][key] for row in value)/len(value)
                       for value in by_scene.values())/len(by_scene) for key in
               ['score','no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress','comfort']}
        result=dict(status='COMPLETE',identity_sha256=identity['sha256'],scenes=len(tokens),seed_count=len(seeds),
            logical_single_output_evaluations=len(rows),reused_records=reused,PDMS=100*means['score'],means=means,
            inference='one native shift5 ODE10 per fixed seed; no best-of',readout=identity['readout'],
            qualification=args.qualification,ledger=env.ledger.snapshot(),
            deterministic_mean_seed_repeats_omitted=args.readout=='mean')
        atomic_json(root/'COMPLETE.json',result)
        print(json.dumps({key:result[key] for key in ['status','scenes','seed_count','PDMS','readout']}))
    finally:
        env.close()


if __name__=='__main__':
    main()
