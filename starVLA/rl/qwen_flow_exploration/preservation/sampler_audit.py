"""Read-only terminal SDE audit records and original-source RNG bookkeeping.

No training loss, optimizer, multiplier or alternative sampler is implemented.
The existing native sampling path remains the only policy-generation path.
"""
import hashlib
import json
import os
from pathlib import Path
import uuid
import numpy as np
import torch
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.rl.simwam_recipe.vendor.action_math import ActionGRPOMixin
from starVLA.rl.simwam_recipe.contracts import seed_for
from starVLA.rl.simwam_recipe.checkpoint import atomic_torch
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest
from starVLA.rl.qwen_sourceflow.contracts import fingerprint,file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


NAMESPACE='preservation_same_kernel_reference_SDE'
METRICS=('score','no_at_fault_collisions','drivable_area_compliance',
         'time_to_collision_within_bound','ego_progress','comfort','driving_direction_compliance')
SAFETY=METRICS[1:4]


def validate_budget(lock, protocol, calibration, *, arm, training_seed):
    body=dict(lock)
    if body.pop('sha256')!=fingerprint(body):raise ValueError('Unsealed terminal sampling budget')
    fixed=dict(schema='preservation_terminal_sampler_audit_budget_v1',phase='TERMINAL_AUDIT_LOCKED',
        protocol_sha256=protocol['sha256'],calibration_sha256=calibration['sha256'],
        reference_pool_sha256=calibration['reference_SDE_pool_sha256'],
        TRAIN_tokens_sha256=fingerprint(protocol['splits']['TRAIN'][:128]),scenes=128,G=16,K=10,
        candidate_chunk=4,stochastic_steps=[7,8,9],new_candidates_per_terminal_arm=4096,
        new_candidates_BASE=2048,total_new_candidates=26624,
        reference_rng_seed=42,reference_rng_namespace=NAMESPACE)
    if any(lock.get(key)!=value for key,value in fixed.items()):raise ValueError('Changed terminal audit population/kernel/RNG/budget')
    if lock['arms']!=['BASE','P0','P1','P2','P3','P4','P5'] or training_seed not in lock['training_seeds'] or arm not in lock['arms']:
        raise ValueError('Unregistered terminal policy/seed')
    if protocol['execution']['candidate_chunk']!=4:raise ValueError('Audit must preserve actual sampling/recompute chunk4')
    return lock


def validate_group(rollout,records,token,protocol,*,G=16):
    if (len(records)!=G or rollout['chain'].shape!=(G,11,8,3) or rollout['old_logp'].shape!=(G,3) or
            rollout['rewards'].shape!=(1,G) or list(rollout['stoch_idx'].tolist())!=[7,8,9] or
            rollout['sampling_chunk']!=4 or rollout['token']!=token or rollout['split']!='TRAIN' or
            len(rollout['timesteps'])!=10 or len(rollout['deltas'])!=10):
        raise ValueError('Wrong complete actual source SDE group/state contract')
    for key in ('chain','old_logp','rewards','timesteps','deltas'):
        if not rollout[key].isfinite().all():raise ValueError('Nonfinite actual SDE chain/probability/reward')
    validate_records(records,token,protocol,G=G)
    for i,row in enumerate(records):
        raw=torch.tensor(row['normalized_action'],dtype=torch.float32)
        if (not torch.equal(raw,rollout['chain'][i,-1].float()) or
                abs(float(rollout['rewards'][0,i])-row['raw_reward'])>1e-7):
            raise ValueError('Scored SDE endpoint differs from raw saved chain/reward')


def validate_records(records,token,protocol,*,G=16):
    if len(records)!=G:raise ValueError('Missing actual requested candidates')
    for row in records:
        feedback=row['feedback'];metrics=row['native_subscores']
        if (row['status']!='OK' or row['scene_token']!=token or row['split']!='TRAIN' or
                row['scorer_config_hash']!=protocol['scorer']['sha256'] or
                row['sampler_config_hash']!=fingerprint(protocol['samplers']) or
                row['raw_reward']!=metrics['score'] or
                any(not isinstance(metrics[k],(int,float)) or not 0<=metrics[k]<=1 for k in METRICS) or
                feedback['status']!='OK' or feedback['token']!=token or feedback['log']!=row['log_id'] or
                feedback['protocol_sha256']!=protocol['scorer']['sha256'] or feedback['metrics']!=metrics):
            raise ValueError('Missing/foreign/failure-as-zero canonical SDE feedback')
        raw=torch.tensor(row['normalized_action'],dtype=torch.float32)
        planned=torch.tensor(row['planned_trajectory'],dtype=torch.float64)
        simulated=torch.tensor(row['simulated_trajectory'],dtype=torch.float64)
        if (raw.shape!=(8,3) or planned.shape!=(8,3) or not raw.isfinite().all() or
                not planned.isfinite().all() or not simulated.isfinite().all() or
                not torch.allclose(planned,ActionCodec.denorm_odo(raw).double(),atol=1e-5,rtol=0) or
                row['planned_trajectory']!=feedback['requested_trajectory'] or
                row['simulated_trajectory']!=feedback['simulated_states']):
            raise ValueError('Scored physical consequences and raw policy-chain endpoint disagree')


def validate_ODE_roots(records,roots):
    if len(records)!=len(roots):raise ValueError('Incomplete matched-root ODE group')
    for i,row in enumerate(records):
        parameters=row['latent_or_proposal_parameters']
        if (parameters['candidate_index']!=i or parameters['oracle_selection'] is not False or
                not torch.equal(torch.tensor(parameters['root'],dtype=torch.float32),roots[i].float())):
            raise ValueError('ODE candidate did not use the exact saved source-SDE root')


def reference_rng_book(pool,*,reference_seed=42):
    """Replay only original _randn draws and verify EVERY stored initial root.

    This does not produce any trajectory: the existing source _randn method
    advances four draws per chunk (root + three actual stochastic steps).
    Matching all saved x0 verifies the cumulative original calibration stream.
    """
    if reference_seed!=42:raise ValueError('Initial registered reference stream is fixed seed42')
    rng=torch.Generator(device='cpu').manual_seed(seed_for(NAMESPACE,reference_seed,'unfiltered128'))
    book=[]
    for token,rollout in zip(pool['tokens'],pool['rollouts']):
        if rollout['token']!=token or rollout['sampling_chunk']!=4 or rollout['stoch_idx'].tolist()!=[7,8,9]:
            raise ValueError('Reference source chunk/window identity differs')
        before=rng.get_state().clone();chain=rollout['chain']
        for start in range(0,len(chain),4):
            shape=(min(4,len(chain)-start),8,3)
            root=ActionGRPOMixin._randn(None,shape,torch.device('cpu'),torch.float32,rng)
            if not torch.equal(root,chain[start:start+4,0]):raise ValueError('Reference source initial RNG stream does not match saved roots')
            for _ in rollout['stoch_idx']:
                ActionGRPOMixin._randn(None,shape,torch.device('cpu'),torch.float32,rng)
        book.append(dict(token=token,before=before,after=rng.get_state().clone(),roots_sha256=tensor_digest(chain[:,0])))
    if len(book)!=len(pool['tokens']):raise ValueError('Incomplete reference stream')
    return book


def freeze_and_hash(environment):
    """Freeze entire audit runtime; hash parameters/buffers without a big copy."""
    environment.model.requires_grad_(False).eval();environment.actor.requires_grad_(False).eval()
    if any(p.grad is not None for p in environment.actor.parameters()):raise ValueError('Read-only audit received training gradients')
    return model_hash(environment)


def model_hash(environment):
    digest=hashlib.sha256();seen=set()
    for owner,module in (('model',environment.model),('actor',environment.actor)):
        for name,value in list(module.named_parameters())+list(module.named_buffers()):
            if id(value) in seen:continue
            seen.add(id(value));digest.update((owner+'.'+name).encode())
            cpu=value.detach().cpu().contiguous();digest.update(str(cpu.dtype).encode());digest.update(str(tuple(cpu.shape)).encode())
            digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def publish_group(root,identity,ordinal,token,rollout,records,ODE_records,rng_boundary):
    if tensor_digest(rollout['chain'][:,0])!=rng_boundary['roots_sha256']:
        raise ValueError('Cannot publish changed initial roots')
    validate_ODE_roots(ODE_records,rollout['chain'][:,0])
    root=Path(root);root.mkdir(parents=True,exist_ok=True);final=root/token
    if final.exists():raise FileExistsError('Never overwrite an existing committed audit scene')
    temp=root/(token+'.incomplete.'+uuid.uuid4().hex);temp.mkdir(exist_ok=False)
    payload=dict(schema='terminal_native_SDE_group_v1',identity_sha256=identity['sha256'],ordinal=ordinal,
        token=token,rollout=rollout,records=records,ODE_records=ODE_records,
        RNG_before=rng_boundary['before'],RNG_after=rng_boundary['after'])
    atomic_torch(temp/'payload.pt',payload)
    atomic_json(temp/'COMPLETE',dict(identity_sha256=identity['sha256'],ordinal=ordinal,token=token,
        payload_sha256=file_sha(temp/'payload.pt'),RNG_before_sha256=tensor_digest(rng_boundary['before']),
        RNG_after_sha256=tensor_digest(rng_boundary['after']),candidates=len(records)))
    os.rename(temp,final)


def load_group(root,identity,ordinal,token,rng_boundary,protocol):
    path=Path(root)/token;seal=json.loads((path/'COMPLETE').read_text())
    if (seal['identity_sha256']!=identity['sha256'] or seal['ordinal']!=ordinal or seal['token']!=token or
            seal['candidates']!=16 or file_sha(path/'payload.pt')!=seal['payload_sha256']):
        raise ValueError('Changed/foreign partial audit scene publication')
    payload=torch.load(path/'payload.pt',map_location='cpu',weights_only=False)
    if (payload['identity_sha256']!=identity['sha256'] or payload['ordinal']!=ordinal or payload['token']!=token or
            not torch.equal(payload['RNG_before'],rng_boundary['before']) or
            not torch.equal(payload['RNG_after'],rng_boundary['after']) or
            seal['RNG_before_sha256']!=tensor_digest(payload['RNG_before']) or
            seal['RNG_after_sha256']!=tensor_digest(payload['RNG_after'])):
        raise ValueError('Partial reuse changed original reference stream')
    validate_group(payload['rollout'],payload['records'],token,protocol)
    validate_records(payload['ODE_records'],token,protocol)
    if tensor_digest(payload['rollout']['chain'][:,0])!=rng_boundary['roots_sha256']:
        raise ValueError('Cached audit initial roots do not match original reference')
    validate_ODE_roots(payload['ODE_records'],payload['rollout']['chain'][:,0])
    return payload


def mean_summary(groups):
    if not groups or any(len(group)!=16 for group in groups):raise ValueError('Compare complete G16 means, never best-of')
    native={key:np.asarray([[row['native_subscores'][key] for row in group] for group in groups],dtype=np.float64) for key in METRICS}
    safe=np.logical_and.reduce([native[key]==1 for key in SAFETY])
    rewards=native['score'];tied=np.ptp(rewards,axis=1)==0
    safety_vary=np.logical_or.reduce([np.ptp(native[key],axis=1)>0 for key in SAFETY])
    otherfixed=np.logical_and.reduce([np.ptp(native[key],axis=1)==0 for key in METRICS if key not in ('score','ego_progress')])
    return dict(scenes=len(groups),candidates=len(groups)*16,mean_PDMS=100*float(rewards.mean()),
        native_mean_percent={key:100*float(value.mean()) for key,value in native.items()},
        expected_cost={key:float((1-native[key]).mean()) for key in SAFETY},
        complete_pass_violation_rate={key:float((native[key]<1).mean()) for key in SAFETY},
        joint_safe_fraction=float(safe.mean()),groups_with_joint_safe_candidate=int(safe.any(1).sum()),
        equal_PDMS_groups=int(tied.sum()),all_zero_groups=int((rewards==0).all(1).sum()),all_one_groups=int((rewards==1).all(1).sum()),
        nonzero_PDMS_advantage_groups=int((~tied).sum()),safety_component_varies_groups=int(safety_vary.sum()),
        progress_only_groups=int(((~tied)&otherfixed&(np.ptp(native['ego_progress'],axis=1)>0)).sum()),
        readout='Mean over all16 independently sampled stochastic candidates, then mean over fixed scenes; no candidate selection')
