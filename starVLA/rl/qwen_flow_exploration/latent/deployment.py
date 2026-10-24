"""Pure root-policy packages. No training/cache/normalizer-file reads on load.

The complete frozen Qwen-Flow base remains a separate immutable model asset.
This module exports only the small q parameters and its optional fixed center.
"""
import copy
import json
import math
import os
from pathlib import Path
import uuid
import torch
from starVLA.rl.qwen_sourceflow.contracts import fingerprint,file_sha
from starVLA.rl.qwen_flow_exploration.checkpoint import load_state
from starVLA.rl.qwen_flow_exploration.latent.policy import LatentNoisePolicy
from starVLA.rl.qwen_flow_exploration.latent.normalization import CENTER_ARM
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest
from starVLA.rl.simwam_recipe.checkpoint import atomic_torch
from starVLA.rl.flow_grpo.transactions import atomic_json

SCHEMA='frozen_root_q_deployment_package_v1'
ARMS=('L1','L2','L3',CENTER_ARM)
INFERENCE_CONTRACT=dict(root_shape=[8,3],root_namespace='native_qwen_sourceflow_ODE',root_RNG_device='cpu',root_innovation_dtype='float32',
                        generator_steps=10,generator_grid='source_shift5',feature_layout='scene_batch1',
                        primary='q_sampled_single',secondary='q_mean_root_single_separate',selection='single output; no scorer/GT/oracle')
METADATA_KEYS={'schema','arm','training_seed','qualification','protocol_sha256','base_model_sha256','policy_sha256',
               'checkpoint_state','policy_identity','normalizer_inline','source_training_identity_sha256',
               'source_training_files_sha256','correction_budget_lock_sha256','inference_contract','dependencies',
               'q_weights_file_sha256','sha256'}


def _sealed(value,description):
    body=dict(value)
    if body.pop('sha256',None)!=fingerprint(body):raise ValueError('Changed package seal: '+description)
    return value


def _center_identity(mean,identity,protocol):
    _sealed(identity,'inline TRAIN center')
    if (identity.get('schema')!='TRAIN512_mean_center_then_existing_LayerNorm_v1' or
        identity.get('protocol_sha256')!=protocol['sha256'] or identity.get('base_model_sha256')!=protocol['base_model_sha256'] or
        identity.get('TRAIN_tokens_sha256')!=fingerprint(protocol['splits']['TRAIN']) or
        identity.get('scenes')!=len(protocol['splits']['TRAIN']) or identity.get('feature_dim')!=7168 or
        identity.get('std_whitening') is not False or identity.get('immutable_mean') is not True or
        identity.get('no_DEV_or_test_statistics') is not True or identity.get('existing_LayerNorm_eps')!=1e-5 or
        tensor_digest({'mean':mean})!=identity.get('mean_tensor_sha256')):
        raise ValueError('Inline center changed source/population/tensor/normalization')


def _construct(metadata,state,protocol,device):
    if metadata['arm'] not in ARMS:raise ValueError('Unsupported root policy arm')
    if metadata['protocol_sha256']!=protocol['sha256'] or metadata['base_model_sha256']!=protocol['base_model_sha256']:
        raise ValueError('Package belongs to another common protocol/frozen base')
    if not isinstance(metadata['qualification'],bool):raise ValueError('Qualification identity must be explicit')
    if (metadata['inference_contract']!=INFERENCE_CONTRACT or
        protocol['samplers']['deployment']!='single source shift5 ODE10, original standard Gaussian root'):
        raise ValueError('Package changed the shared deployment sampler/RNG contract')
    progress=metadata['checkpoint_state']
    if (progress['phase']!='COMPLETE' or progress['next_inner']!=0 or progress['arm']!=metadata['arm'] or
        progress['seed']!=metadata['training_seed']):
        raise ValueError('Export requires the matching fully committed policy boundary')
    if (any(not isinstance(progress[key],int) or isinstance(progress[key],bool) or progress[key]<1 for key in ['round','updates']) or
        progress['updates']!=progress['round']*protocol['screen']['inner_epochs'] or
        progress['raw_scene_exposures']!=progress['round']*protocol['screen']['B'] or
        progress['new_candidates']!=progress['raw_scene_exposures']*protocol['screen']['G'] or
        (not metadata['qualification'] and progress['round']!=protocol['screen']['rollout_rounds'])):
        raise ValueError('Package progress does not match the registered complete budget')
    if not isinstance(state,dict) or any(not isinstance(t,torch.Tensor) or t.dtype!=torch.float32 or not torch.isfinite(t).all() for t in state.values()):
        raise ValueError('Root package is finite FP32 policy tensors only')
    desc=metadata['policy_identity'];bounds=desc.get('log_std_bounds',[])
    if (desc.get('feature_dim')!=7168 or desc.get('conditional')!=(metadata['arm']!='L2') or
        desc.get('root_shape')!=[8,3] or len(bounds)!=2 or
        not math.isclose(bounds[0],-math.log(2),rel_tol=0,abs_tol=1e-15) or
        not math.isclose(bounds[1],math.log(2),rel_tol=0,abs_tol=1e-15)):
        raise ValueError('Package changed root shape, network contract or std bounds')
    kwargs={}
    inline=metadata.get('normalizer_inline')
    if metadata['arm']==CENTER_ARM:
        if not isinstance(inline,dict) or set(inline)!={'identity','source_kind'} or 'feature_center' not in state:
            raise ValueError('CENTER requires its inline immutable mean')
        mean=state['feature_center']
        if mean.shape!=(7168,):raise ValueError('Inline center shape mismatch')
        _center_identity(mean,inline['identity'],protocol)
        if desc.get('center_identity')!=inline['identity']:raise ValueError('Inline mean and policy identities differ')
        kwargs=dict(feature_center=mean,center_identity=inline['identity'])
    elif inline is not None or 'feature_center' in state or 'center_identity' in desc:
        raise ValueError('Original arm unexpectedly enabled centering')
    # Loading a trained policy need not consume global sampling RNG state.
    with torch.random.fork_rng(devices=[]):
        policy=LatentNoisePolicy(7168,conditional=desc['conditional'],log_std_bounds=bounds,**kwargs)
    if fingerprint(policy.identity())!=fingerprint(desc):raise ValueError('Package descriptor differs from the actual policy implementation')
    expected=policy.state_dict()
    if set(state)!=set(expected) or any(state[key].shape!=expected[key].shape for key in expected):
        raise ValueError('Strict root-policy tensor keys/shapes mismatch')
    policy.load_state_dict(state,strict=True)
    if tensor_digest(policy.state_dict())!=metadata['policy_sha256']:raise ValueError('Root policy tensor hash mismatch')
    return policy.to(device).eval().requires_grad_(False)


def export_q_package(checkpoint,protocol,output,*,allow_qualification=False):
    """Export a legal completed small q. No full base weights are copied."""
    state,payload=load_state(checkpoint,protocol['sha256'],restore_rng=False)
    identity=payload['identity'];_sealed(identity,'source training identity')
    if identity['qualification'] and not allow_qualification:raise ValueError('Qualification weights cannot be exported as formal policy')
    if identity['protocol_sha256']!=protocol['sha256'] or identity['base_model_sha256']!=protocol['base_model_sha256']:
        raise ValueError('Source checkpoint does not match the locked common base')
    weights={key:value.detach().cpu().contiguous().clone() for key,value in payload['policy'].items()}
    inline=None
    if identity['arm']==CENTER_ARM:
        # Consume the mean already saved with q, not its original asset path.
        center=identity['center_normalizer']['identity']
        if identity['policy_identity'].get('center_identity')!=center:raise ValueError('Source center provenance mismatch')
        inline=dict(identity=copy.deepcopy(center),source_kind='fixed TRAIN statistic stored inline as feature_center; no feature cache or external normalizer path')
    metadata=dict(schema=SCHEMA,arm=identity['arm'],training_seed=identity['seed'],qualification=identity['qualification'],
        protocol_sha256=protocol['sha256'],base_model_sha256=protocol['base_model_sha256'],
        policy_sha256=tensor_digest(weights),checkpoint_state=copy.deepcopy(state),policy_identity=copy.deepcopy(identity['policy_identity']),
        normalizer_inline=inline,source_training_identity_sha256=identity['sha256'],
        source_training_files_sha256=fingerprint(identity['source_files']),
        correction_budget_lock_sha256=identity.get('correction_budget_lock_sha256'),
        inference_contract=copy.deepcopy(INFERENCE_CONTRACT),
        dependencies='only this q package and the separate immutable complete frozen base; no optimizer, training checkpoint, candidate bank or external normalizer asset')
    _construct(metadata,weights,protocol,'cpu')
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Never overwrite a deployment package: '+str(output))
    output.parent.mkdir(parents=True,exist_ok=True)
    incomplete=output.parent/(output.name+'.'+uuid.uuid4().hex+'.incomplete');incomplete.mkdir(exist_ok=False)
    atomic_torch(incomplete/'q_weights.pt',weights)
    metadata['q_weights_file_sha256']=file_sha(incomplete/'q_weights.pt');metadata['sha256']=fingerprint(metadata)
    atomic_json(incomplete/'identity.json',metadata)
    complete=dict(schema=SCHEMA,status='COMPLETE',identity_sha256=metadata['sha256'],q_weights_file_sha256=metadata['q_weights_file_sha256'])
    complete['sha256']=fingerprint(complete);atomic_json(incomplete/'COMPLETE',complete)
    os.rename(incomplete,output)
    return metadata


def load_q_package(package,protocol,device='cpu',*,allow_qualification=False):
    """Return (frozen q, metadata); reads exactly three package files only."""
    package=Path(package)
    if package.name.endswith('.incomplete'):raise ValueError('Incomplete deployment package is not loadable')
    if {path.name for path in package.iterdir()}!={'q_weights.pt','identity.json','COMPLETE'}:
        raise ValueError('Deployment package contains missing or extra files')
    metadata=_sealed(json.loads((package/'identity.json').read_text()),'metadata')
    if set(metadata)!=METADATA_KEYS:raise ValueError('Unexpected/missing deployment identity fields')
    complete=_sealed(json.loads((package/'COMPLETE').read_text()),'COMPLETE')
    if (metadata.get('schema')!=SCHEMA or complete.get('schema')!=SCHEMA or complete.get('status')!='COMPLETE' or
        complete.get('identity_sha256')!=metadata['sha256'] or complete.get('q_weights_file_sha256')!=metadata['q_weights_file_sha256'] or
        file_sha(package/'q_weights.pt')!=metadata['q_weights_file_sha256']):
        raise ValueError('Package publication/files do not match their sealed identity')
    if metadata['qualification'] and not allow_qualification:raise ValueError('Qualification q cannot be used in formal evaluation')
    weights=torch.load(package/'q_weights.pt',map_location='cpu',weights_only=True)
    policy=_construct(metadata,weights,protocol,device)
    return policy,metadata
