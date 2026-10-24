"""One explicit fixed-TRAIN mean correction; no fitting during train/eval."""
import json
from pathlib import Path
import torch
from starVLA.rl.qwen_sourceflow.contracts import fingerprint,file_sha
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest

CENTER_ARM='L1_CENTER_V1'


def load_center_normalizer(path,protocol,feature_dim=7168):
    path=Path(path).resolve()
    value=torch.load(path,map_location='cpu',weights_only=False)
    if set(value)!={'mean','identity'}:raise ValueError('Center asset must contain a fixed statistic, never qualification policy weights')
    mean=value['mean'];identity=value['identity'];body=dict(identity)
    if body.pop('sha256')!=fingerprint(body):raise ValueError('Changed fixed TRAIN normalizer identity')
    if (identity['schema']!='TRAIN512_mean_center_then_existing_LayerNorm_v1' or
        identity['protocol_sha256']!=protocol['sha256'] or identity['base_model_sha256']!=protocol['base_model_sha256'] or
        identity['TRAIN_tokens_sha256']!=fingerprint(protocol['splits']['TRAIN']) or
        identity['scenes']!=len(protocol['splits']['TRAIN']) or identity['feature_dim']!=feature_dim or
        identity['std_whitening'] is not False or identity['no_DEV_or_test_statistics'] is not True or
        identity['immutable_mean'] is not True or identity['existing_LayerNorm_eps']!=1e-5):
        raise ValueError('Center statistic changed population, source, precision contract or normalization')
    if (not isinstance(mean,torch.Tensor) or mean.dtype!=torch.float32 or mean.shape!=(feature_dim,) or
        mean.requires_grad or not torch.isfinite(mean).all() or tensor_digest({'mean':mean})!=identity['mean_tensor_sha256']):
        raise ValueError('Invalid or changed fixed mean tensor')
    receipt=dict(path=str(path),asset_sha256=file_sha(path),identity=identity,
                 origin='Immutable frozen-base TRAIN statistic; not qualification actor weights')
    return mean.detach().clone(),receipt


def require_center_lock(path,protocol,receipt,seed=42):
    if not path:raise ValueError('Formal CENTER needs its separate ROOT-approved correction budget lock')
    value=json.loads(Path(path).read_text());body=dict(value)
    if body.pop('sha256')!=fingerprint(body):raise ValueError('Changed correction budget lock')
    expected=dict(arm=CENTER_ARM,phase='SCREEN_CORRECTION_LOCKED',protocol_sha256=protocol['sha256'],
                  base_model_sha256=protocol['base_model_sha256'],training_seed=seed,
                  normalizer_asset_sha256=receipt['asset_sha256'],
                  normalizer_identity_sha256=receipt['identity']['sha256'],
                  mean_tensor_sha256=receipt['identity']['mean_tensor_sha256'],
                  TRAIN_tokens_sha256=fingerprint(protocol['splits']['TRAIN']),
                  rounds=64,B=16,G=16,inner_epochs=4,optimizer_updates=256,warmup_updates=8,
                  learning_rate=1e-4,joint_PPO_clip=.2,prior_KL_coefficient=.01)
    if any(value.get(key)!=item for key,item in expected.items()):raise ValueError('Unregistered CENTER budget, source or statistic')
    if value.get('parent_arm')!='L1' or not value.get('parent_budget_lock_sha256') or value.get('root_dispatch_approved') is not True:
        raise ValueError('CENTER is a separate bounded correction, not a silent extension of L1')
    return value


def assert_center_unchanged(policy,identity):
    if identity['arm']==CENTER_ARM:
        receipt=identity['center_normalizer']
        if (policy.feature_center is None or policy.feature_center.requires_grad or
            tensor_digest({'mean':policy.feature_center.detach().cpu()})!=receipt['identity']['mean_tensor_sha256']):
            raise ValueError('The immutable TRAIN center changed during learning or restore')
    elif policy.feature_center is not None:
        raise ValueError('An original arm unexpectedly enabled mean centering')


def policy_from_identity(identity,protocol,device='cpu'):
    from .policy import LatentNoisePolicy
    desc=identity['policy_identity'];kwargs={}
    if identity['arm']==CENTER_ARM:
        receipt=identity['center_normalizer']
        mean,loaded=load_center_normalizer(receipt['path'],protocol,desc['feature_dim'])
        if loaded!=receipt or desc.get('center_identity')!=receipt['identity']:
            raise ValueError('Cannot reconstruct centered deployment from a changed normalizer')
        kwargs=dict(feature_center=mean,center_identity=receipt['identity'])
    elif 'center_normalizer' in identity or 'center_identity' in desc:
        raise ValueError('Old arms must keep the original uncentered computation')
    return LatentNoisePolicy(desc['feature_dim'],conditional=desc['conditional'],
                             log_std_bounds=desc['log_std_bounds'],**kwargs).to(device)
