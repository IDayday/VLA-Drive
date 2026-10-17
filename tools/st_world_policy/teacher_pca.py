"""Train-only channel standardization followed by NON-whitened PCA.

DINO-Foresight reference: pca.py / pca_transform, SHA d98f765d2b743ec90fb38bc657a39c07c7ca4ab0.
Dimensions256/384 share one fixed train-only patch list, one mean/std/covariance
and one eigensystem. This tool reports both; it never chooses the winning width.
No per-token LayerNorm, L2 normalization, or delta normalization is introduced.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path
import sys
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from starVLA.dataloader.st_world_history import file_sha256
from starVLA.dataloader.st_world_schema import TeacherFeatureContract, identity_hash
from starVLA.dataloader.st_world_targets import (
    TeacherCacheReader, LegacyC1Reader, atomic_json, file_lock, payload_key, tensor_digest)


def teacher_channel_family(cache_identity):
    """Audited DINO channel coordinates across only the registered SP grids.

    This permits a common linear basis, NOT equal features at different image
    resolutions. Exact transforms and storage precision remain in each cache
    contract and are never collapsed into this family identifier.
    """
    contract=TeacherFeatureContract.from_dict(cache_identity['contract'])
    execution=cache_identity['execution'];teacher=execution['teacher'];transform=execution['transform']
    if (contract.teacher_identity!=identity_hash(teacher) or contract.transform_identity!=identity_hash(transform) or
            contract.channels!=1024 or contract.patch_hw!=(16,16) or contract.compute_dtype!='float32' or
            contract.feature_layer!='last_block_post_norm_dense_patch' or
            contract.view_names!=('CAM_F0','CAM_L0','CAM_R0')):
        raise ValueError('PCA channel source does not match its complete teacher contract')
    if contract.source_cache_identity==LegacyC1Reader.IDENTITY:
        if (contract.image_hw!=(192,256) or contract.output_grid_hw!=(6,8) or contract.storage_dtype!='float16' or
                dict(contract.pooling)!={'kind':'avg','kernel_hw':(2,2),'stride_hw':(2,2)} or
                transform.get('legacy_recipe')!=teacher or
                teacher.get('precision')!='FP32 model/input/output; TF32 disabled; FP16 storage' or
                teacher.get('normalization')!='ImageNet RGB once; last post-norm patches; no extra L2' or
                teacher.get('source_rgb_range')!='full original16:9 image resized anisotropically to4:3; no crop' or
                teacher.get('interpolation')!='PIL bicubic antialias'):
            raise ValueError('Only the exact audited legacy C1 channel mapping is supported')
        encoder=teacher['encoder_implementation'];mean=teacher['mean'];std=teacher['std']
    else:
        allowed_pool={'kind':'none','kernel_hw':(1,1),'stride_hw':(1,1)}
        pooled={'kind':'avg','kernel_hw':(3,3),'stride_hw':(3,3)}
        if (contract.image_hw!=(288,384) or contract.storage_dtype!='float32' or
                dict(contract.pooling) not in (allowed_pool,pooled) or
                teacher.get('compute_dtype')!='float32' or teacher.get('tf32') is not False or
                teacher.get('autocast') is not False or transform.get('normalization')!='ImageNet_once' or
                transform.get('source')!='complete original16:9 RGB' or transform.get('crop')!='none' or
                transform.get('resize')!='PIL_BICUBIC' or transform.get('augmentation')!='none' or
                transform.get('padding')!='none' or transform.get('image_hw')!=[288,384]):
            raise ValueError('Unregistered PCA precision/preprocessing/grid family')
        encoder=teacher['encoder_source_sha256'];mean=transform['mean'];std=transform['std']
    if (teacher.get('feature')!='last_block_post_norm_dense_patch' or teacher.get('extra_norm')!='none' or
            teacher.get('prefix_tokens_excluded')!=5 or mean!=[.485,.456,.406] or std!=[.229,.224,.225]):
        raise ValueError('PCA feature normalization/channel interpretation changed')
    return {'schema':'st_dino_pca_channel_family_v1','repository':teacher['repository'],
        'revision':teacher['revision'],'weight_sha256':teacher['weight_sha256'],'config_sha256':teacher['config_sha256'],
        'encoder_source_sha256':encoder,'timm':teacher['timm'],'torch':teacher['torch'],'channels':1024,
        'feature':'last_block_post_norm_dense_patch','prefix_tokens_excluded':5,'extra_norm':'none',
        'compute_dtype':'float32','tf32':False,'autocast':False,'mean':mean,'std':std,
        'source_field':'full16:9_no_crop','resize':'PIL_BICUBIC','normalization':'ImageNet_once',
        'allowed_spatial_changes':'registered192x256_pool2_or288x384_native_or_pool3',
        'numerical_equality_across_resolutions_claimed':False}


def register_channel_scope(pca_root, fit_cache_root, target_cache_roots, output):
    """Register exact real cache identities; never rewrites fitted statistics."""
    pca_root=Path(pca_root);receipt=json.loads((pca_root/'PCA.json').read_text())
    fit=TeacherCacheReader(fit_cache_root,expected_identity=receipt['fit_cache_identity'])
    TeacherPCA.load(pca_root,dimension=receipt['dimensions'][0],expected_identity=receipt['identity'],
                    expected_teacher_contract=fit.contract.identity)
    family=teacher_channel_family(fit.identity)
    entries=[]
    for root in [fit_cache_root,*target_cache_roots]:
        root=Path(root).resolve();identity=json.loads((root/'identity.json').read_text())['identity']
        reader=TeacherCacheReader(root,expected_identity=identity)
        candidate=teacher_channel_family(reader.identity)
        if candidate!=family:raise ValueError('PCA channel source/model/software differs from fitted teacher')
        entry={'cache_root':str(root),'cache_identity':identity,'identity_sha256':file_sha256(root/'identity.json'),
               'teacher_contract_identity':reader.contract.identity,'contract':reader.contract.to_dict(),
               'channel_family_identity':identity_hash(candidate)}
        if entry not in entries:entries.append(entry)
    scope={'schema':'st_pca_channel_scope_v1','pca_identity':receipt['identity'],
           'fit_cache_identity':fit.identity['identity'],'fit_contract_identity':fit.contract.identity,
           'channel_family':family,'allowed_caches':entries,'statistics_refitted':False}
    scope['identity']=identity_hash(scope)
    atomic_json(output,scope,immutable=True)
    return scope


def clone_basis_namespace(source, output, fit_cache_root, target_cache_roots):
    """Copy a validated fixed basis byte-for-byte; register a NEW application scope.

    No fitting, reconstruction probe, or width selection is executed. An existing
    scope is immutable, so adding a population requires a new namespace.
    """
    source,output=Path(source).resolve(),Path(output).resolve()
    if source==output:raise ValueError('PCA scope versioning requires a new namespace')
    receipt=json.loads((source/'PCA.json').read_text())
    TeacherPCA.load(source,dimension=receipt['dimensions'][0],expected_identity=receipt['identity'],
                    expected_teacher_contract=receipt['teacher_contract_identity'])
    names=['PCA.json','COMPLETE.json','statistics.safetensors']
    names += [name for name in ['FIT_PATCHES.json','RECONSTRUCTION_PROBES.json','PROBE_TRAIN_PATCHES.json',
        'PROBE_DEV_PATCHES.json','resolved_config.json','DIMENSION_SELECTION.json'] if (source/name).is_file()]
    hashes={name:file_sha256(source/name) for name in names}
    with file_lock(output/'CLONE.lock',blocking=False):
        provenance={'schema':'st_pca_namespace_clone_v1','source_namespace':str(source),
            'source_files_sha256':hashes,'pca_identity':receipt['identity'],'statistics_refitted':False,
            'prior_channel_scope_sha256':file_sha256(source/'CHANNEL_SCOPE.json') if (source/'CHANNEL_SCOPE.json').exists() else None}
        atomic_json(output/'CLONED_FROM.json',provenance,immutable=True)
        for name in names:
            target=output/name
            if target.exists():
                if file_sha256(target)!=hashes[name]:raise ValueError('Conflicting fixed PCA clone: '+name)
                continue
            content=(source/name).read_bytes()
            fd,tmp=tempfile.mkstemp(prefix=name+'.',suffix='.tmp',dir=output)
            try:
                with os.fdopen(fd,'wb') as f:f.write(content);f.flush();os.fsync(f.fileno())
                if file_sha256(tmp)!=hashes[name] or file_sha256(source/name)!=hashes[name]:
                    raise ValueError('PCA source changed during namespace copy')
                os.replace(tmp,target)
            finally:
                if os.path.exists(tmp):os.unlink(tmp)
        scope=register_channel_scope(output,fit_cache_root,target_cache_roots,output/'CHANNEL_SCOPE.json')
        result={'status':'COMPLETE','schema':'st_pca_namespace_v1','pca_identity':receipt['identity'],
                'channel_scope_identity':scope['identity'],'source_files_sha256':hashes,
                'statistics_refitted':False,'old_namespace_modified':False}
        atomic_json(output/'NAMESPACE_COMPLETE.json',result,immutable=True)
    return result


def validate_channel_scope(path, receipt, expected_contract, target_cache_identity):
    scope=json.loads(Path(path).read_text())
    if (scope.get('schema')!='st_pca_channel_scope_v1' or scope.get('identity')!=identity_hash({k:v for k,v in scope.items() if k!='identity'}) or
            scope['pca_identity']!=receipt['identity'] or scope['fit_cache_identity']!=receipt['fit_cache_identity'] or
            scope['fit_contract_identity']!=receipt['teacher_contract_identity'] or scope['statistics_refitted'] is not False):
        raise ValueError('Invalid PCA channel scope/basis binding')
    matches=[r for r in scope['allowed_caches'] if r['cache_identity']==target_cache_identity and r['teacher_contract_identity']==expected_contract]
    if len(matches)!=1:raise ValueError('Target cache is not explicitly admitted by PCA channel scope')
    fit=[r for r in scope['allowed_caches'] if r['cache_identity']==scope['fit_cache_identity']]
    if len(fit)!=1:raise ValueError('PCA scope has no unique original fitted source')
    for entry in [fit[0],matches[0]]:
        path=Path(entry['cache_root'])/'identity.json'
        if file_sha256(path)!=entry['identity_sha256']:raise ValueError('PCA scoped cache identity file changed')
        cache=json.loads(path.read_text())
        if (cache['identity']!=entry['cache_identity'] or identity_hash({k:v for k,v in cache.items() if k!='identity'})!=entry['cache_identity'] or
                cache['contract']!=entry['contract'] or TeacherFeatureContract.from_dict(entry['contract']).identity!=entry['teacher_contract_identity'] or
                teacher_channel_family(cache)!=scope['channel_family']):
            raise ValueError('PCA admitted channel mapping changed')
    return scope['identity']


class StreamingCovariance:
    """Numerically stable FP64 batch-merge statistics; no full fit matrix stored."""
    def __init__(self, channels):
        self.n = 0
        self.mean = torch.zeros(channels, dtype=torch.float64)
        self.m2 = torch.zeros(channels, channels, dtype=torch.float64)

    def update(self, x):
        x = torch.as_tensor(x, dtype=torch.float64, device='cpu')
        if x.ndim != 2 or x.shape[1] != self.mean.numel() or not torch.isfinite(x).all():
            raise ValueError('Invalid PCA fit features')
        if len(x) == 0:
            return
        n, mean = len(x), x.mean(0)
        centered = x - mean
        delta = mean - self.mean
        total = self.n + n
        self.m2 += centered.T @ centered + torch.outer(delta, delta) * (self.n * n / total)
        self.mean += delta * (n / total)
        self.n = total

    def basis(self, max_components, *, std_epsilon=1e-6):
        if not 0 < std_epsilon <= 1e-2 or not 1 <= max_components <= self.mean.numel() or self.n <= max_components:
            raise ValueError('Insufficient train patches or invalid PCA dimension/std floor')
        # Population channel std, then sample covariance of standardized features.
        variance = (self.m2.diag() / self.n).clamp_min(0)
        std = variance.sqrt().clamp_min(std_epsilon)
        covariance = self.m2 / (self.n - 1) / std[:,None] / std[None,:]
        covariance = (covariance + covariance.T) * .5
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True, stable=True)
        values = eigenvalues[order].clamp_min(0)
        components = eigenvectors[:,order[:max_components]].T.contiguous()
        # Pin eigenvector sign; equivalent basis sign otherwise varies by LAPACK.
        anchor = components.abs().argmax(1)
        sign = components[torch.arange(len(components)),anchor].sign()
        components *= torch.where(sign == 0, torch.ones_like(sign), sign)[:,None]
        return {'channel_mean':self.mean.float(),'channel_std':std.float(),
                'pca_mean':torch.zeros_like(self.mean,dtype=torch.float32),
                'components':components.float(),'eigenvalues':values.float(),
                'constant_channels':(variance.sqrt()<std_epsilon)}


class TeacherPCA:
    def __init__(self, tensors, *, dimension, identity, teacher_contract_identity):
        if not 1 <= dimension <= tensors['components'].shape[0]:
            raise ValueError('PCA dimension outside fitted basis')
        if any(not torch.isfinite(t).all() for t in tensors.values()):
            raise ValueError('Nonfinite PCA statistics')
        self.tensors=tensors
        self.dimension=dimension
        self.identity=identity
        self.teacher_contract_identity=teacher_contract_identity

    def _values(self, x):
        if x.shape[-1] != self.tensors['channel_mean'].numel():
            raise ValueError('PCA input channel count changed')
        return {k:v.to(device=x.device,dtype=torch.float32) for k,v in self.tensors.items()
                if k in ('channel_mean','channel_std','pca_mean','components')}

    def transform(self, x):
        p=self._values(x)
        return (((x.float()-p['channel_mean'])/p['channel_std'])-p['pca_mean']) @ p['components'][:self.dimension].T

    def inverse(self, z):
        if z.shape[-1]!=self.dimension:
            raise ValueError('PCA inverse dimension mismatch')
        p={k:v.to(device=z.device,dtype=torch.float32) for k,v in self.tensors.items()
           if k in ('channel_mean','channel_std','pca_mean','components')}
        return (z.float() @ p['components'][:self.dimension] + p['pca_mean']) * p['channel_std'] + p['channel_mean']

    @classmethod
    def load(cls, root, *, dimension, expected_identity, expected_teacher_contract,
             channel_scope=None, target_cache_identity=None, provider_scope=None, target_provider=None):
        root=Path(root)
        receipt=json.loads((root/'PCA.json').read_text())
        done=json.loads((root/'COMPLETE.json').read_text())
        if (receipt['identity']!=expected_identity or identity_hash({k:v for k,v in receipt.items() if k!='identity'})!=expected_identity or
                done['identity']!=expected_identity or
                done['pca_json_sha256']!=file_sha256(root/'PCA.json') or
                receipt['tensor_sha256']!=file_sha256(root/'statistics.safetensors')):
            raise ValueError('PCA source/statistics/checksum mismatch')
        if receipt['fit_role']!='train' or receipt['whiten'] is not False or receipt['standardize_channels'] is not True:
            raise ValueError('PCA normalization/provenance changed')
        scope_identity=None
        if provider_scope is not None or target_provider is not None:
            if (provider_scope is None or target_provider is None or
                    channel_scope is not None or target_cache_identity is not None):
                raise ValueError('Provider PCA scope is explicit and separate from cache identity scopes')
            from tools.st_world_policy.provider_pca import validate_provider_scope
            scope_identity=validate_provider_scope(provider_scope,receipt,expected_teacher_contract,target_provider)
        elif channel_scope is not None:
            if target_cache_identity is None:raise ValueError('Explicit PCA scope requires the target cache identity')
            scope_identity=validate_channel_scope(channel_scope,receipt,expected_teacher_contract,target_cache_identity)
        elif receipt['teacher_contract_identity']!=expected_teacher_contract:
            if channel_scope is None or target_cache_identity is None:
                raise ValueError('PCA exact teacher contract mismatch; explicit registered channel scope required')
            scope_identity=validate_channel_scope(channel_scope,receipt,expected_teacher_contract,target_cache_identity)
        with safe_open(str(root/'statistics.safetensors'),framework='pt',device='cpu') as f:
            tensors={k:f.get_tensor(k) for k in f.keys()}
        if {k:tensor_digest(v) for k,v in tensors.items()}!=receipt['tensor_hashes']:
            raise ValueError('PCA tensor identity mismatch')
        result=cls(tensors,dimension=dimension,identity=expected_identity,teacher_contract_identity=expected_teacher_contract)
        result.channel_scope_identity=scope_identity
        result.fit_teacher_contract_identity=receipt['teacher_contract_identity']
        return result


def make_patch_plan(reader, *, seed, max_patches, purpose):
    """Fixed score-free sample on declared scenes; role validity is not inferred."""
    if max_patches < 1 or purpose not in ('fit','diagnostic'):
        raise ValueError('Invalid bounded PCA plan')
    if purpose=='fit' and reader.manifest['split']!='train':
        raise ValueError('PCA fit is train-only; dev/test statistics are forbidden')
    if any(s['split'] != reader.manifest['split'] for s in reader.manifest['scenes']):
        raise ValueError('Mixed-role PCA manifest')
    h,w=reader.contract.output_grid_hw
    scenes=sorted(reader.manifest['scenes'],key=lambda s:identity_hash({'seed':seed,'scene':s['token']}))
    # First round visits every scene/view/time before taking a second location.
    refs=[]
    for scene in scenes:
        for t,frame in enumerate(scene['frames']):
            for v,ref in enumerate(frame['views']):
                if ref is not None:
                    key=payload_key(ref,reader.contract)
                    g=torch.Generator().manual_seed(int(identity_hash({'seed':seed,'scene':scene['token'],'t':t,'v':v})[:15],16))
                    indices=torch.randperm(h*w,generator=g).tolist()
                    refs.append((scene,t,v,key,indices))
    rows=[]
    for depth in range(h*w):
        for scene,t,v,key,indices in refs:
            rows.append({'token':scene['token'],'log':scene['log'],'split':scene['split'],
                         'time_index':t,'view_index':v,'image_key':key,'patch_index':indices[depth]})
            if len(rows)==max_patches:
                break
        if len(rows)==max_patches:
            break
    if not rows:
        raise ValueError('No real patches for PCA')
    plan={'schema':'st_pca_patch_plan_v1','purpose':purpose,'seed':seed,'cache_identity':reader.identity['identity'],
          'teacher_contract_identity':reader.contract.identity,'split':reader.manifest['split'],
          'max_patches':max_patches,'rows':rows}
    plan['identity']=identity_hash(plan)
    return plan


def read_patch_batches(reader, plan, *, batch_size=256, max_images_in_ram=4):
    if batch_size < 1 or max_images_in_ram < 1 or plan['cache_identity']!=reader.identity['identity']:
        raise ValueError('PCA reader budget/cache mismatch')
    # Group only IO, not the registered patch population. Read each image once
    # instead of rereading a1.7MB native map for every sampled1024D patch.
    grouped=OrderedDict()
    for row in plan['rows']:
        grouped.setdefault(row['image_key'],[]).append(row['patch_index'])
    batch=[]
    for key,indices in grouped.items():
        z,mask=reader.read_image(key)
        flat=z.reshape(-1,z.shape[-1]);valid=mask.flatten()
        for index in indices:
            if not 0<=index<mask.numel():raise ValueError('PCA patch outside true grid')
            if not valid[index]:continue
            batch.append(flat[index].float().clone())
            if len(batch)==batch_size:
                yield torch.stack(batch);batch=[]
    if batch:yield torch.stack(batch)



def reconstruction_probe(reader, plan, tensors, dimensions):
    totals={d:{'sse':0.,'target_sse':0.,'elements':0,'patches':0} for d in dimensions}
    total_var=float(tensors['eigenvalues'].sum())
    for x in read_patch_batches(reader,plan):
        for d in dimensions:
            p=TeacherPCA(tensors,dimension=d,identity='diagnostic',teacher_contract_identity=reader.contract.identity)
            reconstructed=p.inverse(p.transform(x))
            row=totals[d];row['sse']+=float((reconstructed-x).double().square().sum())
            row['target_sse']+=float(x.double().square().sum());row['elements']+=x.numel();row['patches']+=len(x)
    return {str(d):{'raw_domain_mse':r['sse']/r['elements'] if r['elements'] else None,
                    'relative_reconstruction_sse':r['sse']/r['target_sse'] if r['target_sse'] else None,
                    'retained_standardized_variance':float(tensors['eigenvalues'][:d].sum())/total_var if total_var else None,
                    'valid_patches':r['patches'],'elements':r['elements']} for d,r in totals.items()}


def run(args):
    if args.clone_basis_from:
        if args.register_channel_scope:raise ValueError('Choose namespace clone or standalone scope registration')
        actual=json.loads((Path(args.cache_root)/'identity.json').read_text())['identity']
        if actual!=args.cache_identity:raise ValueError('PCA fitted cache identity mismatch')
        return clone_basis_namespace(args.clone_basis_from,args.output,args.cache_root,args.scope_cache_root)
    if args.register_channel_scope:
        actual=json.loads((Path(args.cache_root)/'identity.json').read_text())['identity']
        if actual!=args.cache_identity:raise ValueError('Channel scope fitted cache identity mismatch')
        return register_channel_scope(args.register_channel_scope,args.cache_root,args.scope_cache_root,args.output)
    if min(args.max_patches,args.probe_patches,args.cpu_threads) <= 0 or args.max_patches>1000000:
        raise ValueError('Explicit bounded PCA patch/CPU budgets required')
    dims=sorted(set(args.dimensions))
    if dims!=args.dimensions or dims!=[256,384]:
        raise ValueError('Campaign compares exactly256/384 on one common basis; no automatic width search')
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    reader=TeacherCacheReader(args.cache_root,expected_identity=args.cache_identity)
    if reader.manifest['split']!='train':raise ValueError('Only train cache may fit PCA')
    torch.set_num_threads(args.cpu_threads)
    plan=make_patch_plan(reader,seed=args.seed,max_patches=args.max_patches,purpose='fit')
    config={**vars(args),'fit_plan_identity':plan['identity'],'teacher_contract_identity':reader.contract.identity,
            'implementation_sha256':file_sha256(__file__),'normalization':'channel_standardize_then_nonwhiten_pca'}
    with file_lock(root/'FIT.lock',blocking=False):
        atomic_json(root/'resolved_config.json',config,immutable=True)
        atomic_json(root/'FIT_PATCHES.json',plan,immutable=True)
        if (root/'COMPLETE.json').exists():
            old=json.loads((root/'PCA.json').read_text())
            TeacherPCA.load(root,dimension=dims[0],expected_identity=old['identity'],expected_teacher_contract=reader.contract.identity)
            return {'status':'REUSED','identity':old['identity'],'selection':'NOT_SELECTED_BY_TOOL'}
        stats=StreamingCovariance(reader.contract.channels)
        for x in read_patch_batches(reader,plan):stats.update(x)
        tensors=stats.basis(max(dims),std_epsilon=args.std_epsilon)
        temporary=root/'statistics.safetensors.tmp'
        save_file(tensors,str(temporary),metadata={'fit_plan_identity':plan['identity'],'whiten':'false'})
        temporary.replace(root/'statistics.safetensors')
        receipt={'schema':'st_teacher_pca_v1','fit_role':'train','fit_cache_identity':reader.identity['identity'],
                 'fit_plan_identity':plan['identity'],'teacher_contract_identity':reader.contract.identity,
                 'fit_patches':stats.n,'requested_patches':len(plan['rows']),'dimensions':dims,
                 'standardize_channels':True,'whiten':False,'std_epsilon':args.std_epsilon,
                 'statistics_accumulator_dtype':'float64','transform_dtype':'float32',
                 'tensor_sha256':file_sha256(root/'statistics.safetensors'),
                 'tensor_hashes':{k:tensor_digest(v) for k,v in tensors.items()},
                 'selection':'LEAD_SELECTS_ONCE_AFTER_TWO_PROBES','source_sha256':file_sha256(__file__)}
        receipt['identity']=identity_hash(receipt)
        atomic_json(root/'PCA.json',receipt,immutable=True)
        # Two separate fixed probes: train and supplied dev, or disjoint-seed train.
        probe0=make_patch_plan(reader,seed=args.seed+1,max_patches=args.probe_patches,purpose='diagnostic')
        atomic_json(root/'PROBE_TRAIN_PATCHES.json',probe0,immutable=True)
        reports={'train':reconstruction_probe(reader,probe0,tensors,dims)}
        if args.probe_cache_root:
            if not args.probe_cache_identity:raise ValueError('Explicit probe cache identity required')
            probe_reader=TeacherCacheReader(args.probe_cache_root,expected_identity=args.probe_cache_identity,expected_contract=reader.contract)
            if probe_reader.manifest['split']!='dev':raise ValueError('Heldout probe must be existing dev, never test')
            if {s['log'] for s in reader.manifest['scenes']} & {s['log'] for s in probe_reader.manifest['scenes']}:
                raise ValueError('Train/dev probe raw-log overlap')
            probe1=make_patch_plan(probe_reader,seed=args.seed+2,max_patches=args.probe_patches,purpose='diagnostic')
            atomic_json(root/'PROBE_DEV_PATCHES.json',probe1,immutable=True)
            reports['dev']=reconstruction_probe(probe_reader,probe1,tensors,dims)
        else:
            reports['dev']={'status':'BLOCKED_MISSING_DEV_CACHE','selection_allowed':False}
        atomic_json(root/'RECONSTRUCTION_PROBES.json',{'identity':receipt['identity'],'results':reports,
                    'dimension_selected':None,'no_dev_statistics_fit':True},immutable=True)
        done={'status':'COMPLETE','identity':receipt['identity'],'pca_json_sha256':file_sha256(root/'PCA.json'),
              'probe_report_sha256':file_sha256(root/'RECONSTRUCTION_PROBES.json'),
              'basis_fitted':True,'dimension_selected':None}
        atomic_json(root/'COMPLETE.json',done,immutable=True)
        return done


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cache-root','cache-identity','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--probe-cache-root');p.add_argument('--probe-cache-identity')
    p.add_argument('--register-channel-scope',help='Existing fitted PCA root; register exact compatible cache identities without refitting')
    p.add_argument('--clone-basis-from',help='Copy validated fixed basis/selection to a NEW namespace and register supplied caches; never refit')
    p.add_argument('--scope-cache-root',action='append',default=[],help='Actual complete additional teacher cache admitted by the explicit scope')
    p.add_argument('--dimensions',nargs='+',type=int,default=[256,384])
    p.add_argument('--max-patches',type=int,default=32768)
    p.add_argument('--probe-patches',type=int,default=4096)
    p.add_argument('--std-epsilon',type=float,default=1e-6)
    p.add_argument('--seed',type=int,default=20261017)
    p.add_argument('--cpu-threads',type=int,default=4)
    return p


def main():
    try:
        print(json.dumps(run(parser().parse_args()),sort_keys=True));return 0
    except Exception as exc:
        print(json.dumps({'status':'FAILED','error':repr(exc)}),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
