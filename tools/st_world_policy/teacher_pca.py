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
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from starVLA.dataloader.st_world_history import file_sha256
from starVLA.dataloader.st_world_schema import identity_hash
from starVLA.dataloader.st_world_targets import (
    TeacherCacheReader, atomic_json, file_lock, payload_key, tensor_digest)


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
    def load(cls, root, *, dimension, expected_identity, expected_teacher_contract):
        root=Path(root)
        receipt=json.loads((root/'PCA.json').read_text())
        done=json.loads((root/'COMPLETE.json').read_text())
        if (receipt['identity']!=expected_identity or identity_hash({k:v for k,v in receipt.items() if k!='identity'})!=expected_identity or
                receipt['teacher_contract_identity']!=expected_teacher_contract or done['identity']!=expected_identity or
                done['pca_json_sha256']!=file_sha256(root/'PCA.json') or
                receipt['tensor_sha256']!=file_sha256(root/'statistics.safetensors')):
            raise ValueError('PCA source/statistics/checksum mismatch')
        if receipt['fit_role']!='train' or receipt['whiten'] is not False or receipt['standardize_channels'] is not True:
            raise ValueError('PCA normalization/provenance changed')
        with safe_open(str(root/'statistics.safetensors'),framework='pt',device='cpu') as f:
            tensors={k:f.get_tensor(k) for k in f.keys()}
        if {k:tensor_digest(v) for k,v in tensors.items()}!=receipt['tensor_hashes']:
            raise ValueError('PCA tensor identity mismatch')
        return cls(tensors,dimension=dimension,identity=expected_identity,teacher_contract_identity=expected_teacher_contract)


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
