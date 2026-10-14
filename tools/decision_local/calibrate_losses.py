"""Training-only gradient scale check with the preregistered fixed lambda=1."""
import argparse
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,check_host,digest,source_sha
from tools.decision_local.train_small_model import build_model,tensor_hash
from starVLA.dataloader.decision_local import FrozenConsequenceData,inputs_queries_targets
from starVLA.model.modules.decision_local.consequence_losses import consequence_loss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('features','labels','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--scenes',type=int,default=32)
    p.add_argument('--microbatch',type=int,default=16)
    p.add_argument('--max-seconds',type=float,default=900.)
    a=p.parse_args();resource=check_host();started=time.monotonic()
    if min(a.scenes,a.microbatch,a.max_seconds)<=0:raise ValueError('Positive bounded calibration required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=FrozenConsequenceData(a.features,a.labels)
    if data.feature_identity['split']!='train':raise ValueError('No dev/test calibration')
    model=build_model('P_DIFF',42).cuda();same=build_model('P_ABS',42)
    common=tensor_hash(model.state_dict())==tensor_hash(same.state_dict())
    if not common:raise AssertionError('ABS/DIFF initialization mismatch')
    n=min(a.scenes,len(data.index));named=list(model.named_parameters());norms={};vectors={}
    accumulated={name:[torch.zeros_like(p) for _,p in named] for name in ('absolute','difference')}
    losses={name:torch.zeros((),device='cuda') for name in accumulated}
    for first in range(0,n,a.microbatch):
        if time.monotonic()-started>a.max_seconds:raise TimeoutError('Calibration budget exhausted')
        ids=list(range(first,min(first+a.microbatch,n)))
        batch=data.batch(ids,'cuda',evaluation=True)
        inputs,queries,targets=inputs_queries_targets(batch);raw=model(inputs,queries)
        terms=consequence_loss(raw.float(),targets.relations,targets.valid,difference_weight=1.)
        for name in ('absolute','difference'):
            grads=torch.autograd.grad(terms[name],[p for _,p in named],retain_graph=name=='absolute',allow_unused=True)
            for total,g in zip(accumulated[name],grads):
                if g is not None:total.add_(g,alpha=len(ids)/n)
            losses[name].add_(terms[name].detach(),alpha=len(ids)/n)
    for name in ('absolute','difference'):
        values=[g.flatten() for g in accumulated[name]]
        vectors[name]=torch.cat(values);norms[name]={'total':float(vectors[name].norm()),'raw_loss':float(losses[name])}
        for group in ('memory','query','blocks','output'):
            subset=[v for (n,_),v in zip(named,values) if n.split('.')[0]==group]
            norms[name][group]=float(torch.cat(subset).norm())
    ratio=norms['difference']['total']/max(norms['absolute']['total'],1e-20)
    atomic_json(a.output,{'source_sha':source_sha(ROOT),'tool_sha256':digest(__file__),'resources':resource,
        'features':data.feature_identity['identity'],'labels':data.label_identity['identity'],
        'scenes':n,'microbatch':a.microbatch,'reduction':'mean over original scenes; same per-scene channel/pair reduction as training',
        'common_initialization_exact':common,'norms':norms,
        'difference_to_absolute_gradient_norm':ratio,
        'gradient_cosine':float(torch.nn.functional.cosine_similarity(vectors['absolute'],vectors['difference'],dim=0)),
        'lambda_diff':1.,'physical_scale_m':5.,'distance_clip_m':10.,
        'decision':'retain registered lambda1' if ratio<=1 else 'inspect dominance using training-only evidence before formal acceptance',
        'no_development_or_PDMS_access':True,'GPU_hours':(time.monotonic()-started)/3600.})


if __name__=='__main__':main()
