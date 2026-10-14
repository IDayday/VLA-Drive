"""Physical differences between CURRENT/CV/log-future on identical GT support.

Descriptive label information only: not oracle planning gain, camera-only
accuracy, or a safety reward. No checkpoint or query selection is performed.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,digest,read,source_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('labels','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();meta=read(a.labels/'identity.json')
    if not (a.labels/'COMPLETE.json').exists():raise ValueError('Complete fixed labels required')
    if a.output.exists():raise FileExistsError('Immutable descriptive audit')
    values=[];masks=[];kin=[];tokens=[]
    for path in sorted((a.labels/'records').glob('*.json')):
        receipt=read(path);f=a.labels/'chunks'/(path.stem+'.safetensors')
        if digest(f)!=receipt['sha256']:raise ValueError('Changed label chunk')
        x=load_file(str(f));values.append(x['relations'][:,0].numpy());masks.append(x['common_valid'][:,0].numpy())
        k=x['kinematic_valid'][:,0].numpy().copy();k[:,0]=True;kin.append(k);tokens+=receipt['tokens']
    if len(set(tokens))!=len(tokens):raise ValueError('Duplicate scenes')
    v=np.concatenate(values);common=np.concatenate(masks);k=np.concatenate(kin)
    common&=k[...,None,None]
    results={}
    for name,sa,sb in [('CV_minus_CURRENT',1,0),('FUTURE_minus_CV',2,1),('FUTURE_minus_CURRENT',2,0)]:
        result={}
        for scope,sl in [('q0',slice(0,1)),('all_registered_queries',slice(None)),('variants',slice(1,None))]:
            first,base=v[:,sa,sl],v[:,sb,sl];valid=common[:,sl]
            entry={}
            for c,channel in enumerate(('road_margin_m','obstacle_clearance_m','geometric_overlap')):
                mask=valid[...,c];aa,bb=first[...,c],base[...,c];error=np.abs(aa-bb)
                row={'valid_points':int(mask.sum()),'coverage':float(mask.mean()),
                    'supported_scenes':int(mask.any((1,2)).sum()),
                    'mean_absolute_information_difference':float(error[mask].mean()) if mask.any() else None}
                if c<2:
                    row.update(nonambiguous_difference_ge_0p5m=int(((error>=.5)&mask).sum()),
                        difference_quantiles_50_90_95_99=np.quantile(error[mask],[.5,.9,.95,.99]).tolist() if mask.any() else None,
                        positive_ge_1_to_nonpositive=int(((bb>=1)&(aa<=0)&mask).sum()),
                        nonpositive_to_positive_ge_1=int(((bb<=0)&(aa>=1)&mask).sum()))
                else:
                    row.update(free_to_overlap=int(((bb<.5)&(aa>.5)&mask).sum()),
                        overlap_to_free=int(((bb>.5)&(aa<.5)&mask).sum()),
                        first_positive=int(((aa>.5)&mask).sum()),base_positive=int(((bb>.5)&mask).sum()),
                        any_change_scenes=int((((aa>.5)!=(bb>.5))&mask).any((1,2)).sum()))
                entry[channel]=row
            result[scope]=entry
        results[name]=result
    atomic_json(a.output,{'source_sha':source_sha(ROOT),'writer_sha256':digest(__file__),
        'scenes':len(tokens),'labels':meta['identity'],'sampling_seed':42,'split':meta['split'],
        'comparison_support':'same three-source intersection; variant kinematics mask, anchor always retained',
        'kinematic_variant_valid_fraction':float(k[:,1:].mean()),'results':results,
        'limitations':['CV uses privileged current object state, not camera-only estimates',
            'Known future masks are privileged; no input or population is selected by these statistics',
            'Physical label differences do not establish realizable planning improvement or an oracle upper bound']})


if __name__=='__main__':main()
