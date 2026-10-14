"""Training-side coverage/scale audit; never selects thresholds using PDMS."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,digest,read,source_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();meta=read(a.labels/'identity.json');done=read(a.labels/'COMPLETE.json')
    if done['identity']!=meta['identity']:raise ValueError('Complete immutable labels required')
    chunks=[];fallback=objects=0;count=0
    for receipt in sorted((a.labels/'records').glob('*.json')):
        r=read(receipt);path=a.labels/'chunks'/(receipt.stem+'.safetensors')
        if digest(path)!=r['sha256']:raise ValueError('Label chunk changed')
        t=load_file(str(path));chunks.append({k:v.numpy() for k,v in t.items() if k in ('relations','valid','common_valid','kinematics','kinematic_valid','in_grid','poses')})
        count+=len(r['tokens'])
        for row in r.get('current_cv',[]):fallback+=row['zero_velocity_fallback'];objects+=row['objects']
    if count!=done['scenes']:raise ValueError('Missing scenes')
    data={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    # N, anchor, source, variant, physical time, channel.
    v=data['relations'];mask=data['valid'];kin=data['kinematic_valid'].copy();kin[:,:,0]=True
    training=mask&kin[:,:,None,:,None,None];common=data['common_valid']&kin[:,:,:,None,None]
    sources=['CURRENT','CURRENT_CV','TRUE_FUTURE'];channels=['road_margin_m','clearance_m','overlap']
    result={'scene_count':count,'source_coverage':{},'common_support':{},'label_identity':meta['identity'],
        'split':meta['split'],'ground_support_policy':meta['ground_support_policy'],'query_config':meta['query_config'],
        'source_sha':source_sha(ROOT),'audit_sha256':digest(__file__),
        'threshold_basis':'fixed physical registration:0.5m grid ambiguity,road +/-0.25m,clearance0/1m,overlap binary; no PDMS access',
        'CV_objects':objects,'CV_missing_velocity_zero_fallback':fallback,
        'kinematically_valid_fraction_per_variant':data['kinematic_valid'].mean((0,1)).tolist(),
        'body_in_grid_fraction_per_time':data['in_grid'].mean((0,1,2)).tolist()}
    for j,source in enumerate(sources):
        out={}
        for c,name in enumerate(channels):
            m=training[:,:,j,...,c];values=v[:,:,j,...,c][m]
            out[name]={'raw_coverage':float(mask[:,:,j,...,c].mean()),'after_variant_mask_coverage':float(m.mean()),
                'per_time_coverage':m.mean((0,1,2)).tolist(),'supported_scenes':int(m.any((1,2,3)).sum()),
                'quantiles':np.quantile(values,[0,.01,.1,.5,.9,.99,1]).tolist() if len(values) else None,
                'clipped_10m_fraction':float((np.abs(values)>=9.999).mean()) if len(values) and c<2 else None,
                'positive_fraction':float((values>.5).mean()) if len(values) and c==2 else None}
        result['source_coverage'][source]=out
    for c,name in enumerate(channels):
        m=common[...,c]
        result['common_support'][name]={'coverage':float(m.mean()),'per_time_coverage':m.mean((0,1,2)).tolist(),
            'supported_scenes':int(m.any((1,2,3)).sum())}
    kin_stats=data['kinematics'].reshape(-1,3)
    result['kinematics_percentiles']={name:np.quantile(kin_stats[:,j],[.5,.9,.95,.99,1]).tolist() for j,name in enumerate(('speed_mps','acceleration_mps2','yaw_rate_rps'))}
    differences={}
    for lhs,rhs,name in ((1,0,'CV_minus_CURRENT'),(2,1,'FUTURE_minus_CV')):
        m=training[:,:,lhs]&training[:,:,rhs];d=v[:,:,lhs]-v[:,:,rhs]
        differences[name]={channels[c]:{'mean_abs':float(np.abs(d[...,c][m[...,c]]).mean()) if m[...,c].any() else None,
            'different_at_least_0p5_fraction':float((np.abs(d[...,c][m[...,c]])>=.5).mean()) if m[...,c].any() else None,
            'coverage':float(m[...,c].mean())} for c in range(3)}
    result['information_source_label_differences']=differences
    result['interpretation']='Label differences/coverage, not learned planning value. No threshold selected from dev/test.'
    atomic_json(a.output,result)


if __name__=='__main__':main()
