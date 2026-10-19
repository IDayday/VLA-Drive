"""Paired pre/post analysis; fail on missing, duplicate, or mismatched candidates."""
import argparse,json
from pathlib import Path
import numpy as np

def read(path):return json.loads(Path(path).read_text())
def load_phase(root,phase,expected_tokens):
    receipt=read(root/'RESULT.json')
    if receipt['status']!='COMPLETE' or receipt['phase']!=phase:raise ValueError('Incomplete phase')
    result={}
    for job in receipt['jobs']:
        arm=job['job']['arm'];seed=job['job']['seed'];folder=Path(job['output'])/'evaluation'/phase;accepted=read(folder/'ACCEPTANCE.json');groups={}
        for name in accepted['files']:
            row=read(folder/name);key=(row['token'],row['replicate'])
            if key in groups or [r['candidate'] for r in row['candidates']]!=list(range(8)):raise ValueError('Duplicate or missing candidates')
            groups[key]=row
        expected={(t,r) for t in expected_tokens for r in range(3)}
        if set(groups)!=expected:raise ValueError('Missing/extra evaluation groups')
        result[arm,seed]=groups
    return result

def cluster_ci(values,drives,reps=5000):
    unique=sorted(set(drives));rng=np.random.default_rng(20261019);indices=[np.flatnonzero(np.array(drives)==g) for g in unique];stats=[]
    for _ in range(reps):
        idx=np.concatenate([indices[i] for i in rng.integers(0,len(unique),len(unique))]);stats.append(float(np.mean(values[idx])))
    return {'mean':float(np.mean(values)),'cluster_count':len(unique),'ci95':np.quantile(stats,[.025,.975]).tolist(),'scene_count':len(values)}

def main(a):
    splits=read(Path(a.prepared)/'SPLITS.json');tokens=sorted(splits['test']);before=load_phase(Path(a.initial),'initial',tokens);after=load_phase(Path(a.final),'final',tokens);cfg=read(Path(a.final)/'TRAINING.json');seeds=cfg['training_seeds'];results={};groups={};drives=[]
    for token in tokens:drives.append(before['A',0][token,0]['drive'])
    for arm in cfg['arms']:
        base_arm=arm.split('_')[0];initial=before[base_arm,0];pre=[];post=[]
        for token in tokens:
            pre.append(np.mean([c['metrics']['PDMS'] for rep in range(3) for c in initial[token,rep]['candidates']]))
            post.append(np.mean([c['metrics']['PDMS'] for seed in seeds for rep in range(3) for c in after[arm,seed][token,rep]['candidates']]))
            for seed in seeds:
                for rep in range(3):
                    if initial[token,rep]['noise_sha256']!=after[arm,seed][token,rep]['noise_sha256']:raise ValueError('Pre/post random numbers not paired')
        pre=np.array(pre);post=np.array(post);groups[arm]=(pre,post)
        results[arm]={'pre':cluster_ci(pre,drives),'post_seed_mean':cluster_ci(post,drives),'gain':cluster_ci(post-pre,drives),'per_seed_post':{str(seed):float(np.mean([c['metrics']['PDMS'] for r in after[arm,seed].values() for c in r['candidates']])) for seed in seeds},'exploration':{stage:{name:float(np.mean([r['exploration'][name] for r in collection])) for name in ['pdms_mean','pdms_std','oracle8','all_zero','all_one','constant_other','effective_group','repair_fraction','pairwise_ADE_m','pairwise_FDE_m']} for stage,collection in [('pre',list(initial.values())),('post',[r for seed in seeds for r in after[arm,seed].values()])]}}
    contrasts={}
    for left,right in [('E','C'),('B','A'),('C','B'),('C','F'),('E_VLM','C_VLM'),('E_DiT','C_DiT'),('E','A'),('E_VLM','A_VLM'),('E_DiT','A_DiT')]:
        lp,lq=groups[left];rp,rq=groups[right];contrasts[left+'-'+right]={'pre':cluster_ci(lp-rp,drives),'post':cluster_ci(lq-rq,drives),'difference_in_gain':cluster_ci((lq-lp)-(rq-rp),drives)}
    output=Path(a.out);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps({'status':'COMPLETE','protocol':'NAVSIM v1.1 PDMS','scenes':len(tokens),'source_drives':len(set(drives)),'training_seeds':seeds,'arms':results,'contrasts':contrasts,'ci_note':'Scene-weighted cluster bootstrap over represented source drives; post scores average paired training seeds first. These CIs quantify scene-sampling uncertainty, not a hierarchical training-seed confidence interval. Auxiliary contrasts uncorrected.'},indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--initial',required=True);p.add_argument('--final',required=True);p.add_argument('--prepared',required=True);p.add_argument('--out',required=True);main(p.parse_args())
