"""Audit the added semantic-conflict condition against immutable paired controls."""
import json,hashlib,importlib.util
from pathlib import Path
from collections import defaultdict
import numpy as np
from diversity import summarize
ROOT=Path(__file__).resolve().parent;PARENT=ROOT.parent/'mechanism_screen_v1';RUN=ROOT/'runs/attempt01';OLD=PARENT/'runs/attempt01'
spec=importlib.util.spec_from_file_location('parent_analysis',PARENT/'analyze.py');pa=importlib.util.module_from_spec(spec);spec.loader.exec_module(pa)
read=pa.read;rows=pa.rows;sha=pa.sha;save=pa.save;interval=pa.interval

def main():
    result=read(RUN/'RESULT.json');cfg=read(RUN/'CONFIG.json')
    assert result['status']=='COMPLETE' and result['counts']==dict(groups=96,candidates=768,scenes=32)
    assert not result['training_performed'] and not result['final_test_used'] and result['model_witnesses_unchanged']
    assert cfg['trainable_parameters']==0
    for p,h in cfg['source_sha256'].items():assert sha(Path(p))==h,p
    for p,h in read(PARENT/'VALIDATION.json')['evidence_sha256'].items():assert sha(PARENT/p)==h,p
    data=read(ROOT/'DATA.json');tokens=[r['token'] for r in data['dev']];eval_drives={r['source_group'] for r in data['dev']};assert len(eval_drives)==len(set(tokens))==32
    assert sha(Path(data['source']))==data['source_sha256']
    sources={r['token']:r for r in rows(Path(data['source']))};donors={r['token']:r for r in read(ROOT/'DONORS.json')['assignments']};assert len({r['donor_token'] for r in donors.values()})==32
    for t,d in donors.items():
        assert sources[t]['input_trajectory']==d['input_trajectory'] and sources[d['donor_token']]['input_trajectory']==d['donor_trajectory']
        assert d['donor_log'].rsplit('_',2)[0] not in eval_drives and d['token_delta']==0
        p=np.asarray(d['donor_trajectory']);q=np.asarray(d['input_trajectory']);assert p.shape==(8,8,3)[1:] and np.isfinite(p).all()
        ade=np.linalg.norm(p[:,:2]-q[:,:2],axis=-1).mean();assert abs(ade-d['input_ADE_m'])<1e-9 and ade>=3
        angle=abs(np.arctan2(np.sin(p[-1,2]-q[-1,2]),np.cos(p[-1,2]-q[-1,2])))*180/np.pi
        lengths=[np.linalg.norm(np.diff(np.vstack([np.zeros((1,2)),a[:,:2]]),axis=0),axis=-1).sum() for a in [p,q]]
        assert angle>=30 or ((max(lengths)+1)/(min(lengths)+1)>=4 and abs(lengths[0]-lengths[1])>=8)
        vel=np.diff(np.vstack([np.zeros((1,2)),p[:,:2]]),axis=0)/.5
        assert np.linalg.norm(vel,axis=-1).max()<=30 and np.linalg.norm(np.diff(vel,axis=0)/.5,axis=-1).max()<=8
        dy=np.diff(np.r_[0,p[:,2]]);assert np.abs(np.arctan2(np.sin(dy),np.cos(dy))).max()<=.4
    rr=rows(RUN/'rollouts.jsonl');gg=rows(RUN/'groups.jsonl');assert len(rr)==768 and len(gg)==96
    oldrr=rows(OLD/'rollouts.jsonl');oldgg=rows(OLD/'groups.jsonl');lookup={(g['token'],g['replicate'],g['arm']):g for g in oldgg+gg}
    grouped=defaultdict(list)
    for r in oldrr+rr:grouped[r['token'],r['replicate'],r['arm']].append(r)
    arrays={}
    for t in tokens:
        for rep in range(3):
            for a in 'ACDF':
                g=lookup[t,rep,a];rs=sorted(grouped[t,rep,a],key=lambda x:x['candidate']);assert [x['candidate'] for x in rs]==list(range(8))
                assert all(x['seed']==g['seed'] and x['noise_sha256']==g['noise_sha256'] for x in rs)
                assert g['noise_sha256']==lookup[t,rep,'C']['noise_sha256'] and g['seed']==lookup[t,rep,'C']['seed']
                p=np.asarray([x['trajectory'] for x in rs]);assert p.shape==(8,8,3) and np.isfinite(p).all();arrays[t,rep,a]=p
                expected=summarize(p,[r['metrics'] for r in rs])
                for k,v in expected.items():
                    if isinstance(v,dict):assert all(abs(g[k][j]-w)<1e-10 for j,w in v.items())
                    else:assert abs(float(g[k])-float(v))<1e-10
    replay=rows(RUN/'replay.jsonl');assert len(replay)==24
    oldlookup={(r['token'],r['replicate'],r['arm'],r['candidate']):r for r in oldrr}
    assert all(r==oldlookup[r['token'],r['replicate'],r['arm'],r['candidate']] for r in replay)
    inp={(r['token'],r['arm']):r for r in rows(RUN/'inputs.jsonl')};assert len(inp)==64
    for t in tokens:
        for k in ['input_token_count','special_positions','pixel_values_sha256','state_sha256']:assert inp[t,'C'][k]==inp[t,'F'][k]
    metrics=['pdms_mean','pdms_first','pdms_max','rule_pass_fraction','pairwise_ADE_m','pdms_std']
    sc={(t,a):{k:float(np.mean([lookup[t,rep,a][k] for rep in range(3)])) for k in metrics} for t in tokens for a in 'ACDF'}
    aggregate={a:{k:float(np.mean([sc[t,a][k] for t in tokens])) for k in metrics} for a in 'ACDF'}
    for a in 'ACDF':aggregate[a]['mean_metrics']={k:float(np.mean([lookup[t,rep,a]['mean_metrics'][k] for t in tokens for rep in range(3)])) for k in gg[0]['mean_metrics']}
    contrasts={};geometry={};scene=[];following=[]
    for a in 'ACD':
        name='F-'+a;contrasts[name]={k:interval([sc[t,'F'][k]-sc[t,a][k] for t in tokens]) for k in metrics}
        contrasts[name]['pdms_by_seed']=[float(np.mean([lookup[t,rep,'F']['pdms_mean']-lookup[t,rep,a]['pdms_mean'] for t in tokens])) for rep in range(3)]
        stats=[]
        for t in tokens:
            f=np.stack([arrays[t,rep,'F'] for rep in range(3)]);b=np.stack([arrays[t,rep,a] for rep in range(3)]);dy=f[...,-1,2]-b[...,-1,2]
            stat=dict(path_shift_m=float(np.linalg.norm(f[...,:2]-b[...,:2],axis=-1).mean()),endpoint_dx_m=float((f[...,-1,0]-b[...,-1,0]).mean()),endpoint_dy_m=float((f[...,-1,1]-b[...,-1,1]).mean()),endpoint_yaw_shift_deg=float(np.abs(np.arctan2(np.sin(dy),np.cos(dy))).mean()*180/np.pi))
            stats.append(stat);scene.append(dict(token=t,contrast=name,pdms_delta=sc[t,'F']['pdms_mean']-sc[t,a]['pdms_mean'],**stat))
        geometry[name]={k:interval([s[k] for s in stats]) for k in stats[0]}
    for t in tokens:
        shift=np.stack([arrays[t,rep,'F']-arrays[t,rep,'C'] for rep in range(3)]).mean(axis=(0,1))[:,:2];direction=np.asarray(donors[t]['donor_trajectory'])[:,:2]-np.asarray(donors[t]['input_trajectory'])[:,:2]
        following.append(dict(token=t,cosine=pa.cosine(shift,direction),projection=float((shift*direction).sum()/((direction**2).sum()+1e-12))))
    reps=rows(RUN/'representations.jsonl');vels=rows(RUN/'velocity_contrasts.jsonl');assert len(reps)==32 and len(vels)==96
    def verify(x,y,r):
        x=x.astype('float64').ravel();y=y.astype('float64').ravel();exp=dict(rms=float(np.sqrt(((x-y)**2).mean())),relative_l2=float(np.linalg.norm(x-y)/(np.linalg.norm(y)+1e-12)),cosine=pa.cosine(x,y))
        assert all(np.isclose(exp[k],r[k],rtol=1e-5,atol=1e-6) for k in exp)
    for r in reps:
        t=r['token']
        with np.load(RUN/'features'/f'{t}.npz') as f,np.load(OLD/'features'/f'{t}.npz') as old:
            for a in 'ACD':
                for key,field in [('query','query'),('context','projected_context')]:verify(f[key+'_F'],old[key+'_'+a],r['contrasts']['F-'+a][field])
                for rep in range(3):verify(f[f'velocity_rep{rep}_F'],old[f'velocity_rep{rep}_{a}'],next(v for v in vels if v['token']==t and v['replicate']==rep)['contrasts']['F-'+a])
    features={}
    for a in 'ACD':
        n='F-'+a;features[n]={k:{stat:float(np.mean([r['contrasts'][n][k][stat] for r in reps])) for stat in ['rms','relative_l2','cosine']} for k in ['query','projected_context']}
        features[n]['initial_velocity']={stat:float(np.mean([r['contrasts'][n][stat] for r in vels])) for stat in ['rms','relative_l2','cosine']}
    out=dict(status='ACCEPTED',aggregate=aggregate,contrasts=contrasts,paired_geometry=geometry,feature_differences=features,donor_following={k:interval([r[k] for r in following]) for k in ['cosine','projection']},preparation=read(ROOT/'PREPARATION.json'),runtime=result,limitations=cfg['plan']['limitations'])
    save(ROOT/'ANALYSIS.json',out);save(ROOT/'SCENE_RESULTS.json',scene);save(ROOT/'DONOR_FOLLOWING.json',following)
    evidence=[p for p in RUN.rglob('*') if p.is_file()]+[ROOT/n for n in ['ANALYSIS.json','SCENE_RESULTS.json','DONOR_FOLLOWING.json','analyze.py']]
    save(ROOT/'VALIDATION.json',dict(status='ACCEPTED',new_candidates=768,C_replay_candidates=24,scenes=32,source_hashes_verified=True,parent_evidence_verified=True,common_noise_verified=True,whole_real_donors_verified=True,semantic_conflict_verified=True,input_token_special_position_matches=32,fixed_test_used=False,feature_differences_recomputed=True,evidence_sha256={str(p.relative_to(ROOT)):sha(p) for p in evidence}))
    print(json.dumps(dict(aggregate=aggregate,contrasts={k:v['pdms_mean'] for k,v in contrasts.items()},geometry=geometry['F-C'],following=out['donor_following'],features=features['F-C']),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
