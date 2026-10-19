"""Validate paired screening and summarize scene-clustered behavioral contrasts."""
from collections import defaultdict
from datetime import datetime,timezone
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from diversity import summarize

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/attempt01'
CONTRASTS=[('B','A'),('C','B'),('C','D'),('D','A'),('E','C')]


def read(path):return json.loads(path.read_text())
def rows(path):return [json.loads(line) for line in path.open()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def interval(values):
    a=np.asarray(values,dtype=float);rng=np.random.default_rng(20261019)
    boot=a[rng.integers(0,len(a),size=(5000,len(a)))].mean(axis=1)
    return dict(mean=float(a.mean()),median=float(np.median(a)),bootstrap95=np.quantile(boot,[.025,.975]).tolist(),n_scenes=len(a),positive=int((a>1e-8).sum()),negative=int((a < -1e-8).sum()),ties=int((np.abs(a)<=1e-8).sum()))
def cosine(a,b):
    a=np.asarray(a).ravel();b=np.asarray(b).ravel();return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))


def main():
    result=read(RUN/'RESULT.json');cfg=read(RUN/'CONFIG.json')
    assert result['status']=='COMPLETE' and not result['training_performed'] and not result['final_test_used']
    assert result['counts']=={'groups':480,'candidates':3840,'scenes':32}
    for path,h in cfg['source_sha256'].items():assert sha(Path(path))==h,path
    data=read(ROOT/'DATA.json');metas=data['dev'];tokens=[r['token'] for r in metas];reasons={r['token']:r['primary_reason'] for r in metas}
    assert len(set(tokens))==len({r['source_group'] for r in metas})==32
    assert (ROOT/'DATA.json').read_bytes()==(ROOT.parent/'grpo_probe_v2/DATA.json').read_bytes()
    assert sha(Path(data['source']))==data['source_sha256']
    targets={r['token']:r for r in rows(Path(data['source']))}
    assignments=read(ROOT.parent/'splits/log_reason_90_10_v1/ASSIGNMENTS.json')['logs']
    assert all(assignments[r['log']]=='train' for r in metas)
    donors={r['token']:r for r in read(ROOT/'DONORS.json')['assignments']}
    assert len({d['donor_token'] for d in donors.values()})==32
    for token,d in donors.items():
        donor=targets[d['donor_token']];target=targets[token]
        assert donor['log'].rsplit('_',2)[0] not in {r['source_group'] for r in metas}
        assert donor['input_trajectory']==d['donor_trajectory'] and target['input_trajectory']==d['input_trajectory']
        assert donor['guidance']['navigation']==target['guidance']['navigation']
        assert d['token_delta']==0
    groups=rows(RUN/'groups.jsonl');rollouts=rows(RUN/'rollouts.jsonl');inputs=rows(RUN/'inputs.jsonl');matching=rows(RUN/'token_matching.jsonl')
    assert len(groups)==480 and len(rollouts)==3840 and len(inputs)==160 and len(matching)==32
    lookup={(g['token'],g['replicate'],g['arm']):g for g in groups};assert len(lookup)==480
    grouped=defaultdict(list)
    for row in rollouts:
        assert row['token'] in tokens and row['arm'] in 'ABCDE' and row['replicate'] in range(3)
        assert all(np.isfinite(v) and 0<=v<=1 for v in row['metrics'].values())
        grouped[row['group_id']].append(row)
    trajectories={}
    for token in tokens:
        for rep in range(3):
            gs=[lookup[token,rep,arm] for arm in 'ABCDE']
            assert len({g['seed'] for g in gs})==len({g['noise_sha256'] for g in gs})==1
            for g in gs:
                rr=sorted(grouped[g['group_id']],key=lambda r:r['candidate'])
                assert [r['candidate'] for r in rr]==list(range(8))
                assert all(r['seed']==g['seed'] and r['noise_sha256']==g['noise_sha256'] for r in rr)
                a=np.asarray([r['trajectory'] for r in rr]);trajectories[token,rep,g['arm']]=a
                recomputed=summarize(a,[r['metrics'] for r in rr])
                for key,value in recomputed.items():
                    if isinstance(value,dict):
                        for k,v in value.items():assert abs(g[key][k]-v)<1e-10
                    else:assert abs(float(g[key])-float(value))<1e-10
        assert len({lookup[token,rep,'A']['seed'] for rep in range(3)})==3
    input_lookup={(r['token'],r['arm']):r for r in inputs};assert len(input_lookup)==160
    for token in tokens:
        rr=[input_lookup[token,arm] for arm in 'ABCDE']
        assert len({r['pixel_values_sha256'] for r in rr})==len({r['state_sha256'] for r in rr})==1
        assert input_lookup[token,'C']['input_token_count']==input_lookup[token,'D']['input_token_count']
        assert input_lookup[token,'C']['special_positions']==input_lookup[token,'D']['special_positions']
    old=rows(ROOT.parent/'grpo_probe_v2/runs/attempt01/rollouts.jsonl')
    old={(r['token'],r['candidate']):r for r in old if r['stage']=='initial' and r['arm']=='plain' and r['token'] in tokens}
    for token in tokens:
        g=lookup[token,0,'A']
        for r in grouped[g['group_id']]:
            original=old[token,r['candidate']]
            assert np.max(np.abs(np.asarray(original['trajectory'])-np.asarray(r['trajectory'])))<1e-6
            assert all(abs(v-r['metrics'][k])<1e-9 for k,v in original['metrics'].items())
    metrics=['pdms_mean','pdms_first','pdms_max','rule_pass_fraction','pairwise_ADE_m','pairwise_FDE_m','pdms_std']
    scene=[];scene_lookup={}
    for token in tokens:
        for arm in 'ABCDE':
            gs=[lookup[token,rep,arm] for rep in range(3)]
            tt=np.stack([trajectories[token,rep,arm] for rep in range(3)])
            gt=np.asarray(targets[token]['gt_trajectory']);failure=np.asarray(targets[token]['input_trajectory'])
            row=dict(token=token,arm=arm,primary_reason=reasons[token],**{k:float(np.mean([g[k] for g in gs])) for k in metrics},
                zero_variance_groups=sum(g['zero_reward_variance'] for g in gs),all_zero_groups=sum(g['all_zero'] for g in gs),
                gt_ADE_m=float(np.linalg.norm(tt[...,:2]-gt[None,None,:,:2],axis=-1).mean()),failure_ADE_m=float(np.linalg.norm(tt[...,:2]-failure[None,None,:,:2],axis=-1).mean()),
                endpoint_x_m=float(tt[...,-1,0].mean()),endpoint_y_m=float(tt[...,-1,1].mean()),mean_metrics={k:float(np.mean([g['mean_metrics'][k] for g in gs])) for k in gs[0]['mean_metrics']})
            scene.append(row);scene_lookup[token,arm]=row
    aggregate={};per_seed={}
    extra=['gt_ADE_m','failure_ADE_m','endpoint_x_m','endpoint_y_m']
    for arm in 'ABCDE':
        ss=[scene_lookup[token,arm] for token in tokens]
        aggregate[arm]=dict(**{k:float(np.mean([s[k] for s in ss])) for k in metrics+extra},mean_metrics={k:float(np.mean([s['mean_metrics'][k] for s in ss])) for k in ss[0]['mean_metrics']},zero_variance_groups=sum(s['zero_variance_groups'] for s in ss),all_zero_groups=sum(s['all_zero_groups'] for s in ss))
        per_seed[arm]=[{k:float(np.mean([lookup[t,rep,arm][k] for t in tokens])) for k in metrics} for rep in range(3)]
    contrasts={};reason_contrasts={};geometric_contrasts={}
    for left,right in CONTRASTS:
        name=left+'-'+right
        contrasts[name]={k:interval([scene_lookup[t,left][k]-scene_lookup[t,right][k] for t in tokens]) for k in metrics+extra}
        contrasts[name]['pdms_by_seed']=[float(np.mean([lookup[t,rep,left]['pdms_mean']-lookup[t,rep,right]['pdms_mean'] for t in tokens])) for rep in range(3)]
        reason_contrasts[name]={label:dict(n=sum(reasons[t]==label for t in tokens),mean=float(np.mean([scene_lookup[t,left]['pdms_mean']-scene_lookup[t,right]['pdms_mean'] for t in tokens if reasons[t]==label]))) for label in sorted(set(reasons.values()))}
        shifts=[]
        for token in tokens:
            a=np.stack([trajectories[token,rep,left] for rep in range(3)]);b=np.stack([trajectories[token,rep,right] for rep in range(3)])
            yaw=a[...,-1,2]-b[...,-1,2]
            shifts.append(dict(token=token,path_shift_m=float(np.linalg.norm(a[...,:2]-b[...,:2],axis=-1).mean()),endpoint_dx_m=float((a[...,-1,0]-b[...,-1,0]).mean()),endpoint_dy_m=float((a[...,-1,1]-b[...,-1,1]).mean()),endpoint_yaw_shift_deg=float(np.abs(np.arctan2(np.sin(yaw),np.cos(yaw))).mean()*180/np.pi)))
        geometric_contrasts[name]={k:interval([s[k] for s in shifts]) for k in shifts[0] if k!='token'}
    following=[]
    for token in tokens:
        c=np.stack([trajectories[token,rep,'C'] for rep in range(3)]).mean(axis=(0,1))
        d=np.stack([trajectories[token,rep,'D'] for rep in range(3)]).mean(axis=(0,1))
        donor=np.asarray(donors[token]['donor_trajectory'])[:,:2];true=np.asarray(donors[token]['input_trajectory'])[:,:2]
        direction=donor-true;shift=d[:,:2]-c[:,:2]
        following.append(dict(token=token,cosine=cosine(shift,direction),projection_coefficient=float((shift*direction).sum()/((direction*direction).sum()+1e-12)),donor_geometry_ADE_m=donors[token]['trajectory_ADE_m']))
    repr_rows=rows(RUN/'representations.jsonl');vel_rows=rows(RUN/'velocity_contrasts.jsonl')
    assert len(repr_rows)==32 and len(vel_rows)==96 and len(list((RUN/'features').glob('*.npz')))==32
    repr_lookup={r['token']:r for r in repr_rows};vel_lookup={(r['token'],r['replicate']):r for r in vel_rows}
    def validate_delta(x,y,record):
        x=np.asarray(x,dtype=np.float64).ravel();y=np.asarray(y,dtype=np.float64).ravel()
        expected={'rms':float(np.sqrt(np.mean((x-y)**2))),'relative_l2':float(np.linalg.norm(x-y)/(np.linalg.norm(y)+1e-12)),'cosine':cosine(x,y)}
        for key,value in expected.items():assert np.isclose(value,record[key],rtol=1e-5,atol=1e-6),(key,value,record[key])
    for token in tokens:
        with np.load(RUN/'features'/f'{token}.npz') as arrays:
            assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
            for left,right in CONTRASTS:
                name=left+'-'+right;record=next(c for c in repr_lookup[token]['contrasts'] if c['contrast']==name)
                validate_delta(arrays['query_'+left],arrays['query_'+right],record['query'])
                validate_delta(arrays['context_'+left],arrays['context_'+right],record['projected_context'])
                for rep in range(3):
                    assert vel_lookup[token,rep]['noise_sha256']==lookup[token,rep,'A']['noise_sha256']
                    validate_delta(arrays[f'velocity_rep{rep}_{left}'],arrays[f'velocity_rep{rep}_{right}'],vel_lookup[token,rep]['contrasts'][name])
    feature_stats={}
    for left,right in CONTRASTS:
        name=left+'-'+right
        records=[next(c for c in row['contrasts'] if c['contrast']==name) for row in repr_rows]
        feature_stats[name]={key:{k:float(np.mean([r[key][k] for r in records])) for k in ['rms','relative_l2','cosine']} for key in ['query','projected_context']}
        feature_stats[name]['initial_velocity']={k:float(np.mean([r['contrasts'][name][k] for r in vel_rows])) for k in ['rms','relative_l2','cosine']}
    output=dict(status='ACCEPTED',checked_utc=datetime.now(timezone.utc).isoformat(),counts=result['counts'],aggregate=aggregate,per_seed=per_seed,contrasts=contrasts,by_original_reason=reason_contrasts,paired_geometry=geometric_contrasts,feature_differences=feature_stats,donor_following={k:interval([r[k] for r in following]) for k in ['cosine','projection_coefficient']},wall_seconds=result['elapsed_s'],peak_gpu_gib=result['peak_gpu_allocated_gib'],limitations=cfg['plan']['limitations'])
    save(ROOT/'ANALYSIS.json',output);save(ROOT/'SCENE_RESULTS.json',scene);save(ROOT/'DONOR_FOLLOWING.json',following)
    with (ROOT/'GROUP_RESULTS.csv').open('x') as f:
        fields=['token','replicate','arm','seed']+metrics+['zero_reward_variance','all_zero']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(groups)
    evidence=[RUN/'CONFIG.json',RUN/'RESULT.json',RUN/'rollouts.jsonl',RUN/'groups.jsonl',RUN/'inputs.jsonl',RUN/'representations.jsonl',RUN/'velocity_contrasts.jsonl',ROOT/'ANALYSIS.json',ROOT/'SCENE_RESULTS.json',ROOT/'DONOR_FOLLOWING.json',ROOT/'GROUP_RESULTS.csv']+sorted((RUN/'features').glob('*.npz'))
    save(ROOT/'VALIDATION.json',dict(status='ACCEPTED',groups=480,candidates=3840,unique_scenes=32,arms=5,replicates=3,candidates_each=8,shared_noise_verified=True,baseline_replay_verified_candidates=256,C_D_full_token_and_special_position_matches=32,donors_whole_real_paths=True,donor_source_eval_overlap=0,fixed_test_used=False,source_hashes_verified=True,score_and_geometry_recomputed=True,model_trainable_parameters=0,feature_files=32,feature_differences_recomputed=True,evidence_sha256={str(p.relative_to(ROOT)):sha(p) for p in evidence}))
    print(json.dumps({'aggregate':aggregate,'contrasts':{k:{'pdms':v['pdms_mean'],'per_seed':v['pdms_by_seed']} for k,v in contrasts.items()},'donor_following':output['donor_following'],'feature_differences':feature_stats},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
