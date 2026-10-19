"""Fix all prompts and matched whole-trajectory donors before model inference."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parent
TASK=ROOT.parent
sys.path.insert(0,str(TASK/'mining_v3r2'))
from common import read,write,raw_scene,file_hash


def format_path(path):
    return '; '.join(f'w{i+1}: x={p[0]:+.4f}m, y={p[1]:+.4f}m, yaw={p[2]:+.5f}rad' for i,p in enumerate(path))


def geometry(path):
    p=np.asarray(path,dtype=np.float64);assert p.shape==(8,3) and np.isfinite(p).all()
    xy=np.concatenate([np.zeros((1,2)),p[:,:2]])
    return dict(first_speed=float(np.linalg.norm(p[0,:2])/.5),length=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()))


def main():
    plan=read(ROOT/'PLAN.json');data=read(ROOT/'DATA.json');cfg=read(TASK/'mining_v3r2/CONFIG.json')
    assert file_hash(data['source'])==data['source_sha256']
    rows={r['token']:r for r in map(json.loads,Path(data['source']).open())}
    tokenizer=AutoTokenizer.from_pretrained(cfg['tokenizer_root'],local_files_only=True)
    enc=lambda s:tokenizer.encode(s,add_special_tokens=False)
    metas=data['dev'];excluded={r['source_group'] for r in metas}
    pool=sorted([r for r in rows.values() if r['log'].rsplit('_',2)[0] not in excluded],key=lambda r:r['token'])
    texts={r['token']:format_path(r['input_trajectory']) for r in rows.values()}
    geos={r['token']:geometry(r['input_trajectory']) for r in rows.values()}
    cost=np.full((len(metas),len(pool)),1e9);audits={};prefixes={};scenes={}
    for i,meta in enumerate(metas):
        token=meta['token'];row=rows[token];scene=raw_scene(row['source_paths']);scenes[token]=scene
        prefix=scene['instruction']+'\n'+plan['request']+'\nInitial trajectory: '
        prefixes[token]=prefix;original=prefix+texts[token];n=len(enc(original))
        original_boundaries=[len(enc(prefix+'; '.join(texts[token].split('; ')[:k]))) for k in range(1,9)]
        for j,donor in enumerate(pool):
            dt=donor['token']
            if donor['guidance']['navigation']!=row['guidance']['navigation']:continue
            speed_ratio=(geos[dt]['first_speed']+.5)/(geos[token]['first_speed']+.5)
            length_ratio=(geos[dt]['length']+2)/(geos[token]['length']+2)
            slow_pair=(geos[token]['first_speed']<1 and geos[dt]['first_speed']<1.5 and geos[token]['length']<8 and geos[dt]['length']<8)
            if not slow_pair and (not .5<=speed_ratio<=2 or not .5<=length_ratio<=2):continue
            donor_n=len(enc(prefix+texts[dt]));delta=donor_n-n
            if delta!=0:continue
            boundaries=[len(enc(prefix+'; '.join(texts[dt].split('; ')[:k]))) for k in range(1,9)]
            boundary_l1=sum(abs(a-b) for a,b in zip(boundaries,original_boundaries))
            random_tie=int(hashlib.sha256(f'mechanism-donor-v1:{token}:{dt}'.encode()).hexdigest()[:8],16)/2**32
            value=abs(np.log(speed_ratio))+abs(np.log(length_ratio))+.03*boundary_l1+.75*random_tie
            cost[i,j]=value;audits[i,j]=dict(slow_pair_exception=slow_pair,speed_ratio_with_offset=speed_ratio,length_ratio_with_offset=length_ratio,instruction_token_count=n,token_delta=delta,waypoint_boundary_l1=boundary_l1,true_waypoint_token_ends=original_boundaries,donor_waypoint_token_ends=boundaries,matching_cost=float(value))
    ii,jj=linear_sum_assignment(cost)
    assert len(ii)==32 and np.max(cost[ii,jj])<1e8,{'unmatched':[metas[i]['token'] for i in ii if cost[i,jj[list(ii).index(i)]]>=1e8]}
    assignments=[];prompts=[]
    for i,j in zip(ii,jj):
        meta=metas[i];token=meta['token'];row=rows[token];donor=pool[j];dt=donor['token'];scene=scenes[token]
        prefix=prefixes[token];a=scene['instruction'];b=a+'\n'+plan['request'];c=prefix+texts[token];d=prefix+texts[dt];e=c+'\n'+row['guidance']['guidance_text']
        assert len(enc(c))==len(enc(d))
        assignment=dict(token=token,log=row['log'],donor_token=dt,donor_log=donor['log'],navigation=row['guidance']['navigation'],input_trajectory=row['input_trajectory'],donor_trajectory=donor['input_trajectory'],true_geometry=geos[token],donor_geometry=geos[dt],trajectory_ADE_m=float(np.linalg.norm(np.asarray(row['input_trajectory'])[:,:2]-np.asarray(donor['input_trajectory'])[:,:2],axis=1).mean()),**audits[i,j])
        assignments.append(assignment)
        prompts.append(dict(token=token,log=row['log'],primary_reason=meta['primary_reason'],trajectory_prefix=prefix,arms={arm:dict(text=text,token_count=len(enc(text)),sha256=hashlib.sha256(text.encode()).hexdigest()) for arm,text in zip('ABCDE',[a,b,c,d,e])}))
    write(ROOT/'DONORS.json',dict(status='LOCKED',pool_count=len(pool),source_sha256=data['source_sha256'],data_sha256=file_hash(ROOT/'DATA.json'),assignments=assignments,exact_instruction_token_count_matches=32,unique_donors=len({r['donor_token'] for r in assignments}),donor_sources_overlap_eval=0))
    write(ROOT/'PROMPTS.json',dict(status='LOCKED',request=plan['request'],scenes=prompts))
    write(ROOT/'PREPARATION.json',dict(status='PASS',scene_count=32,arms=5,replicates=3,candidates=3840,same_data_manifest=file_hash(ROOT/'DATA.json')==file_hash(TASK/'grpo_probe_v2/DATA.json'),exact_C_D_text_token_count=32,donor_ADE_range=[min(r['trajectory_ADE_m'] for r in assignments),max(r['trajectory_ADE_m'] for r in assignments)],donor_ADE_mean=float(np.mean([r['trajectory_ADE_m'] for r in assignments])),instruction_token_counts={arm:[p['arms'][arm]['token_count'] for p in prompts] for arm in 'ABCDE'}))
    print(json.dumps(read(ROOT/'PREPARATION.json'),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
