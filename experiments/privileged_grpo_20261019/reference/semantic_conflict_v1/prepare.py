"""Lock valid whole trajectories with large actual motion-semantic conflicts."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parent;TASK=ROOT.parent;PARENT=TASK/'mechanism_screen_v1'
sys.path.insert(0,str(TASK/'mining_v3r2'))
from common import read,write,file_hash


def format_path(path):return '; '.join(f'w{i+1}: x={p[0]:+.4f}m, y={p[1]:+.4f}m, yaw={p[2]:+.5f}rad' for i,p in enumerate(path))
def geometry(path):
    p=np.asarray(path,dtype=float);assert p.shape==(8,3) and np.isfinite(p).all()
    xy=np.concatenate([np.zeros((1,2)),p[:,:2]]);vel=np.diff(xy,axis=0)/.5
    dh=np.diff(np.concatenate([[0.],p[:,2]]));dh=np.arctan2(np.sin(dh),np.cos(dh))
    return dict(length=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),first_speed=float(np.linalg.norm(vel[0])),final_yaw=float(p[-1,2]),max_speed=float(np.linalg.norm(vel,axis=1).max()),max_vector_acceleration=float(np.linalg.norm(np.diff(vel,axis=0)/.5,axis=1).max()),max_step_yaw=float(np.abs(dh).max()))


def main():
    data=read(ROOT/'DATA.json');cfg=read(TASK/'mining_v3r2/CONFIG.json');old=read(PARENT/'PROMPTS.json')
    assert file_hash(data['source'])==data['source_sha256']
    rows={r['token']:r for r in map(json.loads,Path(data['source']).open())};excluded={r['source_group'] for r in data['dev']}
    pool=sorted([r for r in rows.values() if r['log'].rsplit('_',2)[0] not in excluded],key=lambda r:r['token'])
    geos={t:geometry(r['input_trajectory']) for t,r in rows.items()};texts={t:format_path(r['input_trajectory']) for t,r in rows.items()}
    tok=AutoTokenizer.from_pretrained(cfg['tokenizer_root'],local_files_only=True);enc=lambda s:tok.encode(s,add_special_tokens=False)
    cost=np.full((32,len(pool)),1e9);audit={};coverage=[]
    for i,prompt in enumerate(old['scenes']):
        token=prompt['token'];row=rows[token];g=geos[token];prefix=prompt['trajectory_prefix'];original=prompt['arms']['C']['text'];n=len(enc(original))
        ends=[len(enc(prefix+'; '.join(texts[token].split('; ')[:k]))) for k in range(1,9)]
        feasible=0
        for j,donor in enumerate(pool):
            dt=donor['token'];h=geos[dt]
            if h['max_speed']>30 or h['max_vector_acceleration']>8 or h['max_step_yaw']>.4:continue
            if len(enc(prefix+texts[dt]))!=n:continue
            angle=abs(float(np.arctan2(np.sin(h['final_yaw']-g['final_yaw']),np.cos(h['final_yaw']-g['final_yaw']))))
            ratio=max((h['length']+1)/(g['length']+1),(g['length']+1)/(h['length']+1))
            length_diff=abs(h['length']-g['length']);ade=float(np.linalg.norm(np.asarray(row['input_trajectory'])[:,:2]-np.asarray(donor['input_trajectory'])[:,:2],axis=1).mean())
            angle_conflict=angle>=np.pi/6;scale_conflict=ratio>=4 and length_diff>=8
            if not (angle_conflict or scale_conflict) or ade<3:continue
            new_ends=[len(enc(prefix+'; '.join(texts[dt].split('; ')[:k]))) for k in range(1,9)];boundary=sum(abs(x-y) for x,y in zip(ends,new_ends))
            jitter=int(hashlib.sha256(f'semantic-f-v1:{token}:{dt}'.encode()).hexdigest()[:8],16)/2**32
            strength=min(angle/(np.pi/6),3)+min(np.log(ratio)/np.log(4),3)+.25*(donor['guidance']['navigation']!=row['guidance']['navigation'])
            cost[i,j]=-strength+.005*boundary+.01*jitter;feasible+=1
            audit[i,j]=dict(endpoint_heading_difference_deg=angle*180/np.pi,length_ratio_with_1m_offset=ratio,length_difference_m=length_diff,input_ADE_m=ade,heading_conflict=angle_conflict,scale_conflict=scale_conflict,navigation_different=donor['guidance']['navigation']!=row['guidance']['navigation'],waypoint_boundary_l1=boundary,true_waypoint_token_ends=ends,donor_waypoint_token_ends=new_ends,instruction_token_count=n,token_delta=0,semantic_strength=float(strength))
        coverage.append(dict(token=token,feasible_donors=feasible))
    write(ROOT/'COVERAGE.json',dict(pool=len(pool),coverage=coverage))
    ii,jj=linear_sum_assignment(cost);assert len(ii)==32 and np.max(cost[ii,jj])<1e8,coverage
    donors=[];prompts=[]
    for i,j in zip(ii,jj):
        p=old['scenes'][i];t=p['token'];donor=pool[j];dt=donor['token'];f=p['trajectory_prefix']+texts[dt]
        donors.append(dict(token=t,log=rows[t]['log'],donor_token=dt,donor_log=donor['log'],navigation=rows[t]['guidance']['navigation'],donor_navigation=donor['guidance']['navigation'],input_trajectory=rows[t]['input_trajectory'],donor_trajectory=donor['input_trajectory'],true_geometry=geos[t],donor_geometry=geos[dt],**audit[i,j]))
        prompts.append(dict(token=t,log=p['log'],primary_reason=p['primary_reason'],trajectory_prefix=p['trajectory_prefix'],arms={'C':p['arms']['C'],'F':dict(text=f,token_count=len(enc(f)),sha256=hashlib.sha256(f.encode()).hexdigest())}))
    write(ROOT/'DONORS.json',dict(status='LOCKED',pool_count=len(pool),assignments=donors,source_sha256=data['source_sha256'],parent_prompts_sha256=file_hash(PARENT/'PROMPTS.json')))
    write(ROOT/'PROMPTS.json',dict(status='LOCKED',request=old['request'],scenes=prompts))
    write(ROOT/'PREPARATION.json',dict(status='PASS',scenes=32,exact_token_matches=32,unique_donors=len({d['donor_token'] for d in donors}),heading_conflicts=sum(d['heading_conflict'] for d in donors),scale_conflicts=sum(d['scale_conflict'] for d in donors),navigation_different=sum(d['navigation_different'] for d in donors),input_ADE_mean=float(np.mean([d['input_ADE_m'] for d in donors])),input_ADE_range=[min(d['input_ADE_m'] for d in donors),max(d['input_ADE_m'] for d in donors)]))
    print(json.dumps(read(ROOT/'PREPARATION.json'),indent=2))


if __name__=='__main__':main()
