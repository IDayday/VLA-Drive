"""Freeze full A/B/C/F/E prompts without scoring or model rollouts."""
import argparse,collections,hashlib,json,math
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from asset_adapter import config,rows,scene,digest
ROOT=Path(__file__).resolve().parent
REQUEST='Refine the driving plan. Output the complete corrected future trajectory in the original ego frame.'
def drive(row):return row['log'].rsplit('_',2)[0]
def path_text(path):return '; '.join(f'w{i+1}: x={p[0]:+.4f}m, y={p[1]:+.4f}m, yaw={p[2]:+.5f}rad' for i,p in enumerate(path))
def geometry(path):
    p=np.asarray(path,dtype=float);xy=np.concatenate([np.zeros((1,2)),p[:,:2]]);v=np.diff(xy,axis=0)/.5;dh=np.diff(np.r_[0.,p[:,2]]);dh=np.arctan2(np.sin(dh),np.cos(dh))
    return dict(length=float(np.linalg.norm(v,axis=1).sum()*.5),yaw=float(p[-1,2]),max_speed=float(np.linalg.norm(v,axis=1).max()),max_acc=float(np.linalg.norm(np.diff(v,axis=0)/.5,axis=1).max()),max_dyaw=float(abs(dh).max()))
def dump(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def choose_f(choices, token, policy):
    strict=[x for x in choices if x[2]['token_delta']==0]
    if strict:return min(strict,key=lambda x:(x[0],x[1]))
    if token not in policy['exception_tokens']:return None
    relaxed=[x for x in choices if abs(x[2]['token_delta'])<=policy['max_token_delta']]
    return min(relaxed,key=lambda x:(abs(x[2]['token_delta']),x[0],x[1])) if relaxed else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assets',required=True);ap.add_argument('--out',required=True);ap.add_argument('--monitor-source',required=True);ap.add_argument('--f-policy');a=ap.parse_args();c=config(a.assets);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    expected=json.loads((ROOT/'data_interface/manifest_identity.json').read_text());data={s:rows(c,s) for s in ['train','test']}
    for s in data:assert len(data[s])==expected[s]['count'] and digest(c[s+'_pairs'])==expected[s]['sha256']
    assert not {drive(r) for r in data['train']}&{drive(r) for r in data['test']}
    policy={'max_token_delta':0,'exception_tokens':[]}
    if a.f_policy:
        policy=json.loads(Path(a.f_policy).read_text())
        if policy.get('confirmed') is not True or policy.get('max_token_delta')!=2:raise ValueError('F exception policy is not confirmed')
        if len(policy['exception_tokens'])!=22 or len(set(policy['exception_tokens']))!=22 or not set(policy['exception_tokens'])<={r['token'] for r in data['train']}:raise ValueError('Expected exactly 22 explicitly confirmed training exceptions')
    allrows={r['token']:r for s in data.values() for r in s};text={t:path_text(r['input_trajectory']) for t,r in allrows.items()};geo={t:geometry(r['input_trajectory']) for t,r in allrows.items()}
    pool=sorted([r for r in data['train'] if geo[r['token']]['max_speed']<=30 and geo[r['token']]['max_acc']<=8 and geo[r['token']]['max_dyaw']<=.4],key=lambda r:r['token'])
    tok=AutoTokenizer.from_pretrained(c['tokenizer_root'],local_files_only=True);enc=lambda x:tok.encode(x,add_special_tokens=False)
    cache={};prompts=[];donors=[];missing=[];usage=collections.Counter();coverage=[]
    for i,(t,r) in enumerate(sorted(allrows.items())):
        sc=scene(r,c);base=sc['instruction'];prefix=base+'\n'+REQUEST+'\nInitial trajectory: '
        if prefix not in cache:
            cache[prefix]={d['token']:{'count':len(enc(prefix+text[d['token']])), 'ends':[len(enc(prefix+'; '.join(text[d['token']].split('; ')[:k]))) for k in range(1,9)]} for d in pool}
        n=len(enc(prefix+text[t]));ends=[len(enc(prefix+'; '.join(text[t].split('; ')[:k]))) for k in range(1,9)];g=geo[t];choices=[]
        for d in pool:
            dt=d['token'];h=geo[dt];tc=cache[prefix][dt]
            delta=tc['count']-n
            if drive(d)==drive(r) or abs(delta)>(policy['max_token_delta'] if t in policy['exception_tokens'] else 0):continue
            angle=abs(math.atan2(math.sin(h['yaw']-g['yaw']),math.cos(h['yaw']-g['yaw'])));ratio=max((h['length']+1)/(g['length']+1),(g['length']+1)/(h['length']+1));ld=abs(h['length']-g['length'])
            if angle<math.pi/6 and not(ratio>=4 and ld>=8):continue
            ade=float(np.linalg.norm(np.asarray(r['input_trajectory'])[:,:2]-np.asarray(d['input_trajectory'])[:,:2],axis=1).mean())
            if ade<3:continue
            boundary=sum(abs(x-y) for x,y in zip(ends,tc['ends']));different=d['guidance']['navigation']!=r['guidance']['navigation']
            strength=min(angle/(math.pi/6),3)+min(math.log(ratio)/math.log(4),3)+.25*different;jitter=int(hashlib.sha256(f'semantic-f-v1:{t}:{dt}'.encode()).hexdigest()[:8],16)/2**32
            audit=dict(donor_token=dt,donor_log=d['log'],endpoint_heading_difference_deg=angle*180/math.pi,length_ratio_with_1m_offset=ratio,length_difference_m=ld,input_ADE_m=ade,heading_conflict=angle>=math.pi/6,scale_conflict=ratio>=4 and ld>=8,waypoint_boundary_l1=boundary,token_delta=delta,donor_geometry=h)
            choices.append((-strength+.005*boundary+.01*jitter,dt,audit))
        arms={'A':base,'B':base+'\n'+REQUEST,'C':prefix+text[t],'E':prefix+text[t]+'\n'+r['guidance']['guidance_text']};coverage.append(dict(token=t,feasible_donors=len(choices)))
        chosen=choose_f(choices,t,policy)
        if chosen:
            _,dt,audit=chosen;arms['F']=prefix+text[dt];usage[dt]+=1;donors.append(dict(token=t,log=r['log'],**audit));assert len(enc(arms['F']))-len(enc(arms['C']))==audit['token_delta']
        else:missing.append(dict(token=t,log=r['log'],split='train' if t in {x['token'] for x in data['train']} else 'test',token_count=n,geometry=g))
        prompts.append(dict(token=t,log=r['log'],arms={k:{'text':v,'sha256':hashlib.sha256(v.encode()).hexdigest(),'token_count':len(enc(v))} for k,v in arms.items()}))
        if (i+1)%100==0:print(json.dumps({'prepared':i+1,'total':len(allrows),'unmatched':len(missing)}),flush=True)
    monitor=json.loads(Path(a.monitor_source).read_text());monitor=[x['token'] for x in monitor['train']];assert len(monitor)==64 and set(monitor)<={r['token'] for r in data['train']}
    dump(out/'PROMPTS.json',{'request':REQUEST,'scenes':prompts});dump(out/'DONORS.json',{'assignments':donors,'reuse_counts':dict(usage),'missing':missing,'coverage':coverage,'pool_count':len(pool)});dump(out/'SPLITS.json',{'train':[r['token'] for r in data['train']],'test':[r['token'] for r in data['test']],'monitor':monitor})
    result={'status':'PASS' if not missing else 'INCOMPLETE_F_COVERAGE','scenes':len(allrows),'matched_F':len(donors),'missing_F':len(missing),'unique_donors':len(usage),'max_donor_reuse':max(usage.values(),default=0),'drives':{s:len({drive(r) for r in data[s]}) for s in data},'source_manifest_hashes':{s:digest(c[s+'_pairs']) for s in data},'code_sha256':digest(__file__),'files':{p.name:digest(p) for p in out.glob('*.json')}}
    result.update(f_policy=policy,strict_F=sum(x['token_delta']==0 for x in donors),exception_F=sum(x['token_delta']!=0 for x in donors))
    dump(out/'PREPARATION.json',result);print(json.dumps(result,indent=2));return 0 if not missing else 2
if __name__=='__main__':raise SystemExit(main())
