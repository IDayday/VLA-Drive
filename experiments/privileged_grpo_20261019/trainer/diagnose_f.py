"""Read-only feasibility audit. Does not change any locked F prompts."""
import argparse,json,math
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from asset_adapter import config,rows,scene
from prepare import drive,path_text,geometry,REQUEST
p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--prepared',required=True);p.add_argument('--out',required=True);a=p.parse_args();c=config(a.assets);locked=json.loads((Path(a.prepared)/'DONORS.json').read_text());allrows={r['token']:r for s in ['train','test'] for r in rows(c,s)};pool=rows(c,'train');geo={t:geometry(r['input_trajectory']) for t,r in allrows.items()};tok=AutoTokenizer.from_pretrained(c['tokenizer_root'],local_files_only=True);enc=lambda text:len(tok.encode(text,add_special_tokens=False));report=[]
for missing in locked['missing']:
    t=missing['token'];r=allrows[t];g=geo[t];prefix=scene(r,c)['instruction']+'\n'+REQUEST+'\nInitial trajectory: ';n=enc(prefix+path_text(r['input_trajectory']));strong=[];same_length=[]
    for d in pool:
        dt=d['token'];h=geo[dt]
        if drive(r)==drive(d) or h['max_speed']>30 or h['max_acc']>8 or h['max_dyaw']>.4:continue
        delta=enc(prefix+path_text(d['input_trajectory']))-n
        angle=abs(math.atan2(math.sin(h['yaw']-g['yaw']),math.cos(h['yaw']-g['yaw'])));ratio=max((h['length']+1)/(g['length']+1),(g['length']+1)/(h['length']+1));ld=abs(h['length']-g['length']);ade=float(np.linalg.norm(np.asarray(r['input_trajectory'])[:,:2]-np.asarray(d['input_trajectory'])[:,:2],axis=1).mean());row={'donor':dt,'token_delta':delta,'yaw_deg':angle*180/math.pi,'length_ratio':ratio,'length_diff_m':ld,'ade_m':ade}
        if delta==0:same_length.append(row)
        if (angle>=math.pi/6 or (ratio>=4 and ld>=8)) and ade>=3:strong.append(row)
    report.append({'token':t,'strong_conflict_donors_without_token_limit':len(strong),'min_abs_token_delta':min((abs(x['token_delta']) for x in strong),default=None),'closest_token_examples':sorted(strong,key=lambda x:(abs(x['token_delta']),x['donor']))[:3],'exact_token_physical_donors':len(same_length)})
result={'modified_prompts':False,'unmatched':len(report),'diagnosis':report};Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'unmatched':len(report),'minimum_token_delta_histogram':{str(k):sum(x['min_abs_token_delta']==k for x in report) for k in sorted(set(x['min_abs_token_delta'] for x in report),key=str)}}))
