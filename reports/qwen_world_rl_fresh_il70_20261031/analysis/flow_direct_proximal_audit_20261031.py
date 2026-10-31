"""CPU-only bounded Safety journal evidence and primary source pinning."""
import hashlib,json,os,re,urllib.request
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
R=Path('/mnt/project/qwen-world-rl-optimization-20261030');A=R/'analysis';root=Path(next(x for x in json.loads((R/'RUNS.json').read_text())['runs'] if x['id']=='PPO_SAFETY_EP')['root'])

def rows(path,target):
 start=target-63
 with path.open('rb') as f:
  lo=0;hi=path.stat().st_size
  while lo<hi:
   mid=(lo+hi)//2;f.seek(mid)
   if mid:f.readline()
   line=f.readline()
   if not line:hi=mid;continue
   r=json.loads(line)
   if r['updates']>=start:hi=mid
   else:lo=f.tell()
  f.seek(lo);out=[]
  while True:
   line=f.readline()
   if not line:break
   r=json.loads(line)
   if r['updates']>target:break
   if r['updates']>=start:out.append(r)
 assert len(out)==64 and out[0]['updates']==start
 return out


def summarize(ars,stage,parts=None):
 report={}
 for k in ('clipfrac','positive_ppo_blocked_fraction','negative_ppo_blocked_fraction','ppo_blocked_fraction','ratio_mean','step7_ratio_mean','step8_ratio_mean','step9_ratio_mean','step7_ratio_ess_fraction','step8_ratio_ess_fraction','step9_ratio_ess_fraction'):
  v=[r['metrics'][k] for r in ars if k in r['metrics']]
  if v:report[k]=dict(mean=float(np.mean(v)),minimum=float(min(v)),maximum=float(max(v)))
 ch=[r['source_clamp_hit_rate'] for r in ars]
 report['source_logdensity_clamp_hit_rate']=dict(mean=float(np.mean(ch)),maximum=float(max(ch)))
 if parts is not None:
  stats=dict(positive_candidate_steps=0,negative_candidate_steps=0,eligible_candidate_steps=0,positive_blocked_steps=0,negative_blocked_steps=0)
  for rank,r in parts:
   index=((r['updates']-1)//4+1-(stage-64)//4-1)*8+rank
   adv=cache[stage]['advantages'][index];mask=cache[stage]['mask'][index]
   npos=int(((adv>0)&(mask>0)).sum())*3;nneg=int(((adv<0)&(mask>0)).sum())*3
   stats['positive_candidate_steps']+=npos;stats['negative_candidate_steps']+=nneg;stats['eligible_candidate_steps']+=npos+nneg
   stats['positive_blocked_steps']+=int(round(r['metrics']['positive_ppo_blocked_fraction']*npos));stats['negative_blocked_steps']+=int(round(r['metrics']['negative_ppo_blocked_fraction']*nneg))
  stats['positive_blocked_rate']=stats['positive_blocked_steps']/max(1,stats['positive_candidate_steps'])
  stats['negative_blocked_rate']=stats['negative_blocked_steps']/max(1,stats['negative_candidate_steps'])
  stats['overall_blocked_rate']=(stats['negative_blocked_steps']+stats['positive_blocked_steps'])/max(1,stats['eligible_candidate_steps'])
  report['eligible_weighted_counts']=stats
 return report

cache={};result={}
for u in (10000,24000):
 with np.load(A/'SAFETY_EP_PROCESS_CACHE'/f'u{u:06}_rollouts.npz',allow_pickle=False) as z:cache[u]={k:z[k].copy() for k in ('advantages','mask')}
 parts=[(rank,r) for rank in range(8) for r in rows(root/f'metrics_rank{rank}.jsonl',u)]
 result[str(u)]=dict(rows=len(parts),ranks=8,optimizer_updates=64,collections=16,all=summarize([r for _,r in parts],u,parts),
  by_inner={str(i):summarize([r for _,r in parts if r['inner_completed']==i],u,[(rank,r) for rank,r in parts if r['inner_completed']==i]) for i in (1,2,3,4)})

sources=A/'primary_sources/FlowDPPO_20261031';sources.mkdir(parents=True,exist_ok=True)
req=urllib.request.Request('https://api.github.com/repos/Tencent-Hunyuan/UniRL/commits/main',headers={'User-Agent':'RL-research-readonly'})
try:
 head=json.load(urllib.request.urlopen(req,timeout=30))['sha'];pins=[]
 for rel in ('unirl/algorithms/flowdppo.py','unirl/algorithms/base.py','unirl/sde/kernels.py','FlowDPPO/config.yaml'):
  url=f'https://raw.githubusercontent.com/Tencent-Hunyuan/UniRL/{head}/{rel}';raw=urllib.request.urlopen(url,timeout=30).read();p=sources/rel.replace('/','__');p.write_bytes(raw)
  pins.append(dict(path=rel,url=url,local=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
 primary=dict(repo='https://github.com/Tencent-Hunyuan/UniRL',commit=head,files=pins)
except Exception as e:primary=dict(status='NETWORK_PIN_UNAVAILABLE',error=str(e),official_web_source='https://raw.githubusercontent.com/Tencent-Hunyuan/UniRL/main/unirl/algorithms/flowdppo.py')
value=dict(schema='bounded_native_PPO_blocked_diagnostic_v1',created_at=datetime.now(timezone.utc).isoformat(),windows=result,source_pins=primary,
 method='binary byte seeks then exact64committedrows perrank;1280rollout-windowcache reused only for A-sign eligible denominators;no models/scoring/scene scan',actions=dict(GPU=False,new_evaluations=False,process_signals=False,stage_or_commit=False))
p=A/'FLOW_DIRECT_PROXIMAL_JOURNAL_EVIDENCE_20261031.json';p.write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(dict(path=str(p),summary={u:{k:v['all']['eligible_weighted_counts'][k] for k in ('positive_blocked_rate','negative_blocked_rate','overall_blocked_rate')} for u,v in result.items()},primary_commit=primary.get('commit'))))
