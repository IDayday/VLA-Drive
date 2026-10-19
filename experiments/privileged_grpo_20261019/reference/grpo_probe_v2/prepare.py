from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
TASK=ROOT.parent
plan=json.loads((ROOT/'PLAN.json').read_text())
source=TASK/plan['data_source']
rows=[json.loads(line) for line in source.open()]
priority=['NC','DAC','TTC','DDC','EP','CF']
def reason(row):return next(k for k in priority if row['base_score']['metrics'][k]<1)
def group(row):return row['log'].rsplit('_',2)[0]
def hashed(row):return hashlib.sha256(('probe-v2-20261019:'+row['token']).encode()).hexdigest()
rows.sort(key=hashed)
smoke=[json.loads(line) for line in (TASK/'grpo_smoke_v1/runs/attempt01/rollouts.jsonl').open()]
smoke_groups={group(r) for r in smoke}

def select(targets,excluded):
    chosen=[];used=set(excluded)
    # Rare primary reasons first; choose at most one scene per source drive.
    for k in ['NC','DDC','EP','TTC','DAC']:
        candidates=[r for r in rows if reason(r)==k and group(r) not in used]
        count=0
        for r in candidates:
            if group(r) in used:continue
            chosen.append(r);used.add(group(r));count+=1
            if count==targets[k]:break
        assert count==targets[k],(k,count,targets[k])
    return sorted(chosen,key=hashed)

dev=select(plan['primary_counts']['internal_dev'],smoke_groups)
train=select(plan['primary_counts']['update'],{group(r) for r in dev})
monitor=[]
for k,n in plan['primary_counts']['train_monitor'].items():monitor += [r for r in train if reason(r)==k][:n]
assert len(train)==64 and len(dev)==32 and len(monitor)==16
assert not {group(r) for r in train}&{group(r) for r in dev}
assign=json.loads((TASK/'splits/log_reason_90_10_v1/ASSIGNMENTS.json').read_text())
assert all(assign['logs'][r['log']]=='train' for r in train+dev)
data={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'seed_label':'probe-v2-20261019','group_rule':"log.rsplit('_',2)[0]",'smoke_groups_excluded_from_dev':sorted(smoke_groups)}
for label,selected in [('train',train),('dev',dev),('train_monitor',monitor)]:
    data[label]=[{'token':r['token'],'log':r['log'],'source_group':group(r),'primary_reason':reason(r)} for r in selected]
    print(label,len(selected),dict(Counter(reason(r) for r in selected)))
with (ROOT/'DATA.json').open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')
