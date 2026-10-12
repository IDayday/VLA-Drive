from pathlib import Path
import datetime,json,statistics,time
ROOT=Path(__file__).resolve().parent
QUEUE=ROOT.parent/'navsim_navtest_dispatch_20261011T0225'
state=json.loads((QUEUE/'STATE.json').read_text())
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':state['status'],
        'key':state.get('request',{}).get('key'),'jobs':{},'training':{}}
for key in ['G1_FULL_UNIFORM_0060000','G3_EVENT_LOCAL_0090000']:
    p=QUEUE/'jobs'/key/'scores/PROGRESS.json'
    row={}
    if p.exists():
        x=json.loads(p.read_text());row={k:x.get(k) for k in ['prediction_scenes_ready','q_final_scored','failed_new_rows']}
    p=QUEUE/'completed'/(key+'.json')
    if p.exists():
        x=json.loads((QUEUE/'jobs'/key/'scores/COMPLETE.json').read_text())
        row.update(complete=json.loads(p.read_text()).get('valid'),
                   q0_PDMS=100*x['q0']['metrics']['score'],q_final_PDMS=100*x['q_final']['metrics']['score'])
    result['jobs'][key]=row
for group,rel in [('G1','formal_navsim_zt2_G1_G0_v3_io/G1_FULL_UNIFORM'),
                  ('G3','formal_navsim_local_G3_G2_v3_io/G3_EVENT_LOCAL')]:
    p=ROOT.parent/rel/'metrics.jsonl'
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-300000));raw=f.read().splitlines()[1:]
    rows=[json.loads(x) for x in raw][-32:]
    result['training'][group]={'update':rows[-1]['updates'],
        'mean_s':round(statistics.mean(x['seconds'] for x in rows),3),
        'median_s':round(statistics.median(x['seconds'] for x in rows),3)}
with (ROOT/'STATUS_SAMPLES.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
print(json.dumps(result))
