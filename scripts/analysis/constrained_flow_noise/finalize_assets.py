"""Seal the two complete RL_TRAIN caches from actual existing producer receipts."""
import argparse
import json
import hashlib
from pathlib import Path
import time
from starVLA.rl.constrained_flow_noise.contracts import require_dispatch,file_sha,fingerprint
from starVLA.rl.constrained_flow_noise.condition_cache import ConditionStore
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--population',required=True);p.add_argument('--condition-cache',required=True)
    p.add_argument('--metric-producer',action='append',required=True);p.add_argument('--output',required=True)
    p.add_argument('--wait',action='store_true');a=p.parse_args();host=require_dispatch()
    pop=json.loads(Path(a.population).read_text());tokens=pop['train_tokens'];out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    root=Path(a.condition_cache);started=time.monotonic()
    while True:
        records={};protocols=set();receipts={};snapshots={}
        for producer in a.metric_producer:
            path=Path(producer);report_path=path/'report.json'
            if not report_path.exists():continue
            payload=report_path.read_bytes();report=json.loads(payload)
            if report['population_identity']!=pop['identity_sha256']:raise ValueError('Producer population changed')
            protocols.add(report['protocol_sha256']);receipts[str(report_path)]=hashlib.sha256(payload).hexdigest()
            snapshots[path.name]=payload
            rows=json.loads((path/'reused.json').read_text())+[row for log in report['logs'] for row in log['entries']]
            for row in rows:
                token=row['token']
                if token in records and records[token]!=row:raise ValueError('Metric producers disagree on a sealed asset')
                records[token]=row
        condition_ready=any(all((root/f'{prefix}rank{rank}.json').exists() and
                            json.loads((root/f'{prefix}rank{rank}.json').read_text())['status']=='ASSIGNED_COMPLETE'
                            for rank in range(8)) for prefix in ('','zt3_'))
        if set(records)==set(tokens) and condition_ready:break
        status=dict(status='WAITING_FOR_FULL_REAL_ASSETS',metric_scenes=len(records),required_scenes=len(tokens),
                    condition_producer_complete=condition_ready,wall_seconds=time.monotonic()-started,host=host)
        atomic_json(out.with_suffix('.progress.json'),status)
        if not a.wait:raise RuntimeError(json.dumps(status))
        time.sleep(30)
    if len(protocols)!=1:raise ValueError('Producer canonical reward protocols differ')
    cache=ConditionStore(root,json.loads((root/'identity.json').read_text()),readonly=False)
    if not (root/'COMPLETE.json').exists():cache.finalize(tokens)
    else:ConditionStore(root,readonly=True)
    for token in tokens:
        row=records[token]
        if row['log']!=pop['token_logs'][token] or file_sha(row['path'])!=row['sha256']:raise ValueError('Sealed metric asset changed')
    snapshot_root=out.with_suffix('.producer_snapshots');snapshot_root.mkdir(exist_ok=True)
    for name,payload in snapshots.items():(snapshot_root/(name+'.json')).write_bytes(payload)
    result=dict(schema='cfn_full_TRAIN_assets_v1',status='COMPLETE',host=host,population_identity=pop['identity_sha256'],
                count=len(tokens),condition_identity_sha256=cache.identity_sha,condition_complete_sha256=file_sha(root/'COMPLETE.json'),
                protocol_sha256=protocols.pop(),entries=[records[t] for t in tokens],producer_receipts_at_seal=receipts,
                wall_seconds=time.monotonic()-started)
    result['sha256']=fingerprint(result);atomic_json(out,result)
    print(json.dumps(dict(status='COMPLETE',scenes=len(tokens),sha256=result['sha256'])),flush=True)


if __name__=='__main__':main()
