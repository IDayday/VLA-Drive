#!/usr/bin/env python3
"""Collect immutable completed-run evidence; never invent missing results."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    paths={
        'layout':'formal_training_layout_lock.json',
        'throughput':'throughput/16x4/metrics.json',
        'certified_input':'input_cache_v2d_certified/planreg_input_only_manifest.json',
        'query_head_cost':'query_head_cost.json',
    }
    for host in ('local','training-vla-zt2','training-vla-zt3','training-rl-zt4'):
        paths['input_order_'+host]='input_order_'+host+'.json'
    for variant in ('base','vqa'):
        paths[variant+'_run_identity']=f'formal_runs/formal_task_future_lite_{variant}_init_wm_seed0/run_metadata/formal_run_identity.json'
        paths[variant+'_resolved_config_audit']=f'formal_runs/formal_task_future_lite_{variant}_init_wm_seed0/run_metadata/formal_config_pair_audit.json'
    values={};manifest={}
    for label,relative in paths.items():
        source=a.root/relative
        raw=source.read_bytes()  # absent evidence is a hard error
        values[label]={k:v for k,v in json.loads(raw).items() if k not in ('rows','records','entries')}
        manifest[label]=dict(path=str(source.resolve()),sha256=hashlib.sha256(raw).hexdigest())
    orders=[v['actual_enumeration_order_sha256'] for k,v in values.items() if k.startswith('input_order_')]
    assert len(set(orders))==1
    assert values['base_run_identity']['shared_init_sha256']==values['vqa_run_identity']['shared_init_sha256']
    a.output.mkdir(parents=True)
    for label,value in values.items():(a.output/(label+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    (a.output/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output),artifact_count=len(values))))


if __name__=='__main__':main()
