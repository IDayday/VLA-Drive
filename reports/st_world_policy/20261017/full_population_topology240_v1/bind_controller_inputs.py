#!/usr/bin/env python3
"""After the original factory exits successfully, pin its small actual outputs; no DB/tensor reads."""
import argparse
import hashlib
import json
from pathlib import Path
from adapter import bindings,raw,HERE

def main():
    p=argparse.ArgumentParser(description=__doc__);p.parse_args()
    b=bindings(); root=Path(b['factory_root'])
    result=json.loads(raw(root/'controls/RESULT.json'))
    if (result.get('status')!='PREPARATION_COMPLETE' or result.get('training_started') is not False
        or result.get('cli_result',{}).get('status')!='PROVIDER_PREPARATION_COMPLETE'):
        raise ValueError('A successful actual metadata factory result is required')
    def pin(path):return {'path':str(path),'sha256':hashlib.sha256(raw(path)).hexdigest()}
    plan=json.loads(raw(b['plan']['path'],b['plan']['sha256']))
    outputs=result['cli_result']['outputs']; pins={}
    for key,name in [('config','configured/training_config.json'),('teacher_manifest','PROVIDER.json'),
        ('provider_pca_scope','configured/PROVIDER_CHANNEL_SCOPE.json'),('config_lineage','configured/PROVIDER_CONFIG_LINEAGE.json'),
        ('source_migration','SOURCE_MIGRATION.json')]:
        actual=pin(root/name)
        if actual!=outputs[name]:raise ValueError('Original factory output identity differs: '+key)
        pins[key]=actual
    pins.update(base_binding={'path':'/mnt/project/st-world-policy-20261016/bindings/BASE_BINDING.json',
        'sha256':'84544c2242c0c7ecdcbe4450073b23a3c556cad4d822860d5b8652ecee9f1e38'},
        data_manifest={k:plan['data_manifest'][k] for k in ('path','sha256')},
        request_index_complete=plan['inputs']['train_index_complete'],factory_result=pin(root/'controls/RESULT.json'))
    value={'schema':'st_full_index_profile64_controller_inputs_v1','status':'ACTUAL_FACTORY_PINNED_NOT_DISPATCHED',
        'source_sha':b['runtime']['sha'],'pins':pins}
    path=root/'CONTROLLER_INPUTS.json'; encoded=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
    if path.exists():
        if raw(path)!=encoded:raise ValueError('Immutable input conflict')
    else:
        import os
        with path.open('xb') as f:f.write(encoded);f.flush();os.fsync(f.fileno())
    print(json.dumps(pin(path)))
if __name__=='__main__':main()
