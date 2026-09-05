#!/usr/bin/env python3
"""Publish bounded-run evidence without copying checkpoints or any image data."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    paths={
        'real_smoke':'real_smoke_v2/runtime.json',
        'learnability_probe':'learnability_probe_v2.json',
        'config_pair':'config_pair_audit_v2/report.json',
        'input_manifest':'input_cache_v2c/planreg_input_only_manifest.json',
    }
    evidence={}
    for name,relative in paths.items():
        path=a.artifact_root/relative
        raw=path.read_bytes();payload=json.loads(raw)
        evidence[name]=dict(path=str(path.resolve()),sha256=hashlib.sha256(raw).hexdigest())
        if name=='input_manifest':
            payload={k:v for k,v in payload.items() if k not in ('records','rows','entries')}
        (a.output/(name+'.json')).write_text(json.dumps(payload,indent=2)+'\n')
    (a.output/'evidence_manifest.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()
