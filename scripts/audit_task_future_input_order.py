#!/usr/bin/env python3
"""Read-only audit of the actual cache enumeration order on each training host."""
import argparse
import hashlib
import json
from pathlib import Path
import socket


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','split-audit','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    manifest=json.loads((a.cache/'planreg_input_only_manifest.json').read_text())
    split=json.loads(a.split_audit.read_text())
    allowed={row['token'] for row in manifest['rows']}
    assert len(allowed)==manifest['record_count']
    orders=[]
    for field in ('train_logs','val_logs'):
        # Same ordered log list and directory enumeration as CacheOnlyDataset.
        tokens=[]
        for log in split[field]:
            path=a.cache/log
            if path.is_dir():tokens.extend(p.name for p in path.iterdir() if p.name in allowed)
        orders.append(tokens)
    combined=orders[0]+orders[1]
    assert set(combined)==allowed and len(combined)==len(allowed)
    assert not set(orders[0])&set(orders[1])
    sha=lambda x:hashlib.sha256('\n'.join(x).encode()).hexdigest()
    report=dict(hostname=socket.gethostname(),cache=str(a.cache.resolve()),records=len(combined),
                train_records=len(orders[0]),val_records=len(orders[1]),overlap=0,
                actual_enumeration_order_sha256=sha(combined),token_set_sha256=sha(sorted(combined)))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
