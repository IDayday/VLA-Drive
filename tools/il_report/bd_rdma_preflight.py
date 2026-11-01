"""Read-only container RDMA preflight; readiness still needs a real NCCL test."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import socket
import time


def inspect():
    verbs=[]
    for path in sorted(Path('/sys/class/infiniband_verbs').glob('uverbs*/dev')):
        node=Path('/dev/infiniband')/path.parent.name
        row=dict(name=path.parent.name,major_minor=path.read_text().strip(),node=str(node),present=node.exists(),openable=False)
        if node.exists():
            try:
                fd=os.open(node,os.O_RDWR|os.O_CLOEXEC);os.close(fd);row['openable']=True
            except OSError as error:row['open_errno']=error.errno
        verbs.append(row)
    try:ctypes.CDLL('libibverbs.so.1');library=True
    except OSError:library=False
    ports=[]
    for device in sorted(Path('/sys/class/infiniband').glob('*')):
        for port in sorted((device/'ports').glob('*')):
            def read(name):
                p=port/name
                return p.read_text().strip()if p.is_file()else None
            ports.append(dict(device=device.name,port=port.name,state=read('state'),rate=read('rate'),link_layer=read('link_layer')))
    cgroup=Path('/sys/fs/cgroup/devices/devices.list')
    allow=cgroup.read_text().splitlines()if cgroup.exists()else None
    return dict(schema='bd_container_rdma_preflight_v1',at_unix=time.time(),host=socket.gethostname(),
        uverbs=verbs,libibverbs_loadable=library,ports=ports,device_cgroup_v1_allow=allow,
        container_prerequisites_ready=library and any(row['openable']for row in verbs),
        actual_RDMA_NCCL_verified=False,changes_made=False)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path);a=p.parse_args();result=inspect()
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        if a.output.exists():raise FileExistsError('Keep prior preflight evidence; use a new attempt')
        a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
    return 0 if result['container_prerequisites_ready']else 2


if __name__=='__main__':raise SystemExit(main())
