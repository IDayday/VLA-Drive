"""Launch one torchrun job across the two authorized hosts, shared immutable worktree."""
import argparse
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--remote',required=True);p.add_argument('--master',required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--gpus-per-host',type=int,default=8);p.add_argument('--output',required=True);p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();command=a.command[1:] if a.command[:1]==['--'] else a.command
    if a.remote!='training-vla-zt2':raise ValueError('Unauthorized host')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);remote_script=out/'remote_entry.sh'
    common=[sys.executable,'-m','torch.distributed.run','--nnodes=2',f'--nproc_per_node={a.gpus_per_host}',f'--master_addr={a.master}',f'--master_port={a.port}']
    # Exact new child PID is recorded; cleanup never matches another user's job by name.
    variables={k:os.environ[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','NO_ALBUMENTATIONS_UPDATE','PYTHONPATH','NCCL_DEBUG','NCCL_SOCKET_IFNAME') if k in os.environ}
    remote_script.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(Path.cwd()))+'\n'+
        '\n'.join('export '+k+'='+shlex.quote(v) for k,v in variables.items())+'\n'+
        'echo $$ > '+shlex.quote(str(out/'remote_torchrun.pid'))+'\nexec '+shlex.join(common+['--node_rank=1']+command)+'\n')
    remote_log=(out/'remote_process.log').open('a')
    remote=subprocess.Popen(['ssh','-o','BatchMode=yes',a.remote,'bash '+shlex.quote(str(remote_script))],stdout=remote_log,stderr=subprocess.STDOUT)
    local=subprocess.Popen(common+['--node_rank=0']+command)
    interrupted=False
    def stop(signum,frame):
        nonlocal interrupted
        interrupted=True
    old={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        while local.poll() is None or remote.poll() is None:
            if interrupted or local.poll() not in (None,0) or remote.poll() not in (None,0):
                if local.poll() is None:local.terminate()
                if remote.poll() is None:
                    path=out/'remote_torchrun.pid'
                    if path.exists():
                        pid=int(path.read_text())
                        check="from pathlib import Path; import os,signal; p="+str(pid)+"; f=Path('/proc')/str(p)/'cmdline'; s=f.read_bytes() if f.exists() else b''; expected="+repr(str(a.port).encode())+"; assert not s or (b'torch.distributed.run' in s and expected in s); os.kill(p,signal.SIGTERM) if s else None"
                        subprocess.run(['ssh','-o','BatchMode=yes',a.remote,shlex.join([sys.executable,'-c',check])],check=False)
                break
            time.sleep(1)
        code=local.wait(timeout=30);rcode=remote.wait(timeout=30)
        if code or rcode or interrupted:raise SystemExit(code or rcode or 143)
    finally:
        for child in (local,remote):
            if child.poll() is None:child.terminate()
        remote_log.close()
        for signum,handler in old.items():signal.signal(signum,handler)


if __name__=='__main__':main()
