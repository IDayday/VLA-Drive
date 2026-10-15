"""Single-cell controller: real resume check then one bounded diagnostic run."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import socket
import subprocess
from iqe.io import atomic_json,read_json,file_hash,lock
from iqe.contracts import require
from .config import load
from .training import resource


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--kind',choices=('M_scene','M_set'),required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--resume',action='store_true');a=p.parse_args();c=load(a.config)
    folder=c.root/'runtime'/a.kind
    with lock(folder/'CONTROLLER.lock'):
        resource(c,'bounded-decision-controller',training=True)
        require((c.root/'existing_audit/COMPLETE.json').exists(),'complete existing-data audit before new training')
        repo=Path(__file__).parents[2]
        identity=dict(host=socket.gethostname(),pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
             config_hash=c.identity,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
             kind=a.kind,device=a.device,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
             started_utc=datetime.now(timezone.utc).isoformat(),automatic_refit=False,old_experiments_restarted=False)
        atomic_json(folder/'CONTROLLER.json',identity)
        import sys
        commands=[('real_resume',[sys.executable,'-m','iqe.decision_selection.qualification','--config',a.config,'--kind',a.kind,'--device',a.device]),
                  ('diagnostic',[sys.executable,'-m','iqe.decision_selection.cli','train-diagnostic','--config',a.config,'--kind',a.kind,'--device',a.device]+(['--resume','auto'] if a.resume else []))]
        for phase,cmd in commands:
            with open(folder/(phase+'.log'),'ab',buffering=0) as log:
                process=subprocess.Popen(cmd,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
                atomic_json(folder/'STATUS.json',dict(status='RUNNING',phase=phase,child_pid=process.pid,command=cmd))
                code=process.wait()
            atomic_json(folder/(phase+'_EXIT.json'),dict(exit_code=code,command=cmd,utc=datetime.now(timezone.utc).isoformat()))
            if code:
                atomic_json(folder/'STATUS.json',dict(status='FAILED',phase=phase,exit_code=code));return code
        atomic_json(folder/'STATUS.json',dict(status='COMPLETE',phases=['real_resume','diagnostic'],refit_started=False))
        return 0


if __name__=='__main__':raise SystemExit(main())
