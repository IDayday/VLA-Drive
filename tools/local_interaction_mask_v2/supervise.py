"""Account a local process group and all assigned GPUs, including startup/failure."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from tools.local_interaction_mask_v2.budget import Run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('ledger','run-id','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--gpus',type=int,required=True);p.add_argument('--max-gpu-hours',type=float,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command:raise ValueError('Missing executable')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    identity={'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'command':command,
              'cwd':str(Path.cwd()),'output':str(out.resolve()),'max_gpu_hours':a.max_gpu_hours}
    with Run(a.ledger,a.run_id,identity,a.gpus,resume=a.resume,max_gpu_hours=a.max_gpu_hours) as run:
        (out/'supervisor.json').write_text(json.dumps(identity,indent=2)+'\n')
        with (out/'process.log').open('a',buffering=1) as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            stop=False;requested=None
            def interrupt(signum,frame):
                nonlocal stop
                stop=True
            old={s:signal.signal(s,interrupt) for s in (signal.SIGINT,signal.SIGTERM)}
            try:
                while child.poll() is None:
                    progress=out/'progress.json'
                    step=json.loads(progress.read_text()).get('step',run.step) if progress.exists() else run.step
                    if not run.update(step) or stop:
                        if requested is None:
                            # Workers observe this boundary request; no signal destroys in-flight save.
                            (out/'STOP_REQUESTED').write_text('GPU budget or operator stop; finish current optimizer step and save.\n')
                            requested=time.monotonic()
                        elif time.monotonic()-requested>180:
                            os.killpg(child.pid,signal.SIGTERM)
                    time.sleep(2)
                progress=out/'progress.json'
                if progress.exists():run.update(json.loads(progress.read_text()).get('step',run.step))
                if child.returncode:raise RuntimeError(f'Worker process failed with exit {child.returncode}; see {out}/process.log')
                state=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {'status':'complete'}
                run.terminal_status=state['status']
                if run.terminal_status not in ('complete','paused'):raise RuntimeError('Worker did not complete or save a pause boundary')
            finally:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                for s,handler in old.items():signal.signal(s,handler)


if __name__=='__main__':main()
