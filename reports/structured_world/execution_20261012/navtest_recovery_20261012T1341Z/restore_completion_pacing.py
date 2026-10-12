"""Restore this bounded evaluation's pacing only; never control GPU processes."""
from pathlib import Path
import datetime
import fcntl
import json
import os
import time

ROOT=Path(__file__).resolve().parent
QUEUE=ROOT.parent/'navsim_navtest_dispatch_20261011T0225'
POLICY=QUEUE/'COHOST_COST_POLICY.json'
decision=json.loads((ROOT/'PACING_DECISION.json').read_text())
deadline=time.time()+14400
reason=None
while reason is None:
    completed=[]
    for key in decision['scope']:
        p=QUEUE/'completed'/(key+'.json')
        completed.append(p.exists() and json.loads(p.read_text()).get('valid') is True)
    state=json.loads((QUEUE/'STATE.json').read_text())
    if all(completed):reason='Both registered full Navtest jobs valid and complete'
    elif state.get('status')=='FAILED':reason='Dispatcher failure; restore conservative pacing'
    elif time.time()>=deadline:reason='Four-hour restoration watchdog reached'
    else:time.sleep(15)

with (ROOT/'pacing_policy.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    current=json.loads(POLICY.read_text())
    ours=current.get('completion_pacing_selection',{}).get('utc')==decision['utc']
    if ours:
        current['owned_training_duty_by_host'].update(decision['restore_duties_after_scope'])
        current.pop('completion_pacing_selection')
        temp=POLICY.with_name(POLICY.name+'.'+str(os.getpid())+'.tmp')
        temp.write_text(json.dumps(current,indent=2,ensure_ascii=False)+'\n');temp.replace(POLICY)
    record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'restored':ours,'reason':reason,'foreign_override_preserved':not ours,
            'training_or_GPU_workers_signaled':False}
    (ROOT/'PACING_RESTORATION_COMPLETE.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
