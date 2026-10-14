"""Apply the registered consecutive-DEV pause rule to the extra user observation.

The extra checkpoint stays out of model selection. Safety evidence is retained
when the next scheduled full-DEV observation arrives. No optimizer is restarted.
"""
import argparse
import json
from pathlib import Path
import signal
import time

import psutil

from starVLA.rl.constrained_flow_noise.contracts import require_dispatch, fingerprint, file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json
from .analyze_ode import SAFETY


def read_report(path):
    report = json.loads(Path(path).read_text())
    if report['sha256'] != fingerprint({k:v for k,v in report.items() if k!='sha256'}):
        raise ValueError('Changed evaluation report')
    if report['scenes'] != 1696 or report['seed'] != 2101:
        raise ValueError('Full registered DEV with seed2101 required')
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--previous', required=True)
    parser.add_argument('--next-report', required=True)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    host = require_dispatch()
    run, receipt = Path(args.run), Path(args.receipt)
    if receipt.exists():
        raise FileExistsError(receipt)
    previous = read_report(args.previous)
    launch = json.loads(run.with_suffix('.launcher.json').read_text())
    if launch['host'] != host:
        raise ValueError('Safety guard must run on its owned trainer host')
    trainer = psutil.Process(launch['pid'])
    created = trainer.create_time()
    record = dict(status='WAITING_FOR_NEXT_REGISTERED_DEV', previous_report_sha256=previous['sha256'],
                  next_report=str(args.next_report), rule='same registered 0.02 single or two consecutive >=0.005 and positive lower CI',
                  model_selection_schedule_unchanged=True, host=host, pid=trainer.pid,
                  trainer_created=created, source_sha256=file_sha(__file__))
    atomic_json(receipt, record)
    while not Path(args.next_report).exists():
        require_dispatch()
        if not trainer.is_running() or trainer.create_time()!=created:
            record.update(status='TRAINER_EXITED_BEFORE_NEXT_OBSERVATION')
            atomic_json(receipt,record)
            return
        time.sleep(5)
    current = read_report(args.next_report)
    if any(current[k]!=previous[k] for k in ('population_identity','protocol_sha256','baseline_payloads_sha256')):
        raise ValueError('Unmatched safety observations')
    violations=[]
    for name in SAFETY:
        before=previous['safety'][name]['violation_difference']
        after=current['safety'][name]['violation_difference']
        if after['mean']>=.02 or (before['mean']>=.005 and before['CI95'][0]>0 and after['mean']>=.005 and after['CI95'][0]>0):
            violations.append(name)
    record.update(next_report_sha256=current['sha256'], components=violations)
    if violations:
        workers=[]
        for child in trainer.children(recursive=True):
            command=child.cmdline()
            if ('scripts.analysis.constrained_flow_noise.train' in command and '--output' in command
                    and command[command.index('--output')+1]==str(run)):
                workers.append(child)
        if len(workers)!=8:
            raise RuntimeError('Expected exactly eight owned training ranks; no blind signalling')
        for child in workers:
            child.send_signal(signal.SIGUSR1)
        record.update(status='NATIVE_BOUNDARY_STOP_REQUESTED', worker_pids=[p.pid for p in workers],
                      signal='SIGUSR1', requested_unix=time.time(), no_automatic_resume=True)
    else:
        record.update(status='NEXT_OBSERVATION_DID_NOT_TRIGGER_PAUSE')
    atomic_json(receipt,record)
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    main()
