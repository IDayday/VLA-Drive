"""Read-only snapshot of the existing evaluation and training monitors."""
from pathlib import Path
import collections
import datetime
import json
import time

ROOT = Path(__file__).resolve().parent
DISPATCH = ROOT.parent / 'navsim_navtest_dispatch_20261011T0225'
JOB = DISPATCH / 'jobs/G3_EVENT_LOCAL_0080000'
progress = json.loads((JOB / 'scores/PROGRESS.json').read_text())
training = json.loads((ROOT / 'TRAINING_INTERFERENCE_LATEST.json').read_text())
dispatch = json.loads((DISPATCH / 'STATE.json').read_text())
attempt = sorted((JOB / 'attempts').iterdir())[-1]
rows = [json.loads(path.read_text()) for path in (attempt / 'shards').glob('*.json')]
failed_launchers = []
for path in (attempt / 'launches').glob('*.json'):
    launch = json.loads(path.read_text())
    stat = Path('/proc', str(launch['pid']), 'stat')
    if stat.exists():
        fields = stat.read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z' and int(fields[49]) != 0:
            failed_launchers.append({'shard': launch['shard'], 'pid': launch['pid'],
                                     'wait_status': int(fields[49])})
ages = [time.time() - datetime.datetime.fromisoformat(row['observed_utc']).timestamp()
        for row in rows if row['status'] == 'INFERENCE' and 'observed_utc' in row]
snapshot = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'evaluation': progress, 'training': training,
            'unique_prediction_records': sum(1 for _ in (JOB / 'records').glob('*.json')),
            'dispatch_status': dispatch['status'], 'attempt': attempt.name,
            'scoring_progress_age_seconds': round(time.time() - (JOB / 'scores/PROGRESS.json').stat().st_mtime, 1),
            'worker_statuses': dict(collections.Counter(row['status'] for row in rows)),
            'failed_local_launchers': failed_launchers,
            'max_inference_report_age_seconds': round(max(ages), 1) if ages else None,
            'inference_failure': (JOB / 'INFERENCE_FAILED.json').exists(),
            'queue_complete': (DISPATCH / 'completed/G3_EVENT_LOCAL_0080000.json').exists()}
with (ROOT / 'OPERATOR_PROGRESS.jsonl').open('a') as stream:
    stream.write(json.dumps(snapshot) + '\n')
print(json.dumps({**{key: value for key, value in snapshot.items() if key != 'training'},
                  'training_updates_and_median_seconds': [
                      [row['group'], row['updates'][-1], round(row['median_seconds'], 3)]
                      for row in training['rows']]}))
