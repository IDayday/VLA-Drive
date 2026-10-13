"""Preserve a completed host-local scoring queue in shared project storage."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from scripts.analysis.flow_safe_improvement.run_original_evaluation_queue import authorized
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import file_sha, fingerprint
from starVLA.rl.safe_flow_improvement.iteration import require_dispatch


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--round-root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args(); host=authorized(require_dispatch())
    root=Path(a.round_root); queue=(root/'scoring_queue').resolve()
    if not queue.is_relative_to('/var/tmp/flow-safe-improvement-20261013/scoring_queues'):
        raise ValueError('This archiver is limited to the registered host-local Flow scoring queues')
    ranks=[json.loads(p.read_text()) for p in (root/'bank').glob('rank*.json')]
    if (len(ranks)!=8 or any(x['status']!='ASSIGNED_POPULATION_COMPLETE' for x in ranks)
            or not (root/'targets/COMPLETE.json').exists() or not (root/'scoring_complete.json').exists()):
        raise ValueError('Bank, targets and scoring boundary must be complete before archiving')
    lease=(queue/'server.lock').open('a'); fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=json.loads((queue/'status.json').read_text()); manifest=json.loads((queue/'manifest.json').read_text())
    if status['status']!='STOPPED_COMPLETED_REGISTERED_WINDOW' or status['identity_sha256']!=manifest['sha256']:
        raise ValueError('The scorer must have stopped at its completed registered boundary')
    files=sorted(p for p in queue.rglob('*') if p.is_file())
    inventory={str(p.relative_to(queue)):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in files}
    out=Path(a.output)
    if not out.resolve().is_relative_to('/mnt/project/flow-safe-improvement-20261013'):
        raise ValueError('Archive must be in the private shared project run directory')
    out.mkdir(parents=True,exist_ok=False)
    compressor='zstd -T4 -1' if shutil.which('zstd') else 'gzip -1'
    filename='queue.tar.zst' if compressor.startswith('zstd') else 'queue.tar.gz'
    temporary=out/(filename+'.incomplete'); final=out/filename
    command=['tar','--sort=name','-I',compressor,'-cf',str(temporary),'-C',str(queue),'.']
    started=time.monotonic(); subprocess.run(command,check=True)
    # Compare the decompressed archive with every original file while holding
    # the stopped server lease. Do not delete the source even after success.
    comparison=['tar','-I',compressor,'-df',str(temporary),'-C',str(queue)]
    subprocess.run(comparison,check=True,capture_output=True,text=True)
    after={str(p.relative_to(queue)):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
           for p in sorted(queue.rglob('*')) if p.is_file()}
    if after!=inventory:
        raise ValueError('Queue changed during archival; preserve the partial archive and source')
    os.replace(temporary,final)
    atomic_json(out/'source_inventory.json',inventory)
    report=dict(status='COMPLETE_ARCHIVE_BYTE_COMPARISON_PASS_SOURCE_RETAINED',host=host,
        source=str(queue),source_manifest_sha256=manifest['sha256'],source_status=status,
        source_inventory_sha256=fingerprint(inventory),source_files=len(files),source_logical_bytes=sum(x['bytes'] for x in inventory.values()),
        archive=str(final),archive_bytes=final.stat().st_size,archive_sha256=file_sha(final),
        source_deleted=False,commands=[command,comparison],wall_seconds=time.monotonic()-started,
        no_score_or_model_contract_change=True,source_sha256=file_sha(__file__))
    report['sha256']=fingerprint(report);atomic_json(out/'report.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
