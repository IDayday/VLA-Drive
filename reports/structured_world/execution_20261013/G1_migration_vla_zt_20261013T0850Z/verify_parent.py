"""Verify and pin the last committed G1 state; never reconstruct missing updates."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import time

OUT = Path(__file__).resolve().parent
RUN = OUT.parent / 'formal_navsim_zt2_G1_G0_v3_io/G1_FULL_UNIFORM'
TAG = 'latest_0080000'
IDENTITY = '8f331622551d1f1c1c439a6d01411affe012a1777a63c61d8525470dc1e3ded5'
folder = RUN / 'checkpoints' / TAG
raw = (folder / 'COMPLETE.json').read_bytes()
complete = json.loads(raw)
assert complete['completed'] == 80000 and complete['identity'] == IDENTITY
assert complete['ranks'] == 8 and len(complete['file_sha256']) == 17
assert (RUN / 'checkpoints/latest').read_text().strip() == TAG
pin = OUT / 'parent_checkpoint' / TAG
pin.mkdir(parents=True, exist_ok=True)

def verify(item):
    name, expected = item
    assert Path(name).name == name
    src = folder / name
    dst = pin / name
    before = src.stat()
    if not dst.exists():
        os.link(src, dst)
    assert src.stat().st_ino == dst.stat().st_ino
    h = hashlib.sha256()
    start = time.monotonic()
    total = 0
    with dst.open('rb') as stream:
        while block := stream.read(8 * 1024**2):
            h.update(block)
            total += len(block)
            delay = total / (128 * 1024**2) - (time.monotonic() - start)
            if delay > 0:
                time.sleep(delay)
    after = src.stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
    assert h.hexdigest() == expected, name
    record = dict(file=name, bytes=after.st_size, inode=after.st_ino,
                  mtime_ns=after.st_mtime_ns, sha256=expected)
    print(json.dumps(record), flush=True)
    return record

started = time.monotonic()
with ThreadPoolExecutor(max_workers=2) as pool:
    files = list(pool.map(verify, complete['file_sha256'].items()))
assert (folder / 'COMPLETE.json').read_bytes() == raw
if not (pin / 'COMPLETE.json').exists():
    os.link(folder / 'COMPLETE.json', pin / 'COMPLETE.json')
result = dict(status='PASS', utc=datetime.now(timezone.utc).isoformat(),
              identity=IDENTITY, tag=TAG, completed=80000,
              source_checkpoint_status=complete['status'], folder=str(folder),
              pin=str(pin), receipt_sha256=hashlib.sha256(raw).hexdigest(),
              files=files, total_bytes=sum(x['bytes'] for x in files),
              seconds=time.monotonic()-started, max_aggregate_read_MiB_s=256,
              new_payload_bytes=0, original_last_logged_update=84873,
              updates_requiring_replay=4873,
              boundary_preservation='Exact saved 80000 state only; later unsaved progress is unavailable.')
tmp = OUT / 'PARENT_CHECKPOINT_VERIFIED.tmp'
tmp.write_text(json.dumps(result, indent=2)+'\n')
os.replace(tmp, OUT / 'PARENT_CHECKPOINT_VERIFIED.json')
print(json.dumps({k:v for k,v in result.items() if k!='files'}), flush=True)
