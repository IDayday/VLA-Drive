"""Small actual cross-node NCCL check; never model acceptance evidence."""
import datetime
import json
import os
from pathlib import Path
import socket
import time
import torch
import torch.distributed as dist

torch.set_num_threads(1)
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
rank, world = dist.get_rank(), dist.get_world_size()
x = torch.tensor(float(rank + 1), device="cuda")
dist.all_reduce(x)
assert x.item() == world * (world + 1) / 2
rows = []
for size in [2**20, 2**24, 2**26]:
    x = torch.ones(size, device="cuda", dtype=torch.float32)
    for i in range(3):
        x.fill_(1)
        dist.all_reduce(x)
    torch.cuda.synchronize()
    started = time.monotonic()
    for i in range(8):
        x.fill_(1)
        dist.all_reduce(x)
    torch.cuda.synchronize()
    seconds = (time.monotonic()-started)/8
    assert torch.all(x == world)
    rows.append({"bytes": size * 4, "seconds": seconds,
                 "effective_gbps": size * 4 / seconds / 1e9})
out = Path(os.environ["PROBE_OUTPUT"])
out.mkdir(exist_ok=True, parents=True)
(out / f"rank{rank}.json").write_text(json.dumps({
    "rank": rank, "world_size": world, "host": socket.gethostname(),
    "gpu": torch.cuda.get_device_name(), "status": "PASS", "timings": rows,
}, indent=2))
dist.destroy_process_group()
