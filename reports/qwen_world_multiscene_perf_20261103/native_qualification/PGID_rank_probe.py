import os,json
from pathlib import Path
p=dict(pid=os.getpid(),pgid=os.getpgrp(),rank=int(os.environ["RANK"]),expected=int(os.environ["EXPECTED_PGID"]))
assert p["pgid"]==p["expected"]
(Path(os.environ["PROOF_DIR"])/("rank"+str(p["rank"])+".json")).write_text(json.dumps(p))
