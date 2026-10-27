"""Run the archived retirement operator with an explicit audit directory."""
import argparse,importlib.util,sys
from pathlib import Path
p=argparse.ArgumentParser(__doc__);p.add_argument('--audit-dir',required=True)
a,remaining=p.parse_known_args();audit=Path(a.audit_dir).resolve()
if not(audit/'ACTIVE_PROTECTED.json').is_file():p.error('Missing explicit protected-inode manifest')
source=Path(__file__).with_name('retire_states_parallel.py')
spec=importlib.util.spec_from_file_location('retire_training_states_operator',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.ROOT=audit
if remaining[:1]==['--']:remaining=remaining[1:]
sys.argv=[str(source),*remaining];module.main()
