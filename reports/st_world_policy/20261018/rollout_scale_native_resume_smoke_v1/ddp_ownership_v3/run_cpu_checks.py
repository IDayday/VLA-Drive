"""Only the DDP ownership delta and changed budget, bounded CPU fixtures."""
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'run_cpu_checks.py'
SPEC=importlib.util.spec_from_file_location('affected_smoke_checks',SOURCE)
module=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(module)
runner=module.runner
runner.OUT=HERE
runner.COMMAND+=['-k','ddp_ownership or resource_and_scientific_budgets']
if __name__=='__main__':raise SystemExit(runner.main())
