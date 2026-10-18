"""Only the actual installed DeepSpeed serializer delta and changed budget."""
import importlib.util
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'run_cpu_checks.py'
SPEC=importlib.util.spec_from_file_location('affected_ds_fragment_checks',SOURCE)
module=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(module)
runner=module.runner
runner.OUT=HERE
runner.COMMAND+=['-k','ds_fragment or resource_and_scientific_budgets']
os.environ['DS_ACCELERATOR']='cpu'
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
if __name__=='__main__':raise SystemExit(runner.main())
