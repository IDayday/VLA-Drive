"""Only the two cold-import metadata cases; no repeat of original38 tests."""
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
path=HERE.parent/'run_cpu_checks.py'
spec=importlib.util.spec_from_file_location('bounded_cold_route',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
runner=module.runner;runner.OUT=HERE;runner.COMMAND+=['-k','cold_route_initial_process']
if __name__=='__main__':raise SystemExit(runner.main())
