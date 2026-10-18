"""Bounded CPU-only regression for deferred SIGTERM/SIGINT ownership transfer."""
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('bounded_endpoint_checks',HERE.parent/'run_cpu_checks.py')
runner=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
runner.OUT=HERE
runner.COMMAND+=['-k','log_and_post_Popen_receipt_failure or two_loader_slots or complete_owned_session_cleanup']
if __name__=='__main__':raise SystemExit(runner.main())
