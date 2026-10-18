"""Reuse the bounded endpoint CPU test runner for the new narrow fixture suite."""
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'rollout_scale_endpoint_supervisor_v1/run_cpu_checks.py'
SPEC=importlib.util.spec_from_file_location('bounded_native_smoke_checks',SOURCE)
runner=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(runner)
runner.COMMAND=['/root/miniconda3/envs/ddp/bin/python','-m','pytest','-q','tests/st_world/test_rollout_resume_smoke.py']
runner.OUT=HERE
if __name__=='__main__':raise SystemExit(runner.main())
