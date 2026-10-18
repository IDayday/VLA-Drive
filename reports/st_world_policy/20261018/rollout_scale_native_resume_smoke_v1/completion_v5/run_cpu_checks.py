"""Run only the completion lifecycle delta within the existing CPU budget."""
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'run_cpu_checks.py'
SPEC=importlib.util.spec_from_file_location('completion_cpu_checks',SOURCE)
module=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(module)
runner=module.runner;runner.OUT=HERE
runner.COMMAND=['/root/miniconda3/envs/ddp/bin/python','-m','pytest','-q','tests/st_world/test_rollout_resume_completion_v5.py']
if __name__=='__main__':raise SystemExit(runner.main())
