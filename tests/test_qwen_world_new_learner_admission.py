import json
import sys
import pytest
from tools.qwen_world_research import job_worker


@pytest.mark.parametrize('slot',['--primary-shared-training','--research-shared-slot'])
@pytest.mark.parametrize('module',[
    'starVLA.rl.qwen_world_research_algorithms.safety_efficiency_entry',
    'starVLA.rl.qwen_world_research_algorithms.preference_entry',
    'starVLA.rl.qwen_world_research_algorithms.lastdet_entry',
    'starVLA.rl.qwen_world_research_algorithms.rescue_entry',
])
def test_new_learner_passes_exact_slot_validation_before_gpu_admission(monkeypatch,tmp_path,slot,module):
    command=['python','-m',module]
    monkeypatch.setattr(sys,'argv',['worker','--receipt-root',str(tmp_path),'--gpus','0',
        '--share-min-free-mib','32768',slot,'--command-json',json.dumps(command),'--cwd',str(tmp_path)])
    monkeypatch.setattr(job_worker,'require_dispatch',lambda **kw:'authorized')
    monkeypatch.setattr(job_worker,'authorization',lambda:({},{}))
    def refuse_without_gpu(*args):raise job_worker.ResourceRefused('test stops before any GPU or child')
    monkeypatch.setattr(job_worker,'_gpu_admission',refuse_without_gpu)
    assert job_worker.main()==75
    assert json.loads((tmp_path/'RESULT.json').read_text())['child_not_started'] is True


def test_fixed_theta_probe_passes_evaluation_slot_before_gpu_admission(monkeypatch,tmp_path):
    command=['python','-m','tools.qwen_world_research.probe_zero_rollouts','run']
    monkeypatch.setattr(sys,'argv',['worker','--receipt-root',str(tmp_path),'--gpus','0',
        '--share-min-free-mib','32768','--evaluation-shared-slot',
        '--command-json',json.dumps(command),'--cwd',str(tmp_path)])
    monkeypatch.setattr(job_worker,'require_dispatch',lambda **kw:'authorized')
    monkeypatch.setattr(job_worker,'authorization',lambda:({},{}))
    def refuse_without_gpu(*args):raise job_worker.ResourceRefused('test stops before any GPU or child')
    monkeypatch.setattr(job_worker,'_gpu_admission',refuse_without_gpu)
    assert job_worker.main()==75
    assert json.loads((tmp_path/'RESULT.json').read_text())['child_not_started'] is True


def test_unknown_entry_cannot_use_research_slot(monkeypatch,tmp_path):
    monkeypatch.setattr(sys,'argv',['worker','--receipt-root',str(tmp_path),'--share-min-free-mib','32768',
        '--research-shared-slot','--command-json',json.dumps(['python','-m','unknown.module']),'--cwd',str(tmp_path)])
    with pytest.raises(ValueError,match='exact learner'):job_worker.main()
