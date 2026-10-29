import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from starVLA.rl.qwen_world_compare.profiling import (
    PhaseProfiler, aggregate_profiles, profile_inference)
from tools.qwen_world_compare.profile import training_command


def test_finite_cpu_phase_trace_and_explicit_non_gpu_semantics(tmp_path):
    x=torch.randn(8,8,requires_grad=True)
    with PhaseProfiler(tmp_path,rank=0,use_cuda=False,metadata=dict(architecture='tiny_test'))as profile:
        for update in range(1,5):
            with profile.phase('main_action'):
                loss=(x@x).square().mean()
            with profile.phase('backward'):
                loss.backward();x.grad=None
            profile.step(update=update,scenes=32,wall_seconds=.01,data_wait_seconds=.001)
    result=json.loads((tmp_path/'PROFILE_rank00.json').read_text())
    assert result['status']=='COMPLETE'and result['observed_effective_updates']==4
    assert result['global_scene_exposures']==128
    assert result['CUDA_profiled']is False and result['exclusive_GPU_busy_hours']is None
    assert result['shared_worker_GPU_wall_hours']is None
    assert len(result['windows'])==1
    assert Path(result['windows'][0]['trace']).is_file()
    phases=result['windows'][0]['phases']
    assert phases['compare/main_action']['calls']==2
    assert phases['compare/main_action']['device_inclusive_ms']is None
    assert phases['compare/backward']['cpu_inclusive_ms']>0
    assert 'inclusive' in result['timing_semantics']


def test_incomplete_or_failed_rank_profiles_are_not_complete(tmp_path):
    with pytest.raises(ValueError,match='All rank'):
        aggregate_profiles(tmp_path,2)
    for rank in range(2):
        with PhaseProfiler(tmp_path,rank=rank,use_cuda=False,enabled=False,metadata={'same':'identity'})as profile:
            for update in range(1,5):
                profile.step(update=update,scenes=32,wall_seconds=(rank+1)*.1)
    result=aggregate_profiles(tmp_path,2)
    assert result['logical_global_scene_exposures']==128  # not256 after summing ranks
    assert result['observed_effective_updates']==4
    assert result['median_update_wall_seconds']==.2
    assert result['global_scenes_per_second']==160
    assert result['peak_allocated_bytes']is None
    record=json.loads((tmp_path/'PROFILE_rank01.json').read_text());record['status']='FAILED'
    (tmp_path/'PROFILE_rank01.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='Failed'):
        aggregate_profiles(tmp_path,2)


def test_profile_failure_is_attributed_without_overwriting(tmp_path):
    with pytest.raises(RuntimeError,match='injected'):
        with PhaseProfiler(tmp_path,rank=0,use_cuda=False,enabled=False)as profile:
            profile.step(update=1,scenes=32,wall_seconds=.1)
            raise RuntimeError('injected failure')
    record=json.loads((tmp_path/'PROFILE_rank00.json').read_text())
    assert record['status']=='FAILED'and record['observed_effective_updates']==1
    assert 'injected failure'in record['error']
    with pytest.raises(FileExistsError):
        with PhaseProfiler(tmp_path,rank=0,use_cuda=False,enabled=False):pass


def test_profile_does_not_accept_duplicate_or_nonfinite_updates(tmp_path):
    with pytest.raises(ValueError,match='continuous'):
        with PhaseProfiler(tmp_path,use_cuda=False,enabled=False)as profile:
            profile.step(update=2,wall_seconds=.1)
            profile.step(update=2,wall_seconds=.1)
    with pytest.raises(ValueError,match='nonnegative'):
        with PhaseProfiler(tmp_path/'other',use_cuda=False,enabled=False)as profile:
            profile.step(update=1,wall_seconds=float('nan'))


def test_profile_training_command_calls_existing_disposable_qualifier(tmp_path):
    args=SimpleNamespace(config='configs/qwen_world_compare/arch_a.yaml',output=str(tmp_path),
        updates=4,backend='ddp',micro_batch=1,evidence_override=None)
    command=training_command(args)
    assert '--qualification'in command and '--joint'in command and '--profile-no-save'in command
    assert command[command.index('--stop-after')+1]=='4'
    assert command[command.index('--profile-steps')+1]=='2'
    assert command[command.index('--backend')+1]=='ddp'
    assert '--resume'not in command
    args.updates=2
    with pytest.raises(ValueError,match='finite4'):
        training_command(args)


class TinyPolicy(nn.Module):
    def __init__(self):super().__init__();self.calls=0
    def encode_current(self,inputs,ego):return inputs
    def prepare_prefix(self,context,auxiliary=False):
        assert auxiliary is False
        return context
    def denoise_x0(self,state,sigma,prefix):
        self.calls+=1
        return torch.zeros_like(state)+prefix['legal_current']*.1


def test_inference_profile_runs_cleanx_K10_without_scoring_or_teacher():
    policy=TinyPolicy()
    result=profile_inference(policy,{'legal_current':torch.tensor(.3)},
                             torch.zeros(1,8),torch.randn(1,8,3),repeats=2,warmup=1)
    assert result['status']=='COMPLETE'and result['scorer_calls']==0
    assert result['logical_single_candidate_inferences']==3
    assert result['denoiser_forward_calls']==policy.calls==30
    assert len(result['rows'])==2
    assert all(len(row['denoiser_seconds'])==10 for row in result['rows'])
    assert result['peak_allocated_bytes']is None
    assert result['K10_from_prefix_median_seconds']>0

