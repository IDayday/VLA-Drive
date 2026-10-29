import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from starVLA.rl.qwen_world_compare.profiling import (
    PhaseProfiler, aggregate_profiles, profile_inference,
    validate_profile_run,validate_profile_pair)
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


@pytest.mark.parametrize('updates',[1,3])
def test_successful_early_exit_is_an_incomplete_profile(updates,tmp_path):
    x=torch.ones(2,2)
    with PhaseProfiler(tmp_path,rank=0,use_cuda=False)as profiler:
        for update in range(1,updates+1):
            with profiler.phase('main_action'):x=x@x
            profiler.step(update=update,scenes=32,wall_seconds=.01)
    record=json.loads((tmp_path/'PROFILE_rank00.json').read_text())
    assert record['status']=='INCOMPLETE'
    assert record['schedule_complete']is False
    assert record['actual_active_updates']==max(0,updates-2)
    if updates==3:
        assert record['windows'][0]['active_updates']==1
        assert record['windows'][0]['scheduled_active_updates']==2
    with pytest.raises(ValueError,match='Failed'):
        aggregate_profiles(tmp_path,1)


def fixture_gpu_profile(root,backend,micro=2):
    """Synthetic verifier fixtures, never written to a real experiment root."""
    from starVLA.rl.qwen_world_postcot.contracts import fingerprint
    from starVLA.model.modules.qwen_world_compare.contracts import LOSS_WEIGHTS
    from starVLA.rl.qwen_world_compare.batching import post_positions
    run=root/'run';phases=root/'phases';run.mkdir(parents=True);phases.mkdir()
    cfg=dict(il=dict(global_batch=32,loss_weights=LOSS_WEIGHTS,fm_repeat=1),candidate_refresh=False)
    identity=dict(architecture_id='A',world=8,qualification=True,joint_override=True,
        execution=dict(backend=backend,micro=micro,accum=4//micro),uses_rl=False,
        uses_metric_supervision=True,config=cfg,source_files={'fixture_core.py':'same_source'},
        model_config={'fixture':'same_model'},qwen_config={'fixture':'same_qwen'},
        initialization={'fixture':'same_generic'},parameter_roles={'fixture':'new_module'},
        optimizer_ordered_parameter_groups=[{'fixture':'same_parameters'}],
        data_sha256='same_DATA',teacher_identity_sha256='same_TEACHER',candidate_bank_hash='same_BANK',
        evidence_identity={'fixture':'same_evaluator'},evidence_manifest_sha256='same_EVIDENCE',
        target_manifest_sha256='same_TARGET')
    (run/'IDENTITY.json').write_text(json.dumps(identity))
    (run/'INVOCATION_COMPLETE.json').write_text(json.dumps(dict(updates=4,invocation_start_update=0,
        qualification=True,profile_no_save=True,checkpoint=None)))
    tags=['current_Qwen','scene_C160','P_W_prefix','main_action','pre_readouts',
          'clean_candidate_replay','backward','optimizer_communication']
    for rank in range(8):
        trace=phases/f'rank{rank:02d}_trace00.json'
        trace.write_text(json.dumps(dict(deviceProperties=[{'name':'unit-test-fixture'}],
            traceEvents=[{'cat':'kernel','name':'unit-test-fixture'}])))
        metadata=dict(architecture_id='A',backend=backend,config_sha256=fingerprint(cfg),
            world=8,global_batch=32,micro=micro,accum=4//micro)
        rows=[dict(update=i,global_scenes=32,wall_seconds=.2,peak_allocated_bytes=100,
                   peak_reserved_bytes=120)for i in range(1,5)]
        report=dict(rank=rank,status='COMPLETE',metadata=metadata,enabled=True,CUDA_profiled=True,
            observed_effective_updates=4,updates=rows,schedule=dict(wait=1,warmup=1,active=2,repeat=1),
            windows=[dict(trace=str(trace),active_updates=2,phases={'compare/'+tag:{}for tag in tags})],
            interval_wall_seconds=1.,timing_semantics='unit-test synthetic fixture, not actual GPU evidence')
        (phases/f'PROFILE_rank{rank:02d}.json').write_text(json.dumps(report))
        metrics=[]
        for update in range(1,5):
            tokens=[f'fixture_scene_{rank}_{j}'for j in range(4)]
            metrics.append(dict(updates=update,optimizer_global_steps=update,optimizer_skipped_steps=0,
                sample_ids=[tokens[j:j+micro]for j in range(0,4,micro)],
                post_positions=[post_positions(update-1)[rank]],losses={key:1.0 for key in LOSS_WEIGHTS}))
        (run/f'metrics_rank{rank}.jsonl').write_text(''.join(json.dumps(row)+'\n'for row in metrics))
    return identity


def test_bounded_pair_checks_eight_rank_four_updates_and_preserves_denominator_limit(tmp_path):
    zero=tmp_path/'zero';ddp=tmp_path/'ddp'
    fixture_gpu_profile(zero,'zero2');fixture_gpu_profile(ddp,'ddp')
    result=validate_profile_pair(zero,ddp,architecture_id='A',output=tmp_path/'pair.json')
    assert result['status']=='PASS'and result['actual_profile_optimizer_updates']==8
    assert result['actual_profile_global_scene_exposures']==256
    assert result['micro_batch_matched']is True
    assert result['independently_persisted_denominators']is False
    assert result['native_DDP_resume_qualified']is False
    assert result['formal_backend']=='zero2'
    (ddp/'phases/rank07_trace00.json').unlink()
    with pytest.raises(ValueError,match='trace is missing'):
        validate_profile_pair(zero,ddp,architecture_id='A')


def test_bounded_profile_rejects_cpu_trace_and_not_effective_optimizer_update(tmp_path):
    root=tmp_path/'profile';fixture_gpu_profile(root,'zero2')
    trace=root/'phases/rank00_trace00.json';trace.write_text(json.dumps({'traceEvents':[]}))
    with pytest.raises(ValueError,match='CPU-only'):
        validate_profile_run(root,architecture_id='A',backend='zero2')
    trace.write_text(json.dumps(dict(deviceProperties=[{'name':'unit-test'}],traceEvents=[{'cat':'kernel'}])))
    metrics=root/'run/metrics_rank3.jsonl';rows=[json.loads(line)for line in metrics.read_text().splitlines()]
    rows[-1]['optimizer_global_steps']=3;metrics.write_text(''.join(json.dumps(row)+'\n'for row in rows))
    with pytest.raises(ValueError,match='unskipped'):
        validate_profile_run(root,architecture_id='A',backend='zero2')


def test_bounded_pair_reports_microbatch_difference_and_rejects_target_change(tmp_path):
    zero=tmp_path/'zero';ddp=tmp_path/'ddp'
    fixture_gpu_profile(zero,'zero2',2);fixture_gpu_profile(ddp,'ddp',1)
    assert validate_profile_pair(zero,ddp,architecture_id='A')['micro_batch_matched']is False
    path=ddp/'run/IDENTITY.json';data=json.loads(path.read_text());data['target_manifest_sha256']='changed'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='target_manifest'):
        validate_profile_pair(zero,ddp,architecture_id='A')

