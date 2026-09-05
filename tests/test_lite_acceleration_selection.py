import copy
from scripts.select_lite_acceleration_layout import compare_candidate


def metric(batch):
    return dict(status='success',oom=False,deadlock=False,nonfinite_count=0,
        peak_allocated_gib=40,peak_reserved_gib=45,read_only_attention_backend='split_sdpa',
        gradient_checkpointing=False,scorer_partitions_per_scene=1,
        sample_exposure_count_including_warmup=20480,sample_exposure_multiset_sha256='same',
        global_batch_size=batch,p90_step_time=3.1,median_step_time=3.,end_to_end_samples_per_second=40,
        training_curve_global_rank_mean=[{'loss/trajectory_loss':10.,'loss/final_score_loss':2.} for _ in range(320 if batch==64 else 160)])


def test_batch128_requires_equal_exposure_and_no_gross_loss_regression():
    control,candidate=metric(64),metric(128)
    assert compare_candidate(control,candidate)['eligible']
    changed=copy.deepcopy(candidate)
    changed['sample_exposure_multiset_sha256']='different'
    assert not compare_candidate(control,changed)['eligible']
    for row in candidate['training_curve_global_rank_mean']:
        row['loss/trajectory_loss']=12.
    assert not compare_candidate(control,candidate)['eligible']


def test_oom_and_memory_limit_are_not_promoted():
    control,candidate=metric(64),metric(128)
    candidate['oom']=True
    assert not compare_candidate(control,candidate)['eligible']
    candidate['oom']=False;candidate['peak_allocated_gib']=72
    assert not compare_candidate(control,candidate)['eligible']
