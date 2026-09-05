import json
from types import SimpleNamespace
from unittest.mock import patch
from navsim.planning.training.formal_throughput import FormalThroughputBenchmarkCallback


def test_cycle_time_includes_wait_and_curves_include_warmup(tmp_path):
    cb=FormalThroughputBenchmarkCallback(str(tmp_path/'metrics.json'),global_batch_size=2,
        warmup_steps=1,timed_steps=1,layout_name='synthetic_only',scorer_processes_per_rank=1,
        scorer_partitions_per_scene=1,num_workers=1,gradient_checkpointing=False,
        read_only_attention_backend='split_sdpa')
    module=SimpleNamespace(consume_formal_step_timings=lambda:{'loss/trajectory_loss':2.})
    trainer=SimpleNamespace(global_step=0,is_global_zero=True)
    with patch('torch.cuda.is_available',return_value=False):
        cb.on_train_start(trainer,module)
        for i in range(2):
            cb._batch_start=10.;cb._data_wait=.5;trainer.global_step=i+1
            with patch('time.perf_counter',return_value=11.):
                cb.on_train_batch_end(trainer,module,None,({}, {'token':['a','b']}),i)
        cb.on_train_end(trainer,module)
    m=json.loads((tmp_path/'metrics.json').read_text())
    assert m['samples_per_second']==2.
    assert m['end_to_end_samples_per_second']==2/1.5
    assert m['sample_exposure_count_including_warmup']==4
    assert len(m['training_curve_global_rank_mean'])==2
