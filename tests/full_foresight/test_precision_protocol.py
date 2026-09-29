from types import SimpleNamespace
import torch
from tools.foresight.deployment_precision import describe


def test_live_training_and_master_deployment_are_distinct():
    model=torch.nn.Linear(2,3).bfloat16()
    model.foresight_config=SimpleNamespace(num_queries=144)
    live=describe(model,'live BF16','in-memory')
    assert live['parameter_dtype_elements']=={'torch.bfloat16':9}
    assert 'BF16 autocast' in live['compute_policy']
    model.float();model.inference_fp32=True
    restored=describe(model,'FP32 masters','load_student')
    assert restored['parameter_dtype_elements']=={'torch.float32':9}
    assert restored['compute_policy']=='FP32, autocast disabled'
    assert live['parameter_source']!=restored['parameter_source']
