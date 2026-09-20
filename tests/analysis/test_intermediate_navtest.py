import copy
import pandas as pd
import pytest
from scripts.analysis.intermediate_navtest import paired_result, validate_pair_identity


def identity():
    return {**dict.fromkeys(("schema_version", "data_root", "observation_assets_sha256", "metric_assets_sha256",
                            "resolved_data_model_config_sha256", "tokens_sha256", "seed", "split",
                            "metric_protocol", "processor", "dependencies"), "same"),
            "numerics": dict.fromkeys(("model_dtype", "qwen_autocast", "action_history_projector_autocast",
                        "attention_backend", "deterministic", "cudnn_deterministic", "cudnn_benchmark",
                        "matmul_tf32", "cudnn_tf32", "bf16_reduced_precision_reduction",
                        "float32_matmul_precision", "flash_deterministic", "cublas_workspace"), "same")}


def test_training_metadata_does_not_redefine_original_ode_identity():
    a = identity(); b = copy.deepcopy(a)
    a.update(checkpoint_sha256="SFT", source_sha256="old_training")
    b.update(checkpoint_sha256="RL", source_sha256="new_training")
    a['numerics']['activation_checkpointing'] = True
    b['numerics']['activation_checkpointing'] = False
    validate_pair_identity(a, b)


@pytest.mark.parametrize('key', ['seed', 'tokens_sha256', 'processor', 'observation_assets_sha256'])
def test_changed_evaluation_input_is_rejected(key):
    a = identity(); b = copy.deepcopy(a); b[key] = 'changed'
    with pytest.raises(ValueError, match=key): validate_pair_identity(a, b)


def test_changed_attention_is_rejected():
    a = identity(); b = copy.deepcopy(a); b['numerics']['attention_backend'] = 'changed'
    with pytest.raises(ValueError, match='attention_backend'): validate_pair_identity(a, b)


def test_paired_complete_counts_and_missing_scene():
    a = pd.DataFrame(dict(token=['a','b','c'], log_name=['x','x','y'], score=[0.,1.,.5]))
    b = a.copy(); b['score'] = [.5,.5,.5]
    _, result = paired_result(a, b, ['a','b','c'])
    assert result['wins'] == result['losses'] == result['ties'] == 1
    assert result['mean_paired_delta'] == 0 and result['zero_to_nonzero'] == 1
    with pytest.raises(ValueError, match='incomplete'):
        paired_result(a.iloc[:2], b.iloc[:2], ['a','b','c'])
