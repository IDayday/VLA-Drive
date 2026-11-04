import copy
import pytest
from tools.qwen_world_compare import evaluate
from tools.d44_native_rl.evaluate import policy_numerics


def test_fp32_text_precision_is_recorded_without_changing_backend_contract():
    old = policy_numerics({'recipe': {'augustus': {}}})
    new = policy_numerics({'recipe': {'augustus': {'text_action_precision': 'float32'}}})
    assert new.pop('model_forward') == 'float32_text_action_bfloat16_frozen_visual'
    assert old.pop('model_forward') == 'bfloat16'
    assert new == old


def test_changed_precision_is_never_an_unlabelled_rl_gain(monkeypatch):
    base = dict(architecture_id='D', cohort='NAVTEST', qualification=False,
                seed=2101, sha256='fixture', model_sha256='same_IL',
                protocol={'sha256': 'same_canonical'}, token_sha256='same',
                numerics={'model_forward': 'bfloat16', 'matmul_tf32': False})
    current = copy.deepcopy(base)
    current['numerics']['model_forward'] = 'float32_text_action_bfloat16_frozen_visual'
    rows = {'scene': {'log': 'log_1_2', 'feedback': {'metrics': {m: 1. for m in evaluate.METRICS}}}}
    monkeypatch.setattr(evaluate, 'load_evaluation', lambda path, **kw: (current if path == 'new' else base, rows))
    monkeypatch.setattr(evaluate, 'paired_statistics', lambda *a: {'mean_delta': 0.})
    with pytest.raises(ValueError, match='numerics'):
        evaluate.summarize('new', 'old')
    result = evaluate.summarize('new', 'old', allow_numeric_policy_change=True)
    assert result['comparison_numeric_policy_change']['isolated_RL_gain'] is False
    assert result['PDMS'] == 100.
    current['numerics']['matmul_tf32'] = True
    with pytest.raises(ValueError, match='numerics'):
        evaluate.summarize('new', 'old', allow_numeric_policy_change=True)
