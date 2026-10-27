import json
import pytest
from starVLA.rl.qwen_world_postcot.contracts import fingerprint
from starVLA.rl.qwen_world_postcot.snapshot_navtest import policy_binding, validate_snapshot


@pytest.mark.parametrize('scope', [None, 'ATTENTION_LORA', 'ATTN_LORA_MEMORY', 'TIME_MOD_LORA'])
def test_exact_new_snapshot_and_reject_changed_model_or_population(tmp_path, scope):
    tokens = [str(i) for i in range(12146)]
    parent = dict(path='parent', model_sha256='native', seal=dict(updates=52000 if scope is None else 40000, files_sha256='ilfiles'), identity=dict(qualification=False))
    export = dict(model_sha256='deployed', qualification=False, RL_parent=None)
    if scope:
        export.update(RL_checkpoint='rl', RL_updates=18600, RL_checkpoint_files_sha256='rlfiles',
                      RL_parent=dict(purpose='formal', schema='world_postcot_cleanx_canonical_RL_v1' if scope=='ATTENTION_LORA' else 'world_postcot_cleanx_scope_RL_v1', recipe=dict(adaptation=scope)))
    expected = dict(schema='world_user_snapshot_NAVTEST_v1', **policy_binding(parent, export),
        IL_parent_checkpoint=parent['path'], IL_parent_model_sha256='native', model_sha256='deployed',
        data_sha256='data', token_sha256=fingerprint(tokens), population_sha256='population', protocol_sha256='v1',
        output=str(tmp_path/'out'), seed=2101, count=12146, prediction_type='x0',
        inference='single_candidate_cleanx_K10_shift5', used_for_parent_selection=False, used_for_actor_update=False)
    reg=tmp_path/'reg.json'
    def save(row):
        row=dict(row);row['sha256']=fingerprint(row);reg.write_text(json.dumps(row))
    kwargs=dict(parent=parent, export=export, data_sha256='data', tokens=tokens, population_sha256='population', protocol_sha256='v1', output=tmp_path/'out', seed=2101)
    save(expected);validate_snapshot(reg, **kwargs)
    for key, value in [('model_sha256','different'), ('updates',1), ('training_scope','other'), ('seed',2102), ('used_for_actor_update',True)]:
        save(dict(expected, **{key:value}))
        with pytest.raises(ValueError):validate_snapshot(reg, **kwargs)
    save(expected)
    with pytest.raises(ValueError):validate_snapshot(reg, **dict(kwargs,tokens=tokens[:16]))


def test_diagnostic_weights_cannot_become_formal_test():
    with pytest.raises(ValueError):
        policy_binding(dict(identity=dict(qualification=True)), dict(qualification=True))
