import copy
import pytest
from starVLA.rl.qwen_world_compare.execution_migration import validate_patch


def identity():
    return dict(qualification=False,world=8,config=dict(lr=1e-4,seed=42,budget=100000),
        execution=dict(backend='zero2',precision='FP32'),source_files={'tools/qwen_world_compare/train_il.py':'old'},
        model_config=dict(depth=20,width=1024),optimizer_ordered_parameter_groups=['native_original'])


def updated(before):
    value=copy.deepcopy(before);value['source_files']['tools/qwen_world_compare/train_il.py']='new'
    value['execution']['execution_tuning']=dict(checkpoint_policy='short_off',cpu_prefetch_depth=2)
    return value


def test_only_named_execution_patch_and_no_qualification_weight_promotion():
    old=identity();new=updated(old)
    assert list(validate_patch(old,new))==['tools/qwen_world_compare/train_il.py']
    new['qualification']=True
    validate_patch(old,new,qualification=True)
    with pytest.raises(ValueError,match='Qualification'):validate_patch(old,new)


@pytest.mark.parametrize('field',['config','model_config','world','optimizer_ordered_parameter_groups'])
def test_data_optimizer_model_budget_or_scope_drift_cannot_be_hidden(field):
    old=identity();new=updated(old);new[field]='changed'
    with pytest.raises(ValueError,match='scientific'):validate_patch(old,new)


def test_unlisted_import_change_or_dependency_deletion_is_rejected():
    old=identity();new=updated(old);new['source_files']['starVLA/model/modules/qwen_world_compare/model.py']='edited_loss'
    with pytest.raises(ValueError,match='Unapproved'):validate_patch(old,new)
    new=updated(old);new['source_files'].clear()
    with pytest.raises(ValueError,match='delete'):validate_patch(old,new)
