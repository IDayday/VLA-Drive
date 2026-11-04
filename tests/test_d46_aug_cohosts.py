import json
from tools.d46_aug_migration.cohosts import registered_readonly_owner
from starVLA.rl.qwen_world_compare.contracts import file_sha


def test_only_exact_frozen_authorized_command_and_memory_can_share(tmp_path,monkeypatch):
    monkeypatch.setenv('WORLD_COMPARE_TASK','d46_aug_migration_20261104')
    code=tmp_path/'pairs.py';code.write_text('frozen')
    auth=tmp_path/'auth.json';auth.write_text(json.dumps(dict(allowed_hosts=['allowed'])))
    receipt=tmp_path/'receipt.json';args=['python',str(code),'--screen-only']
    receipt.write_text(json.dumps(dict(schema='AUG_registered_frozen_cohost_v1',may_signal_cohost=False,
        maximum_observed_owner_mib=16000,source_files={str(code):file_sha(code)},authorization=str(auth),
        commands=[dict(host='allowed',command=args)])))
    call=lambda argv=args,mem=8000,host='allowed':registered_readonly_owner(argv,mem,receipt=receipt,host=host)
    assert call()
    assert not call(mem=17000) and not call(host='other') and not call(argv=args[:-1])
    code.write_text('changed')
    assert not call()


def test_condition_edit_needs_explicit_frozen_model_receipt(tmp_path,monkeypatch):
    monkeypatch.setenv('WORLD_COMPARE_TASK','d46_aug_migration_20261104')
    code=tmp_path/'pairs.py';code.write_text('model frozen; optimize input delta')
    auth=tmp_path/'auth.json';auth.write_text(json.dumps(dict(allowed_hosts=['allowed'])))
    receipt=tmp_path/'receipt.json';args=['python',str(code),'--base-source','fixed']
    row=dict(schema='AUG_registered_frozen_condition_edit_v1',may_signal_cohost=False,
        model_parameters_frozen=True,maximum_observed_owner_mib=16000,source_files={str(code):file_sha(code)},
        authorization=str(auth),commands=[dict(host='allowed',command=args)])
    receipt.write_text(json.dumps(row))
    assert registered_readonly_owner(args,8000,receipt=receipt,host='allowed')
    row['model_parameters_frozen']=False;receipt.write_text(json.dumps(row))
    assert not registered_readonly_owner(args,8000,receipt=receipt,host='allowed')
