import copy
import importlib
import json
from pathlib import Path


def test_known_D46_owner_requires_sealed_exact_parent_and_registration(monkeypatch,tmp_path):
    from tools.qwen_world_compare import job_worker as w
    from starVLA.rl.qwen_world_compare.contracts import fingerprint,file_sha
    monkeypatch.setattr(w,'D46_ROOT',tmp_path)
    contract=tmp_path/'PARENT_REGISTRATION_v2.json';contract.write_text('{}')
    monkeypatch.setattr(w,'D46_CONTRACT_SHA',file_sha(contract))
    output=tmp_path/'runs'/'D46_D3_FPO';output.mkdir(parents=True)
    body=dict(schema='d44_native_rl_v1',architecture_id='D',parent_model_sha256=w.D46_PARENT_SHA,
        recipe=dict(parent='D_IL46K',arm='D3_FPO',parent_checkpoint=str(tmp_path/'IL46_PARENT'),training_uses_navtest=False,
        parent_registration=dict(path=str(contract),file_sha256=file_sha(contract))),data=dict(NAVTEST_overlap=0))
    def save(value):
        (output/'identity.json').write_text(json.dumps(dict(value,sha256=fingerprint(value))))
    save(body)
    assert w.known_D46_output(output)
    wrong=copy.deepcopy(body);wrong['recipe']['arm']='D4';save(wrong)
    assert not w.known_D46_output(output)
    wrong=copy.deepcopy(body);wrong['parent_model_sha256']=w.D44_PARENT_SHA;save(wrong)
    assert not w.known_D46_output(output)
    wrong=copy.deepcopy(body);wrong['recipe']['parent_registration']['path']=str(tmp_path/'PARENT_REGISTRATION_v1.json');save(wrong)
    assert not w.known_D46_output(output)
    save(body);contract.write_text('{"modified":true}')
    assert not w.known_D46_output(output)


def test_formal_resume_rejects_qualification_and_foreign_checkpoint(monkeypatch,tmp_path):
    from tools.d46_priority_rl import owner
    from starVLA.rl.qwen_world_compare.contracts import file_sha
    import pytest,yaml
    runroot=tmp_path/'formal';runroot.mkdir()
    cp=runroot/'checkpoints/update000000_inner0';cp.mkdir(parents=True)
    (cp/'files.json').write_text('{}');(cp/'COMPLETE').write_text(json.dumps(dict(world=8,files_sha256=file_sha(cp/'files.json'))))
    (cp/'identity.json').write_text(json.dumps(dict(parent_model_sha256=owner.MODEL_SHA,qualification=True,recipe={})))
    (runroot/'latest.json').write_text(json.dumps(dict(checkpoint=str(cp))))
    cfg=tmp_path/'cfg.yaml';cfg.write_text('{}')
    with pytest.raises(ValueError,match='formal IL46'):owner.inspect_latest(dict(output=str(runroot),config=str(cfg)))
    (runroot/'latest.json').write_text(json.dumps(dict(checkpoint=str(tmp_path/'qualification/update000010'))))
    with pytest.raises(ValueError,match='Foreign native'):owner.inspect_latest(dict(output=str(runroot),config=str(cfg)))
