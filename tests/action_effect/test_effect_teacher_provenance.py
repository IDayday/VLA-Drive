"""Own-registration teacher loader tests use synthetic checkpoints, not trained results."""
from dataclasses import asdict
import pytest
import torch
from tools.action_effect.common import artifact,digest,write_json
from tools.action_effect.training import private_state
from tools.action_effect.train_effect_model import load_effect_teacher
from starVLA.model.modules.action_effect.effect_model import ActionEffectModel
from test_model_effect import cfg,spec


def signed(row):return {**row,"identity":digest(row)}


def checkpoint(tmp_path,*,pair_change=None,registration_change=None,completion_change=None):
    reg={"schema":"action_effect_g0_registration_v1","split":"train","max_samples":2,
         "base_checkpoint":{"sha256":"c"*64},"numeric_runtime_hash":"d"*64,"spec":{"score_protocol_hash":"e"*64}}
    if registration_change:reg.update(registration_change)
    reg=signed(reg);write_json(tmp_path/"REGISTRATION.json",reg)
    completion={"registration_identity":reg["identity"],"scenes":[{"fixture":1},{"fixture":2}]}
    if completion_change:completion.update(completion_change)
    write_json(tmp_path/"COMPLETE.json",completion)
    pairs={"schema":"action_effect_pair_dataset_v1","status":"COMPLETE","split":"train","deployment_inputs":False,
        "registration_identity":reg["identity"],"registration":artifact(tmp_path/"REGISTRATION.json"),
        "population_completion":artifact(tmp_path/"COMPLETE.json"),"requested_scenes":2,
        "base_checkpoint":reg["base_checkpoint"],"numeric_runtime_hash":reg["numeric_runtime_hash"],
        "score_protocol_hash":reg["spec"]["score_protocol_hash"],"entries":[{"marked_fixture":True}]}
    if pair_change:pairs.update(pair_change)
    pairs=signed(pairs);write_json(tmp_path/"PAIR_DATASET.json",pairs)
    config=cfg();labels=spec();model=ActionEffectModel(config,labels)
    identity={"schema":"action_effect_small_training_v1","source_commit":"synthetic_checkpoint_NOT_A_TRAINING_RESULT",
              "arm":"delta_feedback","base_binding_hash":"b"*64,"model_config":asdict(config),
              "label_spec":asdict(labels),"data_artifact":artifact(tmp_path/"PAIR_DATASET.json")}
    write_json(tmp_path/"IDENTITY.json",identity)
    p=tmp_path/"step_000001.pt"
    torch.save({"run_identity":digest(identity),"optimizer_step":1,"model":private_state(model)},p)
    write_json(p.with_suffix(".COMPLETE.json"),{"checkpoint":artifact(p),"run_identity":digest(identity),"optimizer_step":1})
    return p,reg["identity"]


def test_own_registration_chain_is_recorded_and_legacy_optional_call_still_loads(tmp_path):
    p,rid=checkpoint(tmp_path)
    model=load_effect_teacher(p,expected_base_binding_hash="b"*64,expected_registration_identity=rid)
    origin=model.provenance["verified_training_origin"]
    assert origin["registration_identity"]==rid
    assert origin["pair_dataset"]==artifact(tmp_path/"PAIR_DATASET.json")
    assert origin["registration"]==artifact(tmp_path/"REGISTRATION.json")
    assert all(not p.requires_grad for p in model.parameters()) and not model.training
    legacy=load_effect_teacher(p,expected_base_binding_hash="b"*64)
    assert "verified_training_origin" not in legacy.provenance


def test_foreign_registration_rejected_even_for_same_base_mode_and_state(tmp_path):
    p,_=checkpoint(tmp_path)
    with pytest.raises(ValueError,match="another campaign registration"):
        load_effect_teacher(p,expected_base_binding_hash="b"*64,expected_registration_identity="f"*64)


@pytest.mark.parametrize("change",[{"status":"PARTIAL"},{"split":"dev"},{"deployment_inputs":True},
    {"schema":"not_pair_bank"},{"requested_scenes":1},{"numeric_runtime_hash":"0"*64},
    {"base_checkpoint":{"sha256":"0"*64}},{"score_protocol_hash":"0"*64}])
def test_incomplete_foreign_role_or_numeric_pair_contract_is_not_own_teacher(tmp_path,change):
    p,rid=checkpoint(tmp_path,pair_change=change)
    with pytest.raises(ValueError):
        load_effect_teacher(p,expected_base_binding_hash="b"*64,expected_registration_identity=rid)


@pytest.mark.parametrize("which",["PAIR_DATASET.json","REGISTRATION.json","COMPLETE.json"])
@pytest.mark.parametrize("action",["delete","modify"])
def test_training_origin_dependency_bytes_cannot_change(tmp_path,which,action):
    p,rid=checkpoint(tmp_path);target=tmp_path/which
    if action=="delete":target.unlink()
    else:target.write_bytes(target.read_bytes()+b"\nchanged")
    with pytest.raises((ValueError,OSError)):
        load_effect_teacher(p,expected_base_binding_hash="b"*64,expected_registration_identity=rid)


@pytest.mark.parametrize("change",[{"scenes":[{"only_one":True}]},{"registration_identity":"f"*64}])
def test_partial_or_foreign_population_completion_rejected(tmp_path,change):
    p,rid=checkpoint(tmp_path,completion_change=change)
    with pytest.raises(ValueError,match="registration/completion"):
        load_effect_teacher(p,expected_base_binding_hash="b"*64,expected_registration_identity=rid)


def test_A_loader_relays_its_actual_repair_registration(tmp_path,monkeypatch):
    from tools.action_effect import train_refiner,train_effect_model
    from test_model_refiner_entry import fixture_dataset,recipe
    dataset,_=fixture_dataset(tmp_path);seen=[]
    def own_teacher(path,**kwargs):
        seen.append(kwargs)
        return ActionEffectModel(cfg(kwargs['allowed_modes'][0]),spec())
    monkeypatch.setattr(train_effect_model,'load_effect_teacher',own_teacher)
    for arm in ('A1','ScoreRefiner'):
        train_refiner.build_model(recipe(),dataset,arm=arm,seed=7,device='cpu',effect_checkpoint='synthetic_own_teacher')
    assert all(x['expected_registration_identity']==dataset.manifest['registration_identity'] for x in seen)
