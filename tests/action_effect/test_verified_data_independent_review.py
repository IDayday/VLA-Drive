"""Independent validation-inventory tests; synthetic files are not NAVSIM evidence."""
from pathlib import Path
import os
import subprocess
import sys
import types

import pytest

from tools.action_effect import verified_data as validation
from tools.action_effect.common import artifact, digest, read_json, verify_artifact, write_json


def fixture_bank(tmp_path, monkeypatch, module=validation):
    """Exercise real checksum/stat plumbing with a clearly synthetic tiny bank."""
    import tools.action_effect.build_effect_pairs as builder
    def save(name, value):
        path = tmp_path / name; write_json(path, value); return artifact(path)
    admitted = save("attempt/ADMITTED.json", {"fixture": "admitted"})
    canonical = save("attempt/NATIVE.json", {"fixture": "canonical"})
    receipt = save("attempt/COMPLETE.json", {"canonical": canonical, "admission_sha256": admitted["sha256"]})
    feedback = save("FEEDBACK.json", {"fixture": "feedback"})
    entry={"scene_id":"synthetic_scene","source_log":"synthetic_log","role":"fit"}
    reg={"split":"train","max_samples":1,"population":[{"context":entry}]}
    reg["identity"]=digest(reg)
    scene = {**entry,"context_entry":entry,"registration_identity":reg["identity"],
             "candidates": [{"feedback": feedback, "canonical_receipt": receipt}]}
    scenepin = save("SCENE.json", scene)
    scenecomplete = save("SCENE_COMPLETE.json", {"scene_record": scenepin,"registration_identity":reg["identity"]})
    registration = save("REGISTRATION.json", reg)
    completion = save("COMPLETE.json", {"scenes": [scenecomplete],"registration_identity":reg["identity"]})
    manifest = {"schema": "action_effect_pair_dataset_v1", "status": "COMPLETE",
                "registration": registration, "population_completion": completion,
                "registration_identity":reg["identity"],"split":"train","deployment_inputs":False}
    manifest["identity"] = digest(manifest)
    pair = save("PAIR_DATASET.json", manifest)
    repair_manifest={k:v for k,v in manifest.items() if k!="identity"}
    repair_manifest["schema"]="action_effect_repair_dataset_v1"
    repair_manifest["identity"]=digest(repair_manifest)
    repair = save("REPAIR_DATASET.json", repair_manifest)
    pins = [admitted, canonical, receipt, feedback, scenepin, scenecomplete, registration, completion, pair, repair]
    calls = []
    def rebuild(_, **kwargs):
        calls.append("full-validation")
        for pin in pins:
            if pin not in (pair,repair) and kwargs.get('_dependency_observer'):
                kwargs['_dependency_observer'](pin)
            verify_artifact(pin)
        return {"pair_dataset": pair, "repair_dataset": repair}
    monkeypatch.setattr(builder, "build", rebuild)
    monkeypatch.setattr(builder, "read_population", lambda _: ({}, [scene]))
    source = {"fixture_source.py": "v1"}
    monkeypatch.setattr(module, "source_contract", lambda: dict(source))
    return Path(pair["path"]), pins, calls, source


def legacy_module():
    name = "tools.action_effect._independent_validation_9ba58a33"
    m = types.ModuleType(name); m.__package__ = "tools.action_effect"; sys.modules[name] = m
    text = subprocess.check_output(["git", "show", "9ba58a33:tools/action_effect/verified_data.py"], text=True)
    exec(compile(text, "verified_data.py@9ba58a33", "exec"), m.__dict__)
    return m


def test_historical_touch_same_bytes_cannot_replace_immutable_validation_receipt(tmp_path, monkeypatch):
    legacy = legacy_module()
    pair, pins, calls, _ = fixture_bank(tmp_path, monkeypatch, legacy)
    assert legacy.validate_compact_dataset(pair, kind="pair")["reused"] is False
    target = Path(pins[0]["path"]); before = target.read_bytes(); st = target.stat()
    os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
    with pytest.raises(ValueError, match="Immutable content conflict"):
        legacy.validate_compact_dataset(pair, kind="pair")
    assert target.read_bytes() == before and len(calls) == 2


def test_unchanged_inventory_reuses_without_native_reconstruction(tmp_path, monkeypatch):
    pair, pins, calls, _ = fixture_bank(tmp_path, monkeypatch)
    first = validation.validate_compact_dataset(pair, kind="pair")
    second = validation.validate_compact_dataset(pair, kind="pair")
    assert first["reused"] is False and second["reused"] is True
    assert first["inventory_files"] == len(pins) and calls == ["full-validation"]


@pytest.mark.parametrize("index", range(10))
@pytest.mark.parametrize("action", ["modify", "delete"])
def test_every_dependency_modified_or_deleted_is_refused(tmp_path, monkeypatch, index, action):
    pair, pins, calls, _ = fixture_bank(tmp_path, monkeypatch)
    validation.validate_compact_dataset(pair, kind="pair")
    target = Path(pins[index]["path"])
    if action == "modify": target.write_bytes(target.read_bytes() + b"\ncorrupt")
    else: target.unlink()
    with pytest.raises((ValueError, OSError)):
        validation.validate_compact_dataset(pair, kind="pair")


def test_changed_source_contract_forces_reconstruction(tmp_path, monkeypatch):
    pair, _, calls, source = fixture_bank(tmp_path, monkeypatch)
    first = validation.validate_compact_dataset(pair, kind="pair")
    source["fixture_source.py"] = "v2"
    second = validation.validate_compact_dataset(pair, kind="pair")
    assert first["reused"] is False and second["reused"] is False and len(calls) == 2
    assert first["receipt"] != second["receipt"]


@pytest.mark.parametrize("index", range(10))
def test_stat_only_change_revalidates_and_then_reuses(tmp_path, monkeypatch, index):
    pair, pins, calls, _ = fixture_bank(tmp_path, monkeypatch)
    first = validation.validate_compact_dataset(pair, kind="pair")
    target = Path(pins[index]["path"]); st = target.stat()
    os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
    second = validation.validate_compact_dataset(pair, kind="pair")
    assert second["reused"] is False and first["receipt"] != second["receipt"]
    third = validation.validate_compact_dataset(pair, kind="pair")
    assert third["reused"] is True and len(calls) == 2


def test_actual_source_closure_covers_common_helpers():
    assert "tools/action_effect/common.py" in validation.source_contract()
