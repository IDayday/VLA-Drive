"""Tiny real D deployment closure; no historical IL recovery files required."""
import copy
import json
import shutil

import pytest
import torch

from test_loading_deployment import (fixture_runtime, TinyModel, save_IL,
                                     save_RL, inputs)
from test_evaluation import write_eval
from starVLA.rl.qwen_world_compare import deployment as old
from starVLA.rl.qwen_world_compare.contracts import fingerprint, file_sha
from starVLA.rl.qwen_world_compare.policy_adapter import FourFrameworkRecipeActor
from starVLA.rl.qwen_world_postcot.rl_runtime import expected_cleanx_recipe
from starVLA.rl.qwen_world_abcd_research import deployment, parents
from tools.qwen_world_abcd_research import evaluate


def parent_fixture(tmp_path, monkeypatch):
    source = tmp_path/'retired_native'
    save_IL(source, 'D', updates=12000)
    root = tmp_path/'selected_IL_export'
    meta = old.export_policy(source, root)
    # This tiny test fixture describes a formal parent. No real 2B asset or
    # original recovery seal is changed or synthesized by production loading.
    meta['IL_parent']['identity'].update(qualification=False, world=8)
    meta['qualification'] = False
    (root/'export.json').write_text(json.dumps(meta))
    (root/'COMPLETE').write_text(json.dumps(dict(export_sha256=file_sha(root/'export.json'))))
    shutil.rmtree(source)
    monkeypatch.setattr(parents, 'model_from_identity',
        lambda identity, train_heads=True:TinyModel(identity['architecture_id'], train_heads))
    assert not source.exists()
    return root, meta


def test_selected_export_survives_missing_native_and_strict_reexport(tmp_path, monkeypatch):
    parent, _ = parent_fixture(tmp_path, monkeypatch)
    model, meta = deployment.load_export(parent)
    assert not __import__('pathlib').Path(meta['IL_parent']['path']).exists()
    assert deployment.parent_export_path(meta) == str(parent.resolve())
    assert meta['IL_parent']['export_parent']['model_sha256'] == meta['model_sha256']
    features, ego, noise = inputs('D'), torch.randn(1, 8), torch.randn(1, 8, 3)
    expected = model.predict(features, ego, noise)
    output = tmp_path/'new_standalone'
    sealed = deployment.export_policy(parent, output)
    restored, again = deployment.load_export(output)
    assert sealed == again and sealed['research_schema'] == deployment.RESEARCH_SCHEMA
    assert deployment.parent_export_path(again) == str(parent.resolve())
    assert not hasattr(restored, 'readouts')
    for key, value in expected.items():
        torch.testing.assert_close(restored.predict(features, ego, noise)[key], value, atol=0, rtol=0)
    with pytest.raises(FileExistsError):
        deployment.export_policy(parent, output)
    invalid = copy.deepcopy(again)
    invalid['parent_export'] = str(tmp_path/'unselected_export')
    with pytest.raises(ValueError, match='disagree'):
        deployment.parent_export_path(invalid)


def test_native_RL_unmerged_exact_D_scope_parity_and_wrong_parent_rejected(tmp_path, monkeypatch):
    parent_path, _ = parent_fixture(tmp_path, monkeypatch)
    model, parent = parents.load_parent_model(parent_path)
    actor = FourFrameworkRecipeActor(model, expected_cleanx_recipe())
    with torch.no_grad():
        for name, parameter in actor.named_parameters():
            if name.endswith('.lora_B'):
                parameter.add_(torch.randn_like(parameter)*.001)
    checkpoint = tmp_path/'native_RL'
    save_RL(checkpoint, actor, parent)
    output = tmp_path/'RL_export'
    meta = deployment.export_policy(parent_path, output, checkpoint)
    native, _ = deployment.load_native_policy(parent_path, checkpoint)
    restored, _ = deployment.load_export(output)
    assert meta['LoRA_merge'] is False and meta['RL_updates'] == 4
    assert any(name.endswith('.lora_B') for name in restored.state_dict())
    features, ego, noise = inputs('D'), torch.randn(1, 8), torch.randn(1, 8, 3)
    reference = actor.action_expert.predict(features, ego, noise)
    for candidate in (native, restored):
        actual = candidate.predict(features, ego, noise)
        for key, value in reference.items():
            torch.testing.assert_close(actual[key], value, atol=0, rtol=0)
    inspect = deployment.inspect_RL_checkpoint
    bad = copy.deepcopy(inspect(checkpoint))
    bad['identity']['IL_parent']['export_parent']['model_sha256'] = 'wrong'
    monkeypatch.setattr(deployment, 'inspect_RL_checkpoint', lambda _:bad)
    with pytest.raises(ValueError, match='exact selected'):
        deployment.load_native_policy(parent_path, checkpoint)


def test_new_selection_discloses_Test_use_and_keeps_canonical_full_denominator(tmp_path, monkeypatch):
    parent_path, meta = parent_fixture(tmp_path, monkeypatch)
    selected = dict(deployment_parent=str(parent_path), deployment_model_sha256=meta['model_sha256'],
        native_model_sha256=meta['IL_parent']['model_sha256'], updates=12000,
        native_export_parity=dict(status='PASS'))
    selection = dict(schema=parents.SELECTION_SCHEMA, Test_used_for_research_selection=True,
        immutable_for_registered_runs=True, selected=dict(D=selected))
    selection['sha256'] = fingerprint(selection)
    path = tmp_path/'selection.json';path.write_text(json.dumps(selection))
    _, enriched = deployment.load_export(parent_path)
    bound = evaluate._selection_binding(path, enriched)
    assert bound['Test_used_for_parent_selection'] is True
    assert bound['NAVTEST_used_for_training_scene_selection'] is False
    assert bound['original_100k_DEV_only_selection_applicable'] is False
    selection['Test_used_for_research_selection'] = False
    selection.pop('sha256');selection['sha256'] = fingerprint(selection)
    path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match='user-authorized'):
        evaluate._selection_binding(path, enriched)
    left, right = tmp_path/'left', tmp_path/'right'
    population = write_eval(left, (.8, .7, 1.));write_eval(right, (.75, .65, 1.))
    summary = evaluate.summarize(left, right, population=population)
    assert summary['scenes'] == summary['scene_denominator'] == 3
    assert summary['paired']['mean_delta'] == pytest.approx(.1/3)
    (left/'scenes/s2.json').unlink()
    with pytest.raises(ValueError, match='incomplete'):
        evaluate.summarize(left, population=population)
