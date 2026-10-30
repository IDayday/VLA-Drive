import argparse
import json
from pathlib import Path

import pytest

from starVLA.rl.qwen_world_postcot.contracts import file_sha, fingerprint
from starVLA.rl.qwen_world_postcot.selected_parent_runtime import (
    INSTRUCTION, SCHEMA, expected_recipe, validate_selection,
)
from starVLA.rl.qwen_world_postcot.scope_contracts import MEMORY, expected_scope_recipe
from tools.qwen_world_postcot import job_worker


def sealed(path, body):
    value = dict(body, sha256=fingerprint(body))
    path.write_text(json.dumps(value))
    return value


def fixture(tmp_path):
    config = tmp_path/'recipe.yaml'; config.write_text('registered recipe')
    manifest = tmp_path/'TRAIN.json'; manifest.write_text('TRAIN only')
    parent = dict(path='/immutable/IL70k', seal=dict(updates=70000, files_sha256='files'),
                  model_sha256='native_model', identity=dict(qualification=False))
    reg = tmp_path/'registration.json'
    sealed(reg, dict(model_sha256='export_model', protocol_sha256='v1_1',
                     IL_parent_model_sha256='native_model', checkpoint_files_sha256='files'))
    summary = tmp_path/'SUMMARY.json'
    sealed(summary, dict(status='COMPLETE', scenes=12146, invalid_scenes=0, seed=2101,
                         PDMS=88.79, model_sha256='export_model', protocol_sha256='v1_1'))
    selected = dict(checkpoint=parent['path'], updates=70000, model_sha256='native_model',
                    files_sha256='files', identity_sha256=fingerprint(parent['identity']), PDMS=88.79)
    receipt = dict(schema=SCHEMA, user_instruction=INSTRUCTION, Test_observed=True,
                   selection_method='user_requested_best_completed_QwenWorld_Test_as_of_registration',
                   training_seed=42, evaluation_seed=2101, adaptation=MEMORY, selected=selected,
                   training_recipe_sha256=file_sha(config), hard_manifest_sha256=file_sha(manifest),
                   selection_evidence=dict(summary=str(summary), summary_sha256=file_sha(summary),
                                           registration=str(reg), registration_sha256=file_sha(reg)))
    path = tmp_path/'parent.json'; sealed(path, receipt)
    return path, receipt, parent, config, manifest


def test_registered_parent_matches_real_score_lineage(tmp_path):
    path, _, parent, config, manifest = fixture(tmp_path)
    assert validate_selection(path, parent, config, manifest)['selected']['updates'] == 70000


@pytest.mark.parametrize('field,value', [('model_sha256','wrong'), ('updates',40000), ('PDMS',99.)])
def test_resealed_wrong_parent_or_score_is_rejected(tmp_path, field, value):
    path, receipt, parent, config, manifest = fixture(tmp_path)
    receipt['selected'][field] = value
    sealed(path, receipt)
    with pytest.raises(ValueError):
        validate_selection(path, parent, config, manifest)


def test_partial_score_and_changed_manifest_cannot_initialize_rl(tmp_path):
    path, _, parent, config, manifest = fixture(tmp_path)
    manifest.write_text('different TRAIN')
    with pytest.raises(ValueError, match='membership'):
        validate_selection(path, parent, config, manifest)


def test_existing_memory_algorithm_is_unchanged():
    old = expected_scope_recipe(MEMORY); new = expected_recipe()
    assert {k for k in old if old[k] != new[k]} == {'reference_observation'}
    assert new['grpo'] == old['grpo']
    assert new['gradient_clipping_effective'] == 1.


def test_primary_training_lease_coexists_with_inference_but_not_another_trainer(tmp_path, monkeypatch):
    monkeypatch.setattr(job_worker, 'PRIMARY_LEASE_ROOT', tmp_path/'primary')
    monkeypatch.setattr(job_worker, 'SHARED_LEASE_ROOT', tmp_path/'inference')
    devices = {0: dict(index=0,uuid='GPU-test',total_MiB=81920,free_MiB=65000,utilization=0)}
    # Driver/process bookkeeping is positively explained by this known cohost.
    owners = [dict(pid=999,uuid='GPU-test',used_MiB=16920)]
    monkeypatch.setattr(job_worker, '_admission_sample', lambda: (devices, owners))
    monkeypatch.setattr(job_worker, '_known_owners', lambda *a: None)
    args = argparse.Namespace(share_min_free_mib=49152,minimum_free_mib=49152,primary_shared_training=True)
    grant = dict(permanent_hosts=[],unbounded_temporary_hosts=['authorized'])
    primary=[]; duplicate=[]; inference=[]
    try:
        job_worker._gpu_admission(args,'authorized',grant,[0],['GPU-test'],{},primary)
        with pytest.raises(job_worker.ResourceRefused, match='lease'):
            job_worker._gpu_admission(args,'authorized',grant,[0],['GPU-test'],{},duplicate)
        args.primary_shared_training=False
        # Existing inference policy remains fixed-host-only, and has its own lock.
        job_worker._gpu_admission(args,'authorized',dict(permanent_hosts=['authorized']),[0],['GPU-test'],{},inference)
    finally:
        for lock in primary+duplicate+inference: lock.close()
