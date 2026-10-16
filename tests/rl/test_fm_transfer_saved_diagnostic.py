import copy
import json
from types import SimpleNamespace

import pytest

from scripts.analysis.fm_transfer.diagnose_saved_batch import TransferDiagnostic, validate_identity
from starVLA.rl.simwam_recipe.contracts import file_sha, fingerprint, load_recipe


def fixture(variant='A_DDP_TIME'):
    endpoint = 20000 if variant == 'A_DDP_TIME' else 100000
    architecture = 'ddp' if variant == 'A_DDP_TIME' else 'source_action_dit'
    parent = dict(checkpoint='own_sft', model_sha256='own_model', identity_sha256='own_identity',
                  qualification=False, updates=endpoint,
                  config=dict(variant=variant, architecture=architecture, max_updates=endpoint))
    contract = dict(SFT_model_sha256='own_model', SFT_identity_sha256='own_identity',
                    architecture=architecture, seed=2101, grid='native', threshold=.9,
                    statistic='one canonical v1.1 score per scene')
    screen = dict(new_SFT=parent, screen_contract=contract, qualification_only=False)
    screen['sha256'] = fingerprint(screen)
    hard = dict(status='COMPLETE_FIXED_SFT_SINGLE_NATIVE_ODE_HARD_SUBSET',
                screen_identity=screen, screen_contract=contract, rows=[], tokens=[])
    hard['sha256'] = fingerprint(hard)
    cfg = load_recipe('configs/flow_rl/simwam_source_ddp.yaml')
    cfg['supervised_transfer'] = dict(variant=variant, architecture=architecture, SFT_identity=parent)
    identity = dict(schema='fm_transfer_source_RL_v1', purpose='formal', world=8,
                    recipe=cfg, SFT_sha256='own_model', data_stream_identity=dict(screen_contract=contract),
                    source_files={})
    return copy.deepcopy(identity), copy.deepcopy(hard)


@pytest.mark.parametrize('variant', ['A_DDP_TIME', 'B_SOURCE_ACTION_DIT'])
def test_each_route_keeps_its_own_complete_SFT(variant):
    identity, hard = fixture(variant)
    cfg, parent, screen = validate_identity(identity, hard, 1250)
    assert 'supervised_transfer' not in cfg
    assert parent['config']['variant'] == variant
    assert screen['new_SFT'] == parent


@pytest.mark.parametrize('corruption', ['historical_SFT', 'partial_SFT', 'recipe', 'world', 'purpose', 'screen'])
def test_mismatched_reference_and_recipe_are_rejected(corruption):
    identity, hard = fixture()
    if corruption == 'historical_SFT': identity['SFT_sha256'] = 'old_F_SFT'
    elif corruption == 'partial_SFT': identity['recipe']['supervised_transfer']['SFT_identity']['updates'] = 5000
    elif corruption == 'recipe': identity['recipe']['grpo']['train']['clip_range'] = .2
    elif corruption == 'world': identity['world'] = 1
    elif corruption == 'purpose': identity['purpose'] = 'qualification_not_formal_initialization'
    else: hard['screen_contract']['seed'] = 42
    with pytest.raises(ValueError): validate_identity(identity, hard, 1250)


def test_no_unregistered_midpoint_diagnostic():
    identity, hard = fixture()
    with pytest.raises(ValueError, match='registered'): validate_identity(identity, hard, 1000)


def test_screen_payload_and_saved_behavior_are_bound(tmp_path):
    identity, hard = fixture()
    token = 'a_scene'
    path = tmp_path/'screen/scenes'/token[:2]/(token+'.json')
    path.parent.mkdir(parents=True)
    path.with_suffix('.pt').write_bytes(b'saved real-score payload placeholder')
    row = dict(token=token, identity_sha256=hard['screen_identity']['sha256'],
               metrics=dict(score=.8), payload_sha256=file_sha(path.with_suffix('.pt')))
    path.write_text(json.dumps(row))
    hard['rows'] = [dict(token=token, metrics=row['metrics'], score_sha256=file_sha(path))]
    hard['tokens'] = [token]
    hard.pop('sha256'); hard['sha256'] = fingerprint(hard)
    manifest = tmp_path/'hard.json'; manifest.write_text(json.dumps(hard))
    identity['hard_manifest_sha256'] = file_sha(manifest)
    (tmp_path/'screen/manifest.json').write_text(json.dumps(hard['screen_identity']))
    runtime = TransferDiagnostic()
    runtime.prepare(SimpleNamespace(hard_manifest=manifest, sft_screen=tmp_path/'screen',
                    collection=1250, gradient_strata_registration=None), identity)
    assert runtime.baseline_metrics(token) == row['metrics']
    runtime.validate_batch(0, dict(token=token, behavior_version=dict(update=4996)))
    with pytest.raises(ValueError): runtime.validate_batch(0, dict(token=token, behavior_version=dict(update=5000)))
    with pytest.raises(ValueError): runtime.validate_batch(0, dict(token='DEV_scene', behavior_version=dict(update=4996)))
    path.with_suffix('.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='payload changed'): runtime.baseline_metrics(token)
