import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.analysis.fm_transfer.repartition_evaluation import (
    migrate, migrate_scores, repartition_identity, republish, value_digest,
)
from starVLA.rl.simwam_recipe.contracts import file_sha, fingerprint
from starVLA.rl.simwam_recipe.checkpoint import atomic_torch
from starVLA.rl.flow_grpo.transactions import atomic_json


def signed(value):
    return dict(value, sha256=fingerprint(value))


def record(path, identity='old', extra=None):
    values = dict(token='abtoken', log='log', identity_sha256=identity,
        initial_noise=torch.tensor([.1, -.2], dtype=torch.bfloat16),
        normalized_actions=torch.zeros(2, 8, 4), noise_key=2101,
        records={m: dict(status='OK', metrics={'score': .875},
            requested_trajectory=np.arange(24, dtype=np.float32).reshape(8, 3),
            simulated_states=np.array([0., -0., float('nan')])) for m in ['E_NATIVE','E_SHIFT5']})
    values.update(extra or {})
    atomic_torch(path, values)
    seal = dict(token='abtoken', log='log', identity_sha256=identity, payload_sha256=file_sha(path),
                metrics={m:r['metrics'] for m,r in values['records'].items()})
    if extra and 'original_prediction_sha256' in extra:
        seal['original_prediction_sha256'] = extra['original_prediction_sha256']
    atomic_json(path.with_suffix('.json'), seal)
    return values


def test_only_shards_and_digest_change():
    old = signed(dict(shards=2, seed=2101, modes=['E_NATIVE','E_SHIFT5'], weight='same'))
    before = copy.deepcopy(old); new = repartition_identity(old, 12)
    assert old == before and new['sha256'] != old['sha256']
    assert {k:v for k,v in new.items() if k not in ('sha256','shards')} == {
        k:v for k,v in old.items() if k not in ('sha256','shards')}


@pytest.mark.parametrize('shards',[0,2,65])
def test_reject_bad_or_unchanged_shards(shards):
    with pytest.raises(ValueError): repartition_identity({'shards':2,'sha256':'x'}, shards)


def test_republish_preserves_all_values_and_parent(tmp_path):
    old=tmp_path/'old.pt'; new=tmp_path/'new.pt'; value=record(old); parent_sha=file_sha(old)
    result=republish(old,new,'old','new'); copied=torch.load(new,weights_only=False)
    assert copied.pop('identity_sha256')=='new'; value.pop('identity_sha256')
    assert value_digest(copied)==value_digest(value)==result['numerical_values_sha256']
    assert file_sha(old)==parent_sha
    with pytest.raises(FileExistsError): republish(old,new,'old','new')


def test_bad_seal_never_silently_regenerated(tmp_path):
    old=tmp_path/'old.pt'; record(old); old.write_bytes(old.read_bytes()+b'bad')
    with pytest.raises(ValueError): republish(old,tmp_path/'new.pt','old','new')
    assert not (tmp_path/'new.pt').exists()


def test_exact_digest_detects_signed_zero_and_dtype():
    assert value_digest(np.array([0.])) != value_digest(np.array([-0.]))
    assert value_digest(torch.zeros(2)) != value_digest(torch.zeros(2,dtype=torch.bfloat16))


def test_no_migration_before_drain(tmp_path):
    receipt=tmp_path/'drain.json';receipt.write_text(json.dumps({'status':'RUNNING'}))
    with pytest.raises(ValueError): migrate(tmp_path/'old',tmp_path/'new',8,receipt)
    assert not (tmp_path/'new').exists()


def test_score_rebinding_preserves_feedback_and_old_cost(tmp_path):
    old_eval=tmp_path/'old_eval';new_eval=tmp_path/'new_eval';old=tmp_path/'scores';new=tmp_path/'new_scores'
    op=old_eval/'scenes/ab/abtoken.pt';npth=new_eval/'scenes/ab/abtoken.pt'
    record(op,'old_prediction');republish(op,npth,'old_prediction','new_prediction')
    atomic_json(new_eval/'manifest.json',{'sha256':'new_prediction'})
    sid=signed(dict(original_manifest_identity='old_prediction',original_evaluation=str(old_eval)))
    (old/'attempts').mkdir(parents=True)
    atomic_json(old/'manifest.json',sid)
    record(old/'scenes/ab/abtoken.pt',sid['sha256'],dict(
        original_prediction_sha256=file_sha(op),original_manifest_identity='old_prediction'))
    atomic_json(old/'attempts/one.json',dict(status='FAILED_PRESERVED',ended_utc='now',physical_scoring_calls=2,wall_seconds=3))
    result=migrate_scores(old,new,old_eval,new_eval,{'sha256':'old_prediction'},{'sha256':'new_prediction'})
    assert result['reused_scenes']==1 and result['additional_scores']==0
    assert result['inherited_physical_scoring_calls']==2 and result['inherited_invocation_wall_seconds']==3
    payload=torch.load(new/'scenes/ab/abtoken.pt',weights_only=False)
    assert payload['original_prediction_sha256']==file_sha(npth)
