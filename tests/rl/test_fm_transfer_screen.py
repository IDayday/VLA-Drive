import json
from pathlib import Path
import pytest
from scripts.analysis.fm_transfer.build_hard_manifest import build
from starVLA.rl.simwam_recipe.contracts import fingerprint,file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def fixture_screen(tmp):
    tokens=['aa','bb','cc'];pop=dict(train_tokens=tokens,dev_tokens=['dd'],navtest_tokens=['ee'],
        identity_sha256='population',token_logs={t:'log1' for t in tokens})
    parent=dict(model_sha256='weight',identity_sha256='identity',config=dict(architecture='ddp',max_updates=20000),
        qualification=False,updates=20000)
    contract=dict(SFT_model_sha256='weight',SFT_identity_sha256='identity',architecture='ddp',seed=2101,
        grid='native',threshold=.9,statistic='one canonical v1.1 score per scene')
    identity=dict(schema='fm_transfer_single_native_TRAIN_screen_v1',new_SFT=parent,screen_contract=contract,
        qualification_only=False,population_identity='population',tokens_sha256=fingerprint(tokens),scene_count=3,shards=1)
    identity['sha256']=fingerprint(identity)
    atomic_json(tmp/'manifest.json',identity)
    atomic_json(tmp/'rank0.json',dict(status='ASSIGNED_COMPLETE',identity_sha256=identity['sha256']))
    for t,score,nc in zip(tokens,[1.,.89,0.],[1.,1.,0.]):
        path=tmp/'scenes'/t/(t+'.pt');path.parent.mkdir(parents=True);path.write_bytes(b'sealed-test-payload')
        atomic_json(path.with_suffix('.json'),dict(token=t,log='log1',identity_sha256=identity['sha256'],
            payload_sha256=file_sha(path),metrics=dict(score=score,no_at_fault_collisions=nc,
                drivable_area_compliance=1.,time_to_collision_within_bound=1.)))
    return pop,identity


def test_complete_fixed_single_score_and_idempotent(tmp_path):
    pop,_=fixture_screen(tmp_path);out=tmp_path/'hard.json'
    actual=build(tmp_path,out,workers=1,population=pop)
    assert actual['tokens']==['bb','cc'] and actual['screened_scene_count']==3
    assert actual['hard_types']=={'safe_lower_quality_or_progress':1,'no_at_fault_collisions':1}
    assert build(tmp_path,out,workers=2,population=pop)==actual


@pytest.mark.parametrize('failure',['qualification','not_endpoint','wrong_seed','wrong_population','missing_rank',
    'missing_scene','tampered_payload','nonfinite','frozen_membership'])
def test_no_partial_or_mixed_population_publication(tmp_path,failure):
    pop,identity=fixture_screen(tmp_path);out=tmp_path/'hard.json'
    if failure=='qualification':identity['qualification_only']=True
    elif failure=='not_endpoint':identity['new_SFT']['updates']=10000
    elif failure=='wrong_seed':identity['screen_contract']['seed']=42
    elif failure=='wrong_population':pop['dev_tokens'].append('aa')
    elif failure=='missing_rank':(tmp_path/'rank0.json').unlink()
    elif failure=='missing_scene':(tmp_path/'scenes/aa/aa.json').unlink()
    elif failure=='tampered_payload':(tmp_path/'scenes/aa/aa.pt').write_bytes(b'changed')
    elif failure=='nonfinite':
        # Inject a foreign/corrupt file; the normal atomic writer itself refuses NaN.
        path=tmp_path/'scenes/aa/aa.json';row=json.loads(path.read_text());row['metrics']['score']=float('nan');path.write_text(json.dumps(row))
    elif failure=='frozen_membership':atomic_json(out,dict(tokens=['aa']))
    if failure in ('qualification','not_endpoint','wrong_seed'):
        identity.pop('sha256');identity['sha256']=fingerprint(identity);atomic_json(tmp_path/'manifest.json',identity)
    with pytest.raises((ValueError,FileNotFoundError)):build(tmp_path,out,workers=1,population=pop)
    if failure!='frozen_membership':assert not out.exists()
