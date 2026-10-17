"""Scope regressions only; no new PROFILE/formal admission is implemented."""
from copy import deepcopy
import json

import pytest

from tools.st_world_policy.check_formal_approval import validate_execution
from tools.st_world_policy.teacher_provider import prepare, configure_smoke
from tools.st_world_policy.training_config import validate_config
from test_training_config import config
from test_fixed_provider_training import fixed_artifacts, prepare_fixed, sign
from test_provider_training import write


def option():
    return {'descriptor':{'path':'/fixture/provider.json','sha256':'a'*64},
            'pca_channel_scope':{'path':'/fixture/scope.json','sha256':'b'*64}}


@pytest.mark.parametrize('scope', ['PROFILE','FORMAL'])
def test_scope_flag_alone_never_admits_provider(scope):
    cfg=config()
    cfg['execution'].update(scope=scope,sample_limit=0 if scope=='FORMAL' else 64)
    cfg['runtime'].update(loader_workers=0,profile_mode='none')
    cfg['data'].update(teacher_root=None,teacher_provider=option())
    with pytest.raises(ValueError,match='SMOKE-only'):
        validate_config(cfg)


def test_profile_general_256_limit_is_distinct_from_complete_240_timing_window():
    cfg=config();cfg['execution']['stop_after']=256
    validate_execution(cfg['execution'])
    with pytest.raises(ValueError,match='two registered 20'):
        validate_config(cfg)
    cfg['runtime']['profile_mode']='none'
    validate_config(cfg)  # Existing immutable-cache path only; no provider.
    cfg['execution']['stop_after']=257
    with pytest.raises(ValueError,match='256'):
        validate_execution(cfg['execution'])


def test_smoke_cannot_smuggle_profile_or_approval_fields():
    cfg=config();cfg['execution'].update(scope='SMOKE',stop_after=2)
    cfg['runtime'].update(loader_workers=0,profile_mode='none')
    cfg['data'].update(teacher_root=None,teacher_provider=dict(option(),profile_approved=True))
    with pytest.raises(ValueError,match='missing/unknown'):
        validate_config(cfg)


@pytest.mark.parametrize('population', [2,65])
def test_new_population_cannot_borrow_a_smaller_real_qualification(tmp_path,population):
    a=fixed_artifacts(tmp_path)
    manifest=json.loads(a.requests.read_text())
    prototype=manifest['scenes'][0]
    manifest['scenes']=[dict(deepcopy(prototype),token='scene-'+str(i)) for i in range(population)]
    write(a.requests,sign(manifest))
    message='outside completed qualification plan' if population==2 else 'bounded to 64'
    with pytest.raises(ValueError,match=message):
        prepare(a.requests,[a.current.root],a.d['teacher_config']['path'],a.report,
                a.fixed_path,limits=a.fixed_limits)
    assert not a.fixed_path.exists()


def test_configure_smoke_cannot_be_used_as_long_profile_writer(tmp_path):
    a=fixed_artifacts(tmp_path);prepare_fixed(a)
    cfg=config();cfg['execution'].update(scope='SMOKE',stop_after=5,sample_limit=1)
    cfg['runtime'].update(profile_mode='none')
    source=write(tmp_path/'long-smoke.json',cfg)
    output=tmp_path/'not-published'
    with pytest.raises(ValueError,match='at most four updates'):
        configure_smoke(source,a.fixed_path,a.fit.root,output)
    assert not output.exists()
