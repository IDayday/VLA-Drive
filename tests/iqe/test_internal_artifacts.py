from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from iqe.io import atomic_json, read_json, digest
from iqe.internal_baseline.features import JointCache
from iqe.internal_baseline.bank import save_candidate
from iqe.internal_baseline.config import Config
from conftest import scene


def setup(tmp_path):
    cfg=SimpleNamespace(root=tmp_path,identity='configuration')
    atomic_json(tmp_path/'BASE_BINDING.json',{'checkpoint_hash':'base','inference_runtime':{'precision':'original'},
                                           'contract_hash':'contract','bound_at_utc':'fixed'})
    atomic_json(tmp_path/'OFFLINE_BASE_CONTRACT.json',{'trajectory':{'horizon':8,'dt':.5,'raw_representation':'normalized_xy_sincos','frame':'ego_relative'}})
    return cfg


def test_joint_cache_atomic_resume_and_corruption(tmp_path):
    config=setup(tmp_path); cache=JointCache(config,capacity=1); s=scene()
    values={'world':torch.randn(3,8),'action':torch.randn(2,8),'current_ego':torch.randn(4)}
    cache.write(s,values)
    path=cache.path(s); Path(str(path)+'.COMPLETE.json').unlink()
    assert not cache.valid(s)
    cache.write(s,values)
    assert cache.valid(s)
    assert all(torch.equal(v,cache.read(s)[k]) for k,v in values.items())
    with open(path,'ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError,match='corrupt'):
        JointCache(config).read(s)


def test_wrong_observation_runtime_key_refused(tmp_path):
    from dataclasses import replace
    config=setup(tmp_path); cache=JointCache(config); s=scene()
    cache.write(s,{'world':torch.zeros(1,8)})
    with pytest.raises(ValueError,match='stale'):
        cache.valid(replace(s,observation_hash='different'))
    b=read_json(config.root/'BASE_BINDING.json');b['inference_runtime']['precision']='changed'
    atomic_json(config.root/'BASE_BINDING.json',b)
    with pytest.raises(ValueError,match='stale'):
        cache.valid(s)


def test_cache_lru_is_bounded(tmp_path):
    cfg=setup(tmp_path); cache=JointCache(cfg,capacity=1)
    for i in range(3):
        cache.write(scene(i),{'world':torch.full((1,8),i)})
        cache.read(scene(i))
        assert len(cache.loaded)==1


def test_candidate_identity_and_actual_trajectory_no_silent_overwrite(tmp_path):
    cfg=setup(tmp_path); s=scene(); trajectory=np.zeros((8,3),dtype=np.float32)
    a=save_candidate(cfg,s,'expert_1','own-head',trajectory)
    b=save_candidate(cfg,s,'expert_1','own-head',trajectory.copy())
    assert a==b
    with pytest.raises(ValueError,match='output changed'):
        save_candidate(cfg,s,'expert_1','own-head',trajectory+1)
    bad=trajectory.copy();bad[0,0]=np.nan
    with pytest.raises(ValueError,match='invalid physical'):
        save_candidate(cfg,s,'expert_1','other-head',bad)


def test_strict_config_does_not_accept_external_scorer_or_final_test(tmp_path):
    kw=dict(experiment_id='test',output_root=str(tmp_path),contract='c',base_run='b',scene_manifest='s')
    with pytest.raises(TypeError):Config(**kw,external_scorer_checkpoint='external.pt')
    with pytest.raises(TypeError):Config(**kw,final_test='forbidden')
    with pytest.raises(ValueError):Config(**kw,fit_limit=999999)
    assert Config(**kw,fit_limit='all_fit').fit_limit=='all_fit'
