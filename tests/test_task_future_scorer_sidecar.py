import os
from pathlib import Path
import pickle
import numpy as np
import pytest


def test_real_train_cache_readonly_and_scorer_parity(monkeypatch):
    root=Path('/mnt/project/DriveVLA-M0-stage2/cache')
    log='2021.05.12.19.36.12_veh-35_00005_00204'
    token='0a678d2136b35b56'
    metric=root/'metric_cache_navtrain_full'/log/'unknown'/token/'metric_cache.pkl'
    feature=root/'feature_cache_navtrain_full'/log/token/'trajectory_target_planreg_wm_v1.gz'
    if not metric.is_file() or not feature.is_file() or not Path('/mnt/navsim/maps').is_dir():
        pytest.skip('NOT_RUN: real logged metric/input cache/maps unavailable')
    monkeypatch.setenv('NUPLAN_MAPS_ROOT','/mnt/navsim/maps')
    from navsim.planning.training.dataset import load_feature_target_from_pickle
    from navsim.agents.EpisodeDrive.score_module.compute_navsim_score import get_sub_score_from_metric_cache
    from navsim.agents.EpisodeDrive.layers.world_model.physical_label_sidecar import load_metric,score_with_physical_sidecar
    gt=np.asarray(load_feature_target_from_pickle(feature)['trajectory'],np.float32)
    proposals=np.repeat(gt[None],64,0)
    cache,_=load_metric(str(metric))
    before=pickle.dumps(cache)
    reference=get_sub_score_from_metric_cache(cache,proposals,False)
    reference_copy=tuple(x.copy() for x in reference)
    scores,labels=score_with_physical_sidecar(str(metric),proposals,gt,np.arange(7),'us-nv-las-vegas-strip')
    assert pickle.dumps(cache)==before
    for actual,expected,original in zip(scores,reference_copy,reference):
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(original,expected)
    assert labels['valid'].all()
    assert labels['timing']['extra_gt_rollouts']==0
    assert labels['physical_values'].shape==(8,8,3)


def test_all_formal_scorer_core_files_unchanged_from_base():
    import subprocess
    root=Path(__file__).resolve().parents[1]
    for name in ('action_decoder.py','transformer_decoder.py','score_module/scorer.py',
                 'layers/losses/episode_drive_loss.py','score_module/compute_navsim_score.py',
                 'score_module/train_pdm_scorer.py'):
        file='navsim/agents/EpisodeDrive/'+name
        original=subprocess.check_output(['git','show','e85e1a1797f1a26303e9ee81d9f3d1231bc59978:'+file],cwd=root)
        assert (root/file).read_bytes()==original
