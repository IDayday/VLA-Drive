from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
import pytest
from iqe.io import atomic_json,digest,file_hash
from iqe.specialist_study.bank import trajectory_variants
from iqe.specialist_study.rich_variant_publish import construct


def test_parallel_preparation_preserves_source_priority_capacity_and_legacy_bytes(tmp_path,monkeypatch):
    from iqe.internal_baseline import bank
    config=SimpleNamespace(root=tmp_path,max_rich_candidates=12)
    image=tmp_path/'image';image.write_bytes(b'current observation fixture')
    observation={'token':'scene','ego_speed':4.,'image_paths':[str(image)]}
    path=tmp_path/'observation.json';atomic_json(path,observation)
    scene=SimpleNamespace(scene_id='scene',observation_ref=str(path),
          observation_hash=digest({'record':observation,'images':{str(image):file_hash(image)}}))
    base=np.column_stack((np.arange(1,9)*2.,np.zeros(8),np.zeros(8))).astype(np.float32)
    @dataclass
    class Candidate:
        trajectory_hash:str
        source:str
    def save(config,scene,source,identity,trajectory):
        return Candidate(digest(np.asarray(trajectory,dtype=np.float32).tolist()),source)
    monkeypatch.setattr(bank,'save_candidate',save)
    proposal=[('offline_optimized',base.copy())]
    # Exact original serial inner loop used by the already running bank owner.
    seen=set();expected=[];count=9
    for source,t in proposal+trajectory_variants(base,.5,4.):
        if count+len(expected)>=config.max_rich_candidates:break
        c=save(config,scene,'rich_'+source,digest(['asset',source]),t)
        if c.trajectory_hash not in seen:expected.append((source,c));seen.add(c.trajectory_hash)
    actual,_=construct(config,scene,base,proposal,set(),count,'asset',.5)
    assert actual==expected and len(actual)==3 and actual[0][0]=='offline_optimized'
    again,_=construct(config,scene,base,proposal,{actual[0][1].trajectory_hash},count,'asset',.5)
    assert again[0][0]=='time_warp_0.85'
    image.write_bytes(b'changed')
    with pytest.raises(Exception,match='observation changed'):construct(config,scene,base,proposal,set(),count,'asset',.5)
