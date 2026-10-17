"""Independent bounded full-profile resource fixtures; no actual population or GPU."""
import ast
from copy import deepcopy
import gc
import json
from pathlib import Path
import weakref

import pytest
from tools.st_world_policy import teacher_provider as tp
from starVLA.dataloader.st_world_metadata_index import Rows
from test_provider_full_index_profile import full
from test_provider_profile_upgrade import f

OUT=Path(__file__).parent


def test_lazy_rows_are_not_population_retained(full,monkeypatch):
    counts={'live':0,'peak':0,'total':0}
    original_iter=Rows.__iter__;original_text=Path.read_text;original_bytes=Path.read_bytes
    forbidden={full.requestpath,full.selectedpath,full.current.root/'requests.json',full.current.root/'identity.json'}
    class Row(dict):pass
    def release():counts['live']-=1
    def tracked(self):
        for source in original_iter(self):
            row=Row(source);counts['live']+=1;counts['total']+=1;counts['peak']=max(counts['peak'],counts['live'])
            weakref.finalize(row,release)
            yield row
    def guarded_text(path,*args,**kwargs):
        assert path not in forbidden,'Unexpected full population Path.read_text'
        return original_text(path,*args,**kwargs)
    def guarded_bytes(path,*args,**kwargs):
        assert path not in forbidden,'Unexpected full population Path.read_bytes'
        return original_bytes(path,*args,**kwargs)
    monkeypatch.setattr(Rows,'__iter__',tracked)
    monkeypatch.setattr(Rows,'__getitem__',lambda *args:pytest.fail('No whole-population ordinal/slice'))
    monkeypatch.setattr(Path,'read_text',guarded_text);monkeypatch.setattr(Path,'read_bytes',guarded_bytes)
    d,_,provider=tp.load_descriptor(tp.pin(full.path))
    try:
        report=provider.memory_report()
        assert report['metadata_population_materialized'] is False
        assert counts['total']>=2*258
        assert counts['peak']<=8
        assert report['metadata_connections']==2
        assert report['metadata_cache_budget_bytes']==32*(1<<20)
        assert provider.max_cached_bytes==full.p['limits']['max_cached_bytes']
        assert provider.max_scene_bytes==full.p['limits']['max_scene_bytes']
        assert provider.max_batch_images==8
    finally:tp.close_provider_metadata(provider)
    gc.collect()
    (OUT/'STREAM_RETENTION_RESULT.json').write_text(json.dumps({'counts':counts,'provider_report':report,'scope':'synthetic258; no real full population'},sort_keys=True,indent=2)+'\n')


@pytest.mark.parametrize('n,world,steps',[(257,1,256),(258,1,256),(258,2,256),(101592,1,64),(101592,2,64)])
def test_cumulative_exposure_and_rank_consumption_across_resume(n,world,steps):
    cfg={'execution':dict(world_size=world,micro_batch=1,global_batch=2,sample_limit=n,stop_after=steps),'runtime':{'profile_mode':'none'}}
    declared=tp.full_profile_consumption(cfg)
    # Stream synthetic batch sizes only; no source manifest or full sampler list.
    per_epoch=(n+1)//2
    batches=[min(2,n-2*(step%per_epoch)) for step in range(steps)]
    for pause in (0,1,steps//2,steps):
        before=batches[:pause];after=batches[pause:]
        assert sum(before)+sum(after)==declared['cumulative_scene_exposures']
        assert len(before)+len(after)==declared['cumulative_optimizer_updates']
        ranks=[sum(len(range(rank,size,world)) for size in before+after) for rank in range(world)]
        assert ranks==declared['per_rank_scene_exposures']
    assert declared['resume_budget']=='same_run_cumulative_not_per_invocation'
    assert declared['gradient_accumulation_steps']==2//world


@pytest.mark.parametrize('change',['stop257','odd_two_rank','continuation','profile_timing','world4'])
def test_unsupported_exposure_or_topology_rejected(change):
    cfg={'execution':dict(world_size=2,micro_batch=1,global_batch=2,sample_limit=258,stop_after=64),'runtime':{'profile_mode':'none'}}
    if change=='stop257':cfg['execution']['stop_after']=257
    elif change=='odd_two_rank':cfg['execution']['sample_limit']=257
    elif change=='continuation':cfg['continuation']={}
    elif change=='profile_timing':cfg['runtime']['profile_mode']='timing'
    else:cfg['execution']['world_size']=4
    with pytest.raises(ValueError):tp.full_profile_consumption(cfg)


def test_world_change_has_new_recipe_and_consumption_identity(full):
    cfg=deepcopy(full.recipe);world2=tp.profile_recipe_identity(cfg);cfg['execution']['world_size']=1
    assert tp.profile_recipe_identity(cfg)!=world2
    assert tp.full_profile_consumption(cfg)!=full.admission['consumption']
    with pytest.raises(ValueError):tp.validate_profile_recipe(full.admission,cfg)


def test_native_resume_and_stop_gate_unchanged_and_before_fetch():
    root=Path(tp.__file__).resolve().parents[2]
    source=(root/'tools/st_world_policy/train_world.py').read_text()
    assert 'world_size=world' in source
    resume=source.index('if args.resume or "continuation" in cfg:')
    restore=source.index('state = dict(receipt["state"])',resume)
    gate=source.index('or state["completed"] >= execution["stop_after"]',restore)
    fetch=source.index('ids = batches[state["offset"]][rank::world]',gate)
    assert restore<gate<fetch
    assert 'state["completed"] += 1' in source and 'state["exposure"] += counts["ego_scenes"]' in source
