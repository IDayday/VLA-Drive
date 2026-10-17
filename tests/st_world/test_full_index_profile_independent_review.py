"""Independent synthetic admission checks; no real-data/profile qualification."""
from copy import deepcopy

import pytest

from test_provider_full_index_profile import full
from test_provider_profile_upgrade import f
from tools.st_world_policy import teacher_provider as tp
from tools.ddpolicy_vehicle.training_state import epoch_batches


@pytest.mark.parametrize('size,world,steps', [(258,2,64),(258,2,256),(257,1,256),(101592,2,64)])
def test_declared_exposures_equal_original_sampler_across_pause(size,world,steps,full):
    cfg=deepcopy(full.recipe)
    cfg['execution'].update(sample_limit=size,world_size=world,stop_after=steps)
    budget=tp.full_profile_consumption(cfg)
    rank_ids=[[] for _ in range(world)]
    completed=epoch=offset=0
    split=min(steps,31)
    # Resume consumes the same original epoch/offset, not prefetched samples.
    for endpoint in (split,steps):
        while completed<endpoint:
            batches=epoch_batches(size,2,42,epoch)
            if offset==len(batches):epoch+=1;offset=0;continue
            for rank in range(world):rank_ids[rank].extend((epoch,i) for i in batches[offset][rank::world])
            completed+=1;offset+=1
    assert [len(x) for x in rank_ids]==budget['per_rank_scene_exposures']
    assert sum(map(len,rank_ids))==budget['cumulative_scene_exposures']
    assert completed==budget['cumulative_optimizer_updates']


def test_two_rank_odd_population_rejects_empty_rank_tail(full):
    cfg=deepcopy(full.recipe);cfg['execution']['sample_limit']=257
    with pytest.raises(ValueError,match='Empty rank tail'):tp.full_profile_consumption(cfg)


@pytest.mark.parametrize('field,value', [('stop_after',0),('stop_after',257),('sample_limit',256),
    ('world_size',4),('micro_batch',2),('global_batch',32)])
def test_new_topology_and_cumulative_ceiling_reject_invalid(full,field,value):
    cfg=deepcopy(full.recipe);cfg['execution'][field]=value
    with pytest.raises(ValueError):tp.full_profile_consumption(cfg)


def test_new_profile_rejects_continuation_identity(full):
    cfg=deepcopy(full.recipe);cfg['continuation']={}
    with pytest.raises(ValueError):tp.full_profile_consumption(cfg)


@pytest.mark.parametrize('kind', ['future_source','ledger_checksum','selection_identity','current_frame','timestamp',
    'current_partition','future_times','log_count','ordered_tokens','ego_identity','current_request_count'])
def test_complete_population_rejects_independent_inconsistencies(full,monkeypatch,kind):
    # Real tiny SQLite indices remain in the path; mutate only a decoded view.
    # This is fault injection, not a substitute for a real full-population run.
    from starVLA.dataloader.st_world_metadata_index import Rows
    original_open=tp.open_metadata_index
    def opened(binding,**kwargs):
        index=original_open(binding,**kwargs)
        future=index.root==full.index
        if future and kind=='future_source':index.receipt['manifest_header']['source']='not_original'
        if future and kind=='ledger_checksum':index.receipt['dependencies']['source_logs_ledger']['sha256']='0'*64
        if future and kind=='selection_identity':index.receipt['manifest_header']['selection_identity']='0'*64
        if not future and kind=='current_request_count':index.receipt['scene_count']-=1
        return index
    monkeypatch.setattr(tp,'open_metadata_index',opened)
    original_iter=Rows.__iter__
    def rows(self):
        for pos,row in enumerate(original_iter(self)):
            row=deepcopy(row)
            if self.index.root==full.index and pos==0:
                if kind=='current_frame':row['frames'][0]['views'][0]['sha256']='0'*64
                if kind=='timestamp':row['timestamp_us']+=1
            yield row
    monkeypatch.setattr(Rows,'__iter__',rows)
    original_pinned=tp.pinned
    def pinned(record):
        result=original_pinned(record)
        population=full.admission['population']
        if record==population['current_registration']:
            if kind=='current_partition':result['source']['partition_sha256']='0'*64
            if kind=='log_count':result['source']['logs']+=1
            if kind=='ordered_tokens':result['source']['ordered_scene_ids_hash']='0'*64
            if kind=='ego_identity':result['source']['ego_identity_sha256']='0'*64
            result['identity']=tp.identity_hash({k:v for k,v in result.items() if k!='identity'})
        if record==population['future_complete'] and kind=='future_times':result['target_times_s']=[0.,.5,1.,2.,3.]
        return result
    monkeypatch.setattr(tp,'pinned',pinned)
    with pytest.raises((ValueError,KeyError)):tp.validate_descriptor(full.d)
