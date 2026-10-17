"""Small metadata-only diagnostic fixtures; not real teacher/cache evidence."""
import copy
from types import SimpleNamespace

import pytest
from starVLA.dataloader.st_world_schema import TeacherFeatureContract
from starVLA.dataloader.st_world_targets import payload_key
from tools.st_world_policy import diagnose_future_current_reuse as d


def fixture():
    contract=TeacherFeatureContract(image_hw=(32,32),patch_hw=(16,16),native_grid_hw=(2,2),
        output_grid_hw=(2,2),channels=2,storage_dtype='float32',compute_dtype='float32',
        view_names=('CAM_F0','CAM_L0','CAM_R0'),teacher_identity='a'*64,transform_identity='b'*64,
        pooling=dict(kind='none',kernel_hw=[1,1],stride_hw=[1,1]))
    a,b=dict(sha256='1'*64),dict(sha256='2'*64)
    manifest=dict(split='train',source_student_identity='c'*64,source_index_sha256='d'*64,
        partition_sha256='e'*64,view_names=list(contract.view_names),scenes=[dict(frames=[
            dict(requested_s=t,views=[a,b,None]) for t in d.TIMES])])
    calls=[]
    def row(table,column,key):
        calls.append(key)
        if key!=payload_key(a,contract):raise KeyError(key)
        return dict(key=key,image=a)
    return manifest,SimpleNamespace(manifest=manifest,row=row),contract,calls


def test_exact_key_contract_counts_missing_repeats_and_queries_once():
    m,i,c,calls=fixture();checks=[]
    result=d.analyze(m,i,c,lambda *a:checks.append(a))
    assert result['total']==dict(slots=15,present=10,missing=5,current_cache_hit=5,current_cache_miss=5)
    assert result['distinct_present_images']==2 and result['distinct_payload_keys']==2
    assert result['repeated_present_slots']==8 and result['distinct_hit_keys']==1
    assert len(calls)==2 and result['sqlite_queries']==2 and checks
    assert all(r['current_cache_hit']==1 for r in result['by_time'].values())


def test_wrong_lineage_fails_before_any_query():
    m,i,c,calls=fixture();m=copy.deepcopy(m);m['partition_sha256']='f'*64
    with pytest.raises(ValueError,match='source contract'):d.analyze(m,i,c,lambda *_:None)
    assert not calls


def test_malformed_checksumming_record_cannot_be_a_hit():
    m,i,c,calls=fixture()
    i.row=lambda *a:dict(key=a[-1],image=dict(sha256='9'*64))
    with pytest.raises(ValueError,match='corruption'):d.analyze(m,i,c,lambda *_:None)


@pytest.mark.parametrize('field,value', [('logical_read_bytes',1<<30),('peak_rss_bytes',(2<<30)+1),('elapsed_s',181)])
def test_each_runtime_budget_blocks(field,value,monkeypatch):
    budget=d.Budget();stats=dict(elapsed_s=0,logical_read_bytes=0,peak_rss_bytes=1)
    stats[field]=value;monkeypatch.setattr(budget,'stats',lambda:stats)
    with pytest.raises(RuntimeError,match='BLOCKED_RESOURCE'):budget.check(1)
