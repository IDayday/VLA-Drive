import pytest
from tools.foresight.score_pdms import requested_population,identity_hash


def test_official_population_is_current_manifest_not_available_score_subset():
    current=[{'token':'a','log':'log1'},{'token':'b','log':'log2'},{'token':'c','log':'log2'}]
    metric=[dict(r,cache_path=r['token']+'.pkl') for r in reversed(current)]
    export={'current_identity':{'index_sha256':identity_hash(current)},'limit':0}
    rows=requested_population(current,metric,export)
    assert [r['token'] for r in rows]==['a','b','c']
    with pytest.raises(ValueError):requested_population(current,metric[:2],export)
    with pytest.raises(ValueError):requested_population(current[::-1],metric,export)
    export['limit']=2
    assert [r['token'] for r in requested_population(current,metric,export)]==['a','b']
    metric[0]['log']='wrong'
    export['limit']=0
    with pytest.raises(ValueError):requested_population(current,metric,export)
