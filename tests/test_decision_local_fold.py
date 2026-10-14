from copy import deepcopy
from types import SimpleNamespace
import pytest
from tools.decision_local.train_small_model import apply_log_fold


def test_readout_folds_are_exhaustive_log_disjoint_and_do_not_mutate_manifest():
    rows=[{'token':str(i),'log':str(i//2)} for i in range(8)]
    feature={'identity':'fixed','index':deepcopy(rows)}
    def data():return SimpleNamespace(index=rows.copy(),feature_identity=feature,label_identity={'identity':'labels'})
    first,second=data(),data()
    partition={'features':'fixed','log_fold':{'0':0,'1':1,'2':0,'3':1}}
    receipt=apply_log_fold(first,second,partition,0)
    assert {r['log'] for r in first.index}.isdisjoint({r['log'] for r in second.index})
    assert sorted(first.index+second.index,key=lambda x:x['token'])==rows
    assert feature['index']==rows and len(rows)==8
    a,b=data(),data();apply_log_fold(a,b,partition,1)
    assert a.index==second.index and b.index==first.index
    assert receipt['training_scenes']==receipt['heldout_scenes']==4
    for bad in [dict(partition,features='other'),dict(partition,log_fold={'0':0,'1':1})]:
        with pytest.raises(ValueError):apply_log_fold(data(),data(),bad,0)
