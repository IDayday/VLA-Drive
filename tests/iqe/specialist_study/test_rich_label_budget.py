from collections import Counter
from types import SimpleNamespace as NS

import pytest

from iqe.specialist_study import bank


def test_rich_exact_duplicate_and_full_scene_never_call_official_backend(monkeypatch):
    config=NS(max_rich_candidates=2)
    scenes=[NS(scene_id=str(i)) for i in range(3)]
    candidates=[NS(scene_id='0',trajectory_hash='same'),
                NS(scene_id='1',trajectory_hash='new1'),
                NS(scene_id='2',trajectory_hash='new2')]
    rows=[[('old','label')],[('old','label'),('old2','label2')],[('old','label')]]
    hashes=[{'same'},{'a','b'},{'a'}]
    sources=Counter();calls=[]
    def score(_config,ss,cc,name):
        calls.append(([s.scene_id for s in ss],[c.trajectory_hash for c in cc],name))
        return ['official_for_scene2']
    monkeypatch.setattr(bank,'score_candidates',score)
    counts=bank.append_unique_rich_column(config,scenes,candidates,rows,hashes,sources,'own','fixed_job')
    assert calls==[(['2'],['new2'],'fixed_job')]
    assert counts=={'duplicate_official_requests_avoided':1,'over_capacity_official_requests_avoided':1,'unique_official_requests':1}
    assert sources=={'own':1}
    assert rows[0]==[('old','label')] and len(rows[1])==2
    assert rows[2][-1]==(candidates[2],'official_for_scene2')
    assert hashes[2]=={'a','new2'}


def test_rich_wrong_scene_cannot_be_scored(monkeypatch):
    def fail(*_):raise AssertionError('must not invoke official backend')
    monkeypatch.setattr(bank,'score_candidates',fail)
    with pytest.raises(Exception,match='reordered'):
        bank.append_unique_rich_column(NS(max_rich_candidates=12),[NS(scene_id='a')],
            [NS(scene_id='b',trajectory_hash='x')],[[]],[set()],Counter(),'own','invalid')


def test_rich_deployment_aliases_deduplicate_without_hiding_bad_labels():
    a,b,c=[NS(trajectory_hash=h) for h in ('x','x','y')]
    label=NS(total_score_01=.8,named_components={'NC':1.},component_valid_masks={'NC':True})
    rows,removed=bank.deduplicate_rich_rows([[(a,label),(b,label),(c,label)]])
    assert rows==[[(a,label),(c,label)]] and removed==1
    bad=NS(total_score_01=.8,named_components={'NC':0.},component_valid_masks={'NC':True})
    with pytest.raises(Exception,match='inconsistent official labels'):
        bank.deduplicate_rich_rows([[(a,label),(b,bad)]])
