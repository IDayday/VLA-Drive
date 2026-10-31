"""CPU checks of useful-signal selection, real source order and native recovery."""
import copy
import io
import json
import pickle
import random

import pytest
import torch

from starVLA.rl.qwen_world_research_data import manifest as mm
from starVLA.rl.qwen_world_research_data.data import SceneStream
from starVLA.rl.qwen_world_safety_exploration.curriculum import (
    DEFAULT_CONFIG, OnlinePriorityStream, CurriculumRuntimeMixin, signal_statistics, validate_config)


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    tokens=[f't{i:04}' for i in range(128)]
    logs={t:f'2026.01.01.00.00.00_veh-{i//8}_00000_00100' for i,t in enumerate(tokens)}
    logs['NAVTEST']='2026.02.01.00.00.00_veh-999_00000_00100'
    population=tmp_path/'population.json'
    population.write_text(json.dumps(dict(train_tokens=tokens[:120],dev_tokens=tokens[120:],
        navtest_tokens=['NAVTEST'],token_logs=logs)))
    monkeypatch.setattr(mm,'POPULATION',population)
    data=tmp_path/'data';data.mkdir()
    (data/'manifest.json').write_text(json.dumps(dict(tokens=tokens,count=len(tokens),old_DEV_tokens=tokens[120:],
        source_population_path=str(population),population_sha256=mm.file_sha(population))))
    path=tmp_path/'full.json'
    mm.write_stream_manifest(path,data_root=data,tokens=tokens,name='full fixture',mode='uniform')
    return path


def state():
    return dict(updates=0,collections=0,scene_cursor=0,next_inner=0)


def rollout(token, position, group=8, kind='mixed'):
    if kind=='mixed':
        c=[[1.,1.,1.,.8]]*(group//2)+[[1.,1.,0.,.9]]*(group//2)
        raw=[.5]*(group//2)+[-.5]*(group//2)
    elif kind=='progress':
        c=[[1.,1.,1.,e] for e in [.2,.8]*(group//2)]
        raw=[-.3,.3]*(group//2)
    elif kind=='saturated':
        c=[[1.,1.,1.,1.]]*group;raw=[0.]*group
    else:
        c=[[1.,0.,1.,0.]]*group;raw=[0.]*group
    return dict(token=token,stream_position=copy.deepcopy(position),
        structured_advantage=dict(components=torch.tensor([c],dtype=torch.float64),
                                  A_raw=torch.tensor([raw],dtype=torch.float64)),
        # Large canonical/Comfort variance must never substitute for Safety rawA.
        rewards=torch.tensor([[1.,5/6]*(group//2)]))


def collect(stream, native, rank=0, group=8, kind='mixed'):
    token, position=stream.at(native['scene_cursor'],rank)
    r=rollout(token,position,group,kind)
    native['collections']+=1;native['scene_cursor']+=stream.world
    stream.commit(r,native)
    return r


def finish(native):
    native['updates']+=4;native['next_inner']=0


def test_signal_is_Safety_credit_not_Comfort_variance():
    for g in [8,16]:
        for kind in ['saturated','allfail']:
            s=signal_statistics(rollout('t',{},g,kind),DEFAULT_CONFIG)
            assert not s['eligible'] and s['rawA_variance']==0
        for kind in ['mixed','progress']:
            assert signal_statistics(rollout('t',{},g,kind),DEFAULT_CONFIG)['eligible']
    with pytest.raises(ValueError):validate_config(dict(DEFAULT_CONFIG,unknown=True))
    with pytest.raises(ValueError):validate_config(dict(DEFAULT_CONFIG,warmup_collections=100))


def test_real_uniform_residues_preserved_priority_activates_before1k(manifest):
    ranks=[]
    for rank in range(2):
        base=SceneStream(manifest,2)
        s=state();stream=OnlinePriorityStream(base,DEFAULT_CONFIG).bind(s)
        uniform=[];priority=[]
        for _ in range(36):
            before=s['curriculum']['uniform_draws']
            r=collect(stream,s,rank)
            if r['stream_position']['selection_source']=='uniform':
                assert r['token']==base.at(before*2,rank)[0]
                uniform.append(r['token'])
            else:
                assert s['curriculum']['uniform_draws']==before
                priority.append(s['collections'])
            finish(s)
        assert priority[0]==12 and all(c%4==0 for c in priority)
        assert s['curriculum']['stats']['uniform_exposures']>=.75*36
        assert len(uniform)==len(set(uniform))
        ranks.append(set(uniform))
    assert not ranks[0]&ranks[1]


def test_native_pending_selection_and_every_inner_roundtrip(manifest):
    native=state();stream=OnlinePriorityStream(SceneStream(manifest,2),DEFAULT_CONFIG).bind(native)
    for _ in range(11):collect(stream,native);finish(native)
    # Save a prepared priority plan before collect: repeating at must not redraw.
    choice=stream.at(native['scene_cursor'],0);before=copy.deepcopy(native)
    restored=OnlinePriorityStream(SceneStream(manifest,2),DEFAULT_CONFIG).bind(copy.deepcopy(before))
    assert restored.at(native['scene_cursor'],0)==choice
    assert restored.current['rng_state']==stream.current['rng_state']
    r=collect(stream,native)
    assert r['stream_position']['selection_source']=='priority'
    prepared=copy.deepcopy(native)
    for cut in range(5):
        saved=copy.deepcopy(prepared);saved['updates']+=cut;saved['next_inner']=cut if cut<4 else 0
        pending=r if cut<4 else None
        payload=io.BytesIO();torch.save(dict(state=saved,rollout=pending),payload);payload.seek(0)
        loaded=torch.load(payload,map_location='cpu',weights_only=False)
        st=loaded['state'];clone=OnlinePriorityStream(SceneStream(manifest,2),DEFAULT_CONFIG).bind(st)
        clone.validate(st,loaded['rollout'])
        unchanged=pickle.dumps(st)
        clone.commit(r,st)  # exact duplicate callback cannot re-admit/increment RNG.
        assert pickle.dumps(st)==unchanged
        assert clone.at(st['scene_cursor'],0)==stream.at(native['scene_cursor'],0)
        assert clone.current['rng_state']==stream.current['rng_state']


def test_global_rng_untouched_lifetime_cap_survives_eviction_and_recovery(manifest):
    cfg=dict(DEFAULT_CONFIG,capacity=2,cooldown_collections=1,max_priority_repeats=2,max_age_collections=16)
    native=state();stream=OnlinePriorityStream(SceneStream(manifest,2),cfg).bind(native)
    py_rng=random.getstate();torch_rng=torch.get_rng_state().clone()
    for i in range(200):
        collect(stream,native);finish(native)
        if i==60:
            native=pickle.loads(pickle.dumps(native))
            stream=OnlinePriorityStream(SceneStream(manifest,2),cfg).bind(native)
    assert random.getstate()==py_rng and torch.equal(torch.get_rng_state(),torch_rng)
    assert max(stream.current['priority_counts'].values())<=2
    assert stream.current['stats']['exits_capacity']>0
    assert len(stream.current['bank'])<=2
    assert stream.current['uniform_draws']+stream.current['stats']['priority_exposures']==200


def test_saturation_exits_pool_and_foreign_or_wrong_rank_resume_rejected(manifest):
    s=state();stream=OnlinePriorityStream(SceneStream(manifest,2),DEFAULT_CONFIG).bind(s)
    for _ in range(11):collect(stream,s);finish(s)
    token,_=stream.at(s['scene_cursor'],0)
    collect(stream,s,kind='saturated');finish(s)
    assert token not in stream.current['bank'] and stream.current['stats']['exits_saturated']==1
    with pytest.raises(ValueError,match='another rank'):stream.at(s['scene_cursor'],1)
    broken=copy.deepcopy(s);broken['curriculum']['priority_counts']['NAVTEST']=1
    with pytest.raises(ValueError):OnlinePriorityStream(SceneStream(manifest,2),DEFAULT_CONFIG).bind(broken)
    broken=copy.deepcopy(s);broken['curriculum']['selection_plan']['token']='NAVTEST'
    with pytest.raises(ValueError,match='selection plan'):OnlinePriorityStream(SceneStream(manifest,2),DEFAULT_CONFIG).bind(broken)
    with pytest.raises(ValueError,match='lacks complete'):OnlinePriorityStream(SceneStream(manifest,2),DEFAULT_CONFIG).bind(dict(collections=1,scene_cursor=2))


def test_runtime_uniform_control_is_original_stream(manifest):
    class Base:
        def make_scene_stream(self,cfg,args,world):
            self.stream=SceneStream(manifest,world);return self.stream
        def validate_boundary(self,state,rollout,identity=None):
            return dict(collections=state['collections'])
        def validate_ranks(self,summaries):return summaries
        def public_training_state(self,state):return dict(state)
    class Runtime(CurriculumRuntimeMixin,Base):pass
    runtime=Runtime();actual=runtime.make_scene_stream(dict(research=dict(data_policy='uniform_full')),None,2)
    assert type(actual) is SceneStream and runtime.stream is actual
    actual=runtime.make_scene_stream(dict(research=dict(data_policy='online_priority',curriculum=DEFAULT_CONFIG)),None,2)
    assert type(actual) is OnlinePriorityStream and runtime.stream is actual
    native=state();actual.bind(native)
    summary=runtime.validate_boundary(native,None,identity={})
    assert runtime.validate_ranks([summary,summary])
    collect(actual,native)
    summary=runtime.validate_boundary(native,None,identity={})
    with pytest.raises(ValueError,match='Cross-rank'):runtime.validate_ranks([summary,summary])
    json.dumps(runtime.public_training_state(native))
