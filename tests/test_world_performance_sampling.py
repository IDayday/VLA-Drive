import copy,json,random
from collections import Counter
from pathlib import Path
import pytest
import torch
from starVLA.rl.qwen_world_performance import sampling_priority as S,data_protocol as P

REAL='/mnt/project/qwen-world-performance-rl-20261103/data/NAVTRAIN_SCENES.json'


def window(s,state):
    start=s._state['cursor'];positions=[];tokens=[]
    for micro in range(s.batch_size//s.world):
        for rank in range(s.world):
            t,m=s.at(start+micro*s.world,rank);tokens.append(t);positions.append(m)
    next_cursor=s.commit_window(positions,start+s.batch_size);state['scene_cursor']=next_cursor
    return tokens,positions


def test_real_complete_first_epoch_padding_and_batch_boundary():
    s=S.PerformanceSceneStream(REAL,8,batch_size=128,run_identity='cpu-contract',priority=True)
    state={'scene_cursor':0};s.bind(state);seen=[];padding=0;before=torch.get_rng_state().clone();py=random.getstate()
    # One actual full registered population, no images/scoring/model reads.
    while s._state['epoch']==0:
        p=s._plan();seen.extend(x['token']for x in p['items']if x['valid']);padding+=sum(not x['valid']for x in p['items'])
        if len(seen)==128:
            with pytest.raises(ValueError,match='epoch boundary'):s.set_batch_size(64)
        nxt=p['start_cursor']+s.batch_size;s.commit_window(None,nxt);state['scene_cursor']=nxt
    g=torch.Generator().manual_seed(42);expected=[s.tokens[i]for i in torch.randperm(103288,generator=g).tolist()]
    assert seen==expected and len(set(seen))==103288 and padding==8
    assert s.summary()['valid_scene_exposures']==103288 and s.summary()['uniform_cursor']==103288
    s.set_batch_size(64)
    assert s.epoch_size==103296 and torch.equal(before,torch.get_rng_state())and random.getstate()==py


def test_priority_native_plan_restore_caps_stale_feedback_and_test_leak_rejection(tmp_path,monkeypatch):
    tokens=[f't{i:03}'for i in range(200)];logs={t:f'physical-{i%4}_00000_00100'for i,t in enumerate(tokens)}
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    monkeypatch.setattr(S,'load_manifest',lambda p,n:({'tokens':tokens},logs))
    def make():return S.PerformanceSceneStream(manifest,2,batch_size=8,run_identity='own')
    s=make();state={'scene_cursor':0};s.bind(state)
    while s._state['epoch']==0:window(s,state)
    def row(t,ep):
        metrics=dict(no_at_fault_collisions=1,drivable_area_compliance=1,time_to_collision_within_bound=1,ego_progress=ep,comfort=1)
        return dict(log=logs[t],metrics=metrics,score=(5*ep+7)/12)
    records={t:row(t,.4 if i<30 else.95)for i,t in enumerate(tokens[:60])}
    prior=tmp_path/'prior.json';P.write_feedback(prior,run_identity='own',checkpoint_identity='prior-own-cp',observed_valid_scene_clock=0,records={t:row(t,.95)for t in tokens[:60]},logs=logs)
    bank=tmp_path/'b1.json';P.write_feedback(bank,run_identity='own',checkpoint_identity='sealed-cp',observed_valid_scene_clock=200,records=records,logs=logs,previous=prior)
    s.offer_feedback(bank);s.at(state['scene_cursor'],0)
    restored=make();saved=copy.deepcopy(state);restored.bind(saved)
    summary=S.stream_state_validation(saved,run_identity='own',stream_identity=restored.identity,tokens=tokens)
    assert summary['pending_selection_plan_sha256'] is not None
    bad=copy.deepcopy(saved);bad['scene_stream']['valid_scene_exposures']+=1
    with pytest.raises(ValueError,match='valid/padding'):
        S.stream_state_validation(bad,run_identity='own',stream_identity=restored.identity,tokens=tokens)
    bad=copy.deepcopy(saved);p=bad['scene_stream']['selection_plan'];p['items'][0]['valid_mask']=0;p.pop('sha256');p['sha256']=P.fingerprint(p)
    with pytest.raises(ValueError,match='padding contributes'):
        S.stream_state_validation(bad,run_identity='own',stream_identity=restored.identity,tokens=tokens)
    assert restored.at(saved['scene_cursor'],1)==s.at(state['scene_cursor'],1)
    a,am=window(s,state);b,bm=window(restored,saved);assert(a,am,state)==(b,bm,saved)
    counts=Counter(t for t,m in zip(a,am)if m['selection_kind']in('nominal_hard','regression'))
    while s._state['epoch']==1:
        ts,ms=window(s,state)
        assert len(set(t for t,m in zip(ts,ms)if m['valid']))==sum(m['valid']for m in ms)
        counts.update(t for t,m in zip(ts,ms)if m['selection_kind']in('nominal_hard','regression'))
    assert counts and max(counts.values())<=4
    uniform_before=s._state['uniform_cursor'];priority_before=sum(s._state['counters'][k]for k in('nominal_hard','regression'))
    # Feedback becomes older than1 full valid epoch; priority cannot reuse it.
    for _ in range(3):
        ts,ms=window(s,state)
        assert all(m['selection_kind']=='uniform'for m in ms)
    assert sum(s._state['counters'][k]for k in('nominal_hard','regression'))==priority_before
    assert s._state['uniform_cursor']==uniform_before+24
    with pytest.raises(ValueError,match='No planned'):s.commit_window(None,state['scene_cursor'])
    wrong=json.loads(bank.read_text());wrong['records']['NAVTEST-token']=wrong['records'].pop(tokens[0]);wrong.pop('sha256');P.atomic_json(tmp_path/'wrong.json',P.sealed(wrong))
    with pytest.raises(ValueError,match='Forbidden'):s.offer_feedback(tmp_path/'wrong.json')
    wrong=json.loads(bank.read_text());wrong['run_identity']='other-policy';wrong.pop('sha256');P.atomic_json(tmp_path/'other.json',P.sealed(wrong))
    with pytest.raises(ValueError,match='same-run'):s.offer_feedback(tmp_path/'other.json')
