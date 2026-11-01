"""Two checks: independent inversion science+guards and reused native8 path."""
import copy

import pytest
import torch
import torch.multiprocessing as mp

from test_world_safety_exploration_protected import (
    cfg as protected_cfg, identity as protected_identity, population as protected_population,
    install_gather, _native_worker,
)
from test_world_safety_exploration_groups import records, rollout, state
from starVLA.rl.qwen_world_safety_exploration import inversion_credit as I
from starVLA.rl.qwen_world_safety_exploration import protected_credit as P
from starVLA.rl.qwen_world_safety_exploration.credit import GroupCredit, canonical_components
from starVLA.rl.qwen_world_safety_exploration.inversion_cancellation import cancel_inversions


def cfg(g):
    out=protected_cfg(g);out['research']['structured_advantage']=I.SCHEMA
    return out


def identity(g):
    out=protected_identity(g);out['recipe']=cfg(g)
    return out


def population(g):
    x,recs,_,_,_=protected_population(g)
    # A mixed feasible group with a lower-canonical, Comfort0 positive witness.
    recs[1][1]['metrics']['comfort']=0.
    m=recs[1][1]['metrics'];score=(5*m['ego_progress']+5*m['time_to_collision_within_bound'])/12
    m['score']=score;recs[1][1]['readouts']['canonical_PDMS']=score
    # Complete cancellation of all opposing signs -> PG0 with old mask retained.
    x[4,:,:3]=1;x[4,:g//2,3]=.9;x[4,g//2:,3]=1
    recs[4]=records(x[4])
    for j,r in enumerate(recs[4]):
        m=r['metrics'];m['comfort']=1. if j<g//2 else 0.
        m['score']=(5*m['ego_progress']+5+2*m['comfort'])/12
        r['readouts']['canonical_PDMS']=m['score']
    # High negatives without a lower-score positive witness remain unchanged.
    x[3,:,:3]=1;x[3,:,3]=torch.linspace(.8,1,g,dtype=torch.float64)
    recs[3]=records(x[3])
    for r in recs[3]:
        m=r['metrics'];m['comfort']=1.;m['score']=(5*m['ego_progress']+7)/12;r['readouts']['canonical_PDMS']=m['score']
    rewards=torch.cat([canonical_components(r,g)['canonical_rewards'] for r in recs])
    material=torch.cat((rewards.double().unsqueeze(-1),x),-1)
    return x,recs,rewards,material,None


def test_pair_math_conservation_unchanged_candidates_new_PG0_and_recovery_guards(monkeypatch):
    for g in (8,16):
        x,recs,rewards,material,_=population(g)
        credit=I.InversionCredit(g);base=GroupCredit(g);I.validate_inversion_recipe(cfg(g))
        for rank in range(8):
            got=credit.advantages(rewards[rank:rank+1],x[rank:rank+1],records=recs[rank],all_material=material)
            old=base.advantages(rewards[rank:rank+1],x[rank:rank+1],all_material=material)
            info=got['protection'];changed=info['eligible']
            pure=cancel_inversions(old['A_raw'],info['quality_material'][:,:,1],x[rank:rank+1],info['quality_material'][:,:,0])
            assert torch.equal(got['A_raw'],pure['updated_rawA']) and info['pairs']==pure['pairs']
            for k in ('A_safe','normalization_std','adv_mask'):assert torch.equal(got[k],old[k])
            assert torch.equal(got['advantages'][~changed.reshape(-1)],old['advantages'][~changed.reshape(-1)])
            assert torch.equal(got['A_raw'][~changed],old['A_raw'][~changed])
            torch.testing.assert_close(got['A_raw'].sum(),old['A_raw'].sum(),atol=1e-14,rtol=0)
            torch.testing.assert_close(got['A_safe']+got['A_EP'],got['A_raw'],atol=1e-14,rtol=0)
            if rank==4:
                assert old['adv_mask'].sum()==g and got['adv_mask'].sum()==g
                assert got['A_raw'].eq(0).all() and got['statistics']['inversion_credit_new_PG0_groups']==1
            if rank==3:
                assert not info['activated'] and torch.equal(got['advantages'],old['advantages'])
                protected=P.ProtectedCredit(g).advantages(rewards[rank:rank+1],x[rank:rank+1],records=recs[rank],all_material=material)
                assert protected['protection']['activated']  # distinct from one-sided zeroing
        calls=install_gather(monkeypatch,material)
        rs=[credit.finalize(rollout(rank,g,recs[rank])) for rank in range(8)]
        assert len(calls)==8
        for u in range(4):
            ss=[I.validate_inversion_boundary(state(g,u),r,identity(g),credit=credit) for r in rs]
            I.validate_inversion_ranks(ss,credit=credit)
        for field in ('pairs','newraw','delta','quality','mask','protected'):
            bad=copy.deepcopy(rs[0]);info=bad['inversion_credit']
            if field=='pairs':info['pairs'][0]['transfer']+=.001
            if field=='newraw':info['updated_raw_A'][0,0]+=.01
            if field=='delta':info['delta'].requires_grad_(True)
            if field=='quality':info['quality_material'][0,0,0]=0
            if field=='mask':bad['adv_mask'][0]=0
            if field=='protected':bad['protected_credit']={}
            with pytest.raises(ValueError):
                summaries=[I.validate_inversion_boundary(state(g,3),bad if rank==0 else r,identity(g),credit=credit) for rank,r in enumerate(rs)]
                I.validate_inversion_ranks(summaries,credit=credit)
        with pytest.raises(ValueError):
            wrong=copy.deepcopy(ss);wrong[4]['advantage_statistics']['inversion_credit_new_PG0_groups']=0
            I.validate_inversion_ranks(wrong,credit=credit)
    class Wrong(P.ProtectedRuntimeMixin,I.InversionRuntimeMixin):pass
    class WrongOtherOrder(I.InversionRuntimeMixin,P.ProtectedRuntimeMixin):pass
    for cls in (Wrong,WrongOtherOrder):
        with pytest.raises(ValueError):cls().make_credit(cfg(8))


def test_existing_native8_pending_four_inner_path_parameterized_for_inversion(tmp_path):
    torch.set_num_threads(1)
    mp.start_processes(_native_worker,args=(str(tmp_path/'gloo'),str(tmp_path/'native'),'inversion'),
                       nprocs=8,join=True,start_method='fork')
    assert (tmp_path/'native/NATIVE_CPU_PASS').is_file()
