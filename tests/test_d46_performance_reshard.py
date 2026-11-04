import copy
import torch
from starVLA.rl.d46_fpo_engineering.virtual_lanes import slots
from starVLA.rl.d46_fpo_engineering.reshard import repartition
from starVLA.rl.d46_world16.planning import fixed_scene_rng_plan
from starVLA.rl.d44_native_rl.recovery import material_digest


def test_all_scenes_once_same_virtual_rng_and_ddp_weights():
    start=torch.Generator().manual_seed(42).get_state()
    plan=fixed_scene_rng_plan(start,'D3_FPO')
    for world in(16,32):
        order=[s for r in range(world)for s in slots(r,world)]
        assert sorted(order)==list(range(64))
        for rank in range(world):
            for slot in slots(rank,world):
                g=torch.Generator().set_state(plan['starts'][slot//8])
                assert torch.equal(g.get_state(),plan['starts'][slot//8])
        # Rank mean of weighted local sums equals global scene mean.
        values=torch.arange(64,dtype=torch.float64)
        result=sum(values[list(slots(r,world))].sum()*world/64 for r in range(world))/world
        assert result==values.mean()


def test_pending_roundtrip_preserves_all_material_and_virtual_bank():
    rows=[]
    for rank in range(16):
        scenes=[dict(chain=torch.tensor([s]),old_mean=torch.tensor([s+.5]))for s in slots(rank,16)]
        batch=dict(scenes=scenes,behavior_update=20,active_global_scenes=64)
        batch['digest']=material_digest(batch)
        rows.append(dict(state={'updates':21,'sampler':torch.tensor([42])},rollout=batch,
                         rng={'rank':rank},identity_sha256='old'))
    distributed=repartition(rows,16,32)
    restored=repartition(distributed,32,16)
    for old,new in zip(rows,restored):
        assert old['rollout']['digest']==new['rollout']['digest']
        assert torch.equal(old['state']['sampler'],new['state']['sampler'])
        assert old['rng']==new['rng']
    for own in distributed:
        assert len(own['rollout']['scenes'])==2
    ready=copy.deepcopy(rows)
    for own in ready:own['rollout']=None
    assert all(own['rollout']is None for own in repartition(ready,16,32))
