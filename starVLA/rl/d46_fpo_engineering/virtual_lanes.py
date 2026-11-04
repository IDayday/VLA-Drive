"""B64/G16/M2 on 16 or 32 ranks with the original eight sampler streams."""
import torch
import torch.distributed as dist
from datetime import timedelta
from starVLA.rl.d44_native_rl.recovery import Boundary, checkpoint
from starVLA.rl.d46_world16.native import STATE_SCHEMA, validate_virtual, validate_fpo_material
from starVLA.rl.d46_world16.planning import fixed_scene_rng_plan, assert_scene_rng_end


def slots(rank, world):
    if world not in (16, 32) or not 0 <= rank < world:
        raise ValueError('Qualified B64 execution only supports 16/32 ranks')
    size = 64 // world
    start = (rank // 8) * size
    return tuple((start + j) * 8 + rank % 8 for j in range(size))


class SceneStream:
    def __init__(self, base): self.base = base
    def __getattr__(self, name): return getattr(self.base, name)
    def at(self, cursor, rank, local_scene, world):
        position = slots(rank, world)[local_scene]
        return self.base.at(cursor, position % 8, position // 8, 8)


class Sampler:
    @staticmethod
    def init_control_group():
        from starVLA.rl.simwam_recipe import checkpoint as source
        if source.CONTROL_GROUP is not None or dist.get_world_size() not in (16,32):
            raise ValueError('Invalid virtual-lane control group')
        source.CONTROL_GROUP = dist.new_group(backend='gloo', timeout=timedelta(seconds=600))
    def before_collection(self, state, generator, rank, world):
        validate_virtual(state)
        self.rank, self.world = rank, world
        self.indices = [p // 8 for p in slots(rank, world)]
        self.plan = fixed_scene_rng_plan(state['world16_virtual_sampler']['lanes'][rank % 8], 'D3_FPO', mc=4)
        self.seen = []
        self.cpu_rng = torch.get_rng_state().clone()
        self.cuda_rng = torch.cuda.get_rng_state().clone()
    def before_scene(self, generator, index):
        if index != len(self.seen): raise ValueError('Scene ordinal changed')
        generator.set_state(self.plan['starts'][self.indices[index]])
    def after_scene(self, generator, index):
        assert_scene_rng_end(generator, self.plan['ends'][self.indices[index]])
        self.seen.append(index)
    def after_collection(self, state, generator, rank, world):
        error = None
        if (self.seen != list(range(64//world)) or not torch.equal(self.cpu_rng, torch.get_rng_state())
                or not torch.equal(self.cuda_rng, torch.cuda.get_rng_state())):
            error = 'Scene count or global RNG consumption changed'
        checkpoint.check_errors(error)
        rows = checkpoint.gather(dict(rank=rank, end=self.plan['next_virtual_lane_state']))
        rows.sort(key=lambda r:r['rank'])
        if [r['rank'] for r in rows] != list(range(world)):
            raise ValueError('Missing sampler rank')
        if any(not torch.equal(rows[r]['end'], rows[r%8]['end']) for r in range(world)):
            raise ValueError('Virtual sampler lanes disagree')
        state['world16_virtual_sampler'] = dict(schema=STATE_SCHEMA,
            lanes=[r['end'].clone() for r in rows[:8]], committed_collections=state['collections']+1)
    @staticmethod
    def json_result(state):
        from starVLA.rl.d46_world16.runtime import World16Extension
        return World16Extension.json_result(state)


class DistributedBoundary(Boundary):
    @staticmethod
    def validate_boundary(state, batch, identity, optimizer=None, scheduler=None, **kw):
        world = identity['world']; cfg = identity['recipe']
        if world not in (16,32) or (cfg['global_scenes'],cfg['group'],cfg['inner_updates']) != (64,16,2):
            raise ValueError('Changed distributed scientific batch')
        validate_fpo_material(batch,cfg)
        value = Boundary.validate_boundary(state,batch,identity,optimizer,scheduler,**kw)
        value['virtual8_rng_sha256'] = validate_virtual(state)
        return value
    @staticmethod
    def validate_ranks(rows):
        Boundary.validate_ranks(rows)
        if len({r['virtual8_rng_sha256'] for r in rows}) != 1:
            raise ValueError('Different virtual RNG banks')
        return rows
