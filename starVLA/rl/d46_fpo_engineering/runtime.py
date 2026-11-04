"""Qualified execution optimizations around the original native trainer."""
import copy
import json
from pathlib import Path

import torch
import torch.distributed as dist

from starVLA.rl.d44_native_rl.actor import D44Actor
from starVLA.rl.d44_native_rl.batch_runtime import collect_scene
from starVLA.rl.d44_native_rl.config import read
from starVLA.rl.d44_native_rl.data import DataStream
from starVLA.rl.d44_native_rl.execution_cache import install_invariant_cache
from starVLA.rl.d44_native_rl.recovery import Attempt
from starVLA.rl.d46_fpo_verified.runtime import VerifiedExtension
from starVLA.rl.d46_world16.runtime import World16Extension, VirtualSceneStream
from .frozen_weights import FrozenLinearCache
from .joint_flow import JointVerifiedActor, OptimizedVerifiedCollection
from .prefix_cache import collection_prefix_cache
from . import migration


class PerformanceExtension:
    bulk_metrics = True
    make_attempt = staticmethod(Attempt)

    def read_recipe(self, args):
        if not args.resume or args.qualification:
            raise ValueError('Performance continuation requires its own formal native state')
        self.args = args
        self.cfg = read(args.config)
        root = Path(args.output).resolve()
        cp = Path(args.resume).resolve()
        if cp.parent != root/'checkpoints':
            raise ValueError('Only own execution-root native resume')
        self.saved = json.loads((cp/'identity.json').read_text())
        migration.validate(root, self.saved)
        if self.saved['recipe'] != self.cfg:
            raise ValueError('Scientific training recipe changed')
        self.verified = 'verified_improvement' in self.cfg
        self.world = self.saved['world']
        if self.world != (8 if self.verified else 16):
            raise ValueError('Unqualified route/world combination')
        self.actor_factory = JointVerifiedActor if self.verified else D44Actor
        self.boundary = migration.boundary(self.saved)
        self.delegate = OptimizedVerifiedCollection() if self.verified else World16Extension()
        self.delegate.cfg = self.cfg
        self.delegate.saved_identity = self.saved
        self.caches = []
        self.active = False
        return self.cfg

    def validate_world(self, world):
        if world != self.world:
            raise ValueError('Native world size changed')

    def init_control_group(self):
        (VerifiedExtension if self.verified else World16Extension).init_control_group()

    def make_stream(self, cfg):
        stream = DataStream(cfg, cfg['global_scenes'])
        return stream if self.verified else VirtualSceneStream(stream)

    def identity(self, actual):
        # Parent, data, scope, reward protocol and recipe remain byte-equivalent.
        for key, value in actual.items():
            if key not in ('source_files', 'sha256', 'fresh_Adam') and value != self.saved.get(key):
                raise ValueError('Actual scientific contract changed: ' + key)
        migration.validate(self.args.output, self.saved)
        return copy.deepcopy(self.saved)

    def after_restore(self, actor, conditions, stream, state, identity):
        self.actor = actor
        self.state = state
        self.delegate.stream = stream
        # Reuse the native post-restore B1 equality check before installing hooks.
        from starVLA.rl.d46_world16.runtime import compare_restored_B1
        from starVLA.rl.simwam_recipe.contracts import seed_for
        from starVLA.rl.flow_grpo.transactions import atomic_json
        rank = dist.get_rank()
        token, _ = stream.at(state['scene_cursor'], rank, 0, self.world)
        observation = stream.data.observation(token)
        inputs = conditions.processor([observation])
        ego = torch.tensor([observation['proprio']], device=actor.device, dtype=torch.float32)
        noise = torch.randn((1,8,3), generator=torch.Generator().manual_seed(
            seed_for('native_qwen_sourceflow_ODE',2101,token))).to(actor.device)
        payload = torch.load(Path(self.args.resume)/'optimizer.pt', map_location='cpu', weights_only=False)
        errors = compare_restored_B1(actor, payload['adapters'], inputs, ego, noise)
        atomic_json(Path(self.args.output)/f'restored_B1_rank{rank}.json', dict(
            status='PASS', updates=state['updates'], errors=errors, checkpoint=str(self.args.resume)))

    def configure_actor(self, actor):
        origin = self.saved['performance_execution']['origin_updates']
        actor.execution_fast_prefix = self.verified
        actor.execution_fuse_fm_mc = False
        actor.execution_joint_flow_aux = False
        if self.verified:
            self.caches = [install_invariant_cache(p.decision, rotary=True, additive=True)
                           for p in (actor.action_expert, actor.ref_action_expert)]
        # A legacy pending inner update must use its original MC4 layout.
        if self.state['updates'] > origin:
            self.activate()

    def activate(self):
        if self.active:
            return
        for cache in reversed(self.caches):
            cache.close()
        self.caches = []
        fast = self.saved['performance_execution']['profile'] == 'fast'
        self.actor.execution_fast_prefix = True
        self.actor.execution_fuse_fm_mc = fast
        self.actor.execution_joint_flow_aux = fast
        if self.verified and not fast:
            self.caches.extend(install_invariant_cache(p.decision, rotary=True, additive=True)
                               for p in (self.actor.action_expert, self.actor.ref_action_expert))
        self.caches.append(FrozenLinearCache(self.actor))
        self.active = True

    def before_collection(self, state, generator, rank, world):
        self.activate()
        self.delegate.before_collection(state, generator, rank, world)

    def before_scene(self, generator, index):
        self.delegate.before_scene(generator, index)

    def after_scene(self, generator, index):
        self.delegate.after_scene(generator, index)

    def collect_scene(self, actor, *args, **kwargs):
        if not self.active:
            raise RuntimeError('Collection started before execution profile activation')
        if self.verified:
            return self.delegate.collect_scene(actor, *args, **kwargs)
        with collection_prefix_cache(actor):
            return collect_scene(actor, *args, **kwargs, validate_sample_mean=False)

    def after_collection(self, state, generator, rank, world):
        self.delegate.after_collection(state, generator, rank, world)

    def json_result(self, state):
        return self.delegate.json_result(state)
