"""One frozen Qwen prefill, current-memory graph rebuilt, G candidate views."""
from starVLA.rl.qwen_world_postcot.contracts import fingerprint
from starVLA.rl.qwen_world_postcot.scope_conditions import MemoryLearningConditions
from starVLA.rl.qwen_world_postcot.scope_contracts import MEMORY_OBSERVATION


class GroupMemoryConditions(MemoryLearningConditions):
    def __init__(self, model, data, parent_identity, actor, *, group_size, processor=None):
        if type(group_size) is not int or group_size not in (8, 16, 32):
            raise ValueError('Only explicitly registered G8/G16 conditions')
        super().__init__(model, data, parent_identity, actor, processor=processor)
        self.group_size = group_size
        self.identity.update(schema='world_Safety_exploration_memory_conditions_v1',
                             group_size=group_size)
        self.sha = fingerprint(self.identity)

    def get(self, token, device):
        if token not in self._tokens:
            raise ValueError('Observation outside the complete registered NAVTRAIN')
        return dict(schema=MEMORY_OBSERVATION, token=token, identity=self.sha,
                    repeats=self.group_size)

    def materialize(self, descriptor, device):
        # The parent's exact one-scene frozen-feature/cache lifetime is retained.
        # Its batch8 views are replaced, not its Qwen or reference calculations.
        value = super().materialize(descriptor, device)
        return dict(value, context=value['base_memory'].expand(self.group_size, -1, -1),
                    valid=value['base_valid'].expand(self.group_size, -1),
                    ego=value['base_ego'].expand(self.group_size, -1))
