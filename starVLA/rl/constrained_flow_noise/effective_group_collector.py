"""Bounded DAPO-style whole-group collection in deterministic input order.

No Ray trainer is copied. Unlike the upstream exception at the generation cap,
this task accepts a masked partial/empty batch and counts every raw exposure.
"""
import copy
import torch
from .contracts import fingerprint


class EffectiveGroupCollector:
    def __init__(self, behavior_identity, lambda_snapshot, *, target=16, limit=32):
        if (target, limit) != (16, 32):
            raise ValueError('Registered collector is target16/cap32')
        self.behavior_identity = copy.deepcopy(behavior_identity)
        self.lambda_snapshot = lambda_snapshot.detach().cpu().clone()
        self.target, self.limit = target, limit
        self.rows, self.accepted, self.pending = [], [], {}

    def next_count(self, remaining_raw_budget):
        if self.done or remaining_raw_budget <= 0: return 0
        # First16 is a fixed uniform stream prefix, regardless of filtering.
        # Refill in bounded groups of up to8; surplus never enters another version.
        return min(16 if not self.rows else 8, self.limit-len(self.rows), remaining_raw_budget)

    @property
    def done(self):
        return len(self.accepted) >= self.target or len(self.rows) >= self.limit

    def add_returned(self, indexed_rows, behavior_identity, lambda_snapshot):
        if behavior_identity != self.behavior_identity or not torch.equal(lambda_snapshot.detach().cpu(), self.lambda_snapshot):
            raise ValueError('Behavior or lambda changed while collecting')
        for index, row in indexed_rows:
            if index < len(self.rows) or index in self.pending or index >= self.limit:
                raise ValueError('Repeated/out-of-range collector index')
            if row['advantage'].shape != (16,) or not torch.isfinite(row['advantage']).all():
                raise ValueError('Complete finite group advantage required')
            self.pending[index] = row
        while len(self.rows) in self.pending:
            index = len(self.rows); row = self.pending.pop(index)
            active = bool((row['advantage'] != 0).any())
            reason = 'ACCEPTED' if active and len(self.accepted) < self.target else ('SURPLUS_SAME_BEHAVIOR' if active else 'ZERO_FINAL_ADVANTAGE')
            self.rows.append({**row, 'reason': reason})
            if reason == 'ACCEPTED': self.accepted.append(index)

    def first_costs(self):
        if self.pending or not self.rows:
            raise ValueError('First-batch statistics require contiguous returned groups')
        return torch.stack([x['costs'] for x in self.rows[:16]])

    def state_dict(self):
        return copy.deepcopy(vars(self))

    @classmethod
    def restore(cls, state):
        obj = cls(state['behavior_identity'], state['lambda_snapshot'], target=state['target'], limit=state['limit'])
        obj.rows, obj.accepted, obj.pending = copy.deepcopy(state['rows']), list(state['accepted']), copy.deepcopy(state['pending'])
        if obj.accepted != [i for i,r in enumerate(obj.rows) if r['reason'] == 'ACCEPTED']:
            raise ValueError('Collector restore changed selection')
        return obj
