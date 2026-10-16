"""Strict contracts for the registered spatial and temporal comparisons.

Teacher channel count and actual tensor layout come from a verified cache
contract. Experiment constants only check the intended controlled comparison.
Sparse position readout and temporal differencing are project adaptations, not
claims that DINO-Foresight implemented these exact objectives.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from collections.abc import Mapping
import math
from starVLA.dataloader.st_world_schema import TeacherFeatureContract


SPATIAL_CELLS = {
    'S-P0': {'image_hw': (192, 256), 'pool': 2, 'world': 48, 'positions': 48},
    'S-P1': {'image_hw': (288, 384), 'pool': 3, 'world': 48, 'positions': 48},
    'S-P2': {'image_hw': (288, 384), 'pool': 1, 'world': 48, 'positions': 96},
    'S-P3': {'image_hw': (288, 384), 'pool': 1, 'world': 96, 'positions': 96},
}
TEMPORAL_CELLS = {
    'T0': (False, False, False),
    'T1': (True, False, False),
    'T2': (False, True, False),
    'T3': (True, True, False),
    'T4': (True, True, True),
}
FUTURE_TIMES_S = (.5, 1., 2., 4.)
VIEWS = ('CAM_F0', 'CAM_L0', 'CAM_R0')


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def _strict_kwargs(cls, value):
    if not isinstance(value, Mapping):
        raise ValueError(f'{cls.__name__} requires a mapping')
    unknown = set(value) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f'Unknown {cls.__name__} fields: {sorted(unknown)}')
    return dict(value)


@dataclass(frozen=True)
class SpaceTimeConfig:
    cell: str
    teacher: TeacherFeatureContract
    world_queries_per_view: int
    sampled_positions: int
    current: bool = True
    future: bool = False
    delta: bool = False
    history: bool = False
    current_weight: float = 1.
    future_weight: float = 0.
    delta_weight: float = 0.
    spatial_parent: str | None = None
    future_times_s: tuple = FUTURE_TIMES_S
    readout_dim: int = 512
    readout_layers: int = 2
    readout_heads: int = 8

    @classmethod
    def from_dict(cls, value):
        kwargs = _strict_kwargs(cls, value)
        if 'teacher' in kwargs and not isinstance(kwargs['teacher'], TeacherFeatureContract):
            kwargs['teacher'] = TeacherFeatureContract.from_dict(kwargs['teacher'])
        if 'future_times_s' in kwargs:
            if not isinstance(kwargs['future_times_s'], (list, tuple)):
                raise ValueError('Physical future times require a resolved sequence')
            kwargs['future_times_s'] = tuple(kwargs['future_times_s'])
        try:
            result = cls(**kwargs)
        except TypeError as exc:
            raise ValueError(f'Incomplete space-time config: {exc}') from exc
        return result.validate()

    @property
    def world_queries(self):
        return len(self.teacher.view_names) * self.world_queries_per_view

    def to_dict(self):
        result = {field.name: getattr(self, field.name) for field in fields(self)}
        result['teacher'] = self.teacher.to_dict()
        result['future_times_s'] = list(self.future_times_s)
        return result

    def validate(self):
        if not isinstance(self.teacher, TeacherFeatureContract):
            raise ValueError('Verified teacher feature contract required')
        self.teacher.validate()
        if not isinstance(self.cell, str):
            raise ValueError('A registered string cell name is required')
        if self.teacher.view_names != VIEWS:
            raise ValueError('Registered current three-view order is required')
        for name in ('world_queries_per_view', 'sampled_positions', 'readout_dim',
                     'readout_layers', 'readout_heads'):
            _positive_int(getattr(self, name), name)
        if self.readout_dim % self.readout_heads:
            raise ValueError('Readout width must divide into heads')
        if self.future_times_s != FUTURE_TIMES_S:
            raise ValueError('This registered study fixes physical future times .5,1,2,4')
        for name in ('current', 'future', 'delta', 'history'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'{name} must be a Boolean, not an unresolved string')
        if self.cell in SPATIAL_CELLS:
            if self.spatial_parent is not None or (self.history, self.future, self.delta) != (False, False, False):
                raise ValueError('Spatial cells use current input/current supervision only')
            spatial = self.cell
        elif self.cell in TEMPORAL_CELLS:
            if not isinstance(self.spatial_parent, str) or self.spatial_parent not in SPATIAL_CELLS:
                raise ValueError('Temporal cell must bind the selected spatial parent')
            if (self.history, self.future, self.delta) != TEMPORAL_CELLS[self.cell]:
                raise ValueError('Temporal cell name and actual objectives disagree')
            spatial = self.spatial_parent
        else:
            raise ValueError('Unknown registered spatial/temporal cell')
        spec = SPATIAL_CELLS[spatial]
        if not self.current:
            raise ValueError('All registered spatial/temporal cells retain current supervision')
        pool = self.teacher.pooling
        expected_pool = {'kind': 'none' if spec['pool'] == 1 else 'avg',
                         'kernel_hw': (spec['pool'], spec['pool']),
                         'stride_hw': (spec['pool'], spec['pool'])}
        if (self.teacher.image_hw, self.teacher.patch_hw, dict(pool),
            self.world_queries_per_view, self.sampled_positions) != (
            spec['image_hw'], (16, 16), expected_pool, spec['world'], spec['positions']):
            raise ValueError('Config changes the registered spatial comparison')
        if self.sampled_positions > math.prod(self.teacher.output_grid_hw):
            raise ValueError('Cannot sample nonexistent teacher positions')
        for enabled, weight, name in (
            (self.current, self.current_weight, 'current'),
            (self.future, self.future_weight, 'future'),
            (self.delta, self.delta_weight, 'delta'),
        ):
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight):
                raise ValueError(f'{name} weight must be finite and resolved')
            if (enabled and weight <= 0) or (not enabled and weight != 0):
                raise ValueError(f'{name} objective and weight disagree')
        if self.current_weight != 1.:
            raise ValueError('Current feature weight is fixed at one')
        return self
