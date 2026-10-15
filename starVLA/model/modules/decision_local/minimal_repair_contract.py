"""Versioned execution inputs. A historical clean q0 is never a repair input."""
from dataclasses import dataclass
from enum import IntEnum
import torch
from .consequence_contract import ModelInputs, PredictionQueries
from .residual_refiner import ResidualRefiner

SCHEMA = 'scene_conditioned_minimal_repair_v1'


class CaseStatus(IntEnum):
    KEEP = 0
    REPAIR = 1
    UNCERTAIN = 2
    OUT_OF_SCOPE = 3
    UNSOLVED = 4
    AMBIGUOUS = 5
    TASK_UNCERTAIN = 6


class StrictMapping:
    @classmethod
    def from_mapping(cls, values):
        if set(values) != set(cls.__dataclass_fields__):
            raise ValueError(f'{cls.__name__} rejects missing or extra fields, including targets/clean q0')
        return cls(**values)


@dataclass(frozen=True)
class CurrentFeatures(StrictMapping):
    H_A: torch.Tensor
    W: torch.Tensor
    state: torch.Tensor
    body: torch.Tensor


@dataclass(frozen=True)
class PlanInputs(StrictMapping):
    plan_in_physical: torch.Tensor
    plan_in_encoded: torch.Tensor
    physical_times: torch.Tensor

    def __post_init__(self):
        B, T, C = self.plan_in_physical.shape
        if C != 3 or T not in (6, 8) or self.plan_in_encoded.shape != (B, T, 4) or self.physical_times.shape != (T,):
            raise ValueError('Plan shape/horizon mismatch')


@dataclass(frozen=True)
class RepairTargets:
    original_ego_GT: torch.Tensor
    repair_target: torch.Tensor
    repair_target_valid: torch.Tensor
    offline_case_status: torch.Tensor
    relations: torch.Tensor
    relation_validity: torch.Tensor
    target_eligible: torch.Tensor


def execution_view(current, plan):
    """Compatibility wrapper for the unchanged operators; both q0 fields are plan_in.

    The public schema contains no original q0. This adapter preserves the exact
    old network/state inventory, not a second reference trajectory.
    """
    if not isinstance(current, CurrentFeatures) or not isinstance(plan, PlanInputs):
        raise TypeError('Explicit current features and actual PlanInputs required')
    return ModelInputs(current.H_A, current.W, plan.plan_in_encoded,
                       plan.plan_in_physical, current.state, current.body)


class MinimalRepairRefiner(ResidualRefiner):
    def camera_only(self, current, plan, queries, predicted=None):
        if not isinstance(queries, PredictionQueries):
            raise TypeError('PredictionQueries required')
        return super().camera_only(execution_view(current, plan), queries, predicted)

    def oracle_probe(self, current, plan, queries, relations):
        return super().oracle_probe(execution_view(current, plan), queries, relations)
