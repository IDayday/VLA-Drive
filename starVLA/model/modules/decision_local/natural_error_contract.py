"""Label-only local facts and independently optional full-trajectory targets."""
from dataclasses import dataclass
import torch
from .consequence_losses import consequence_loss
from starVLA.model.modules.structured_world.losses import final_trajectory_loss

SCHEMA='natural_error_local_supervision_v1'
RELATION_SEMANTICS=('road_body_margin_m','current_annotated_object_distance_m','current_known_overlap',
                    'future_annotated_object_distance_m','future_known_overlap')


@dataclass(frozen=True)
class LocalTrainingTargets:
    relations: torch.Tensor
    relation_valid: torch.Tensor
    positive_known: torch.Tensor
    negative_known: torch.Tensor
    full_repair_target: torch.Tensor
    full_repair_valid: torch.Tensor
    offline_case_status: torch.Tensor

    def __post_init__(self):
        shape=self.relations.shape
        if len(shape)!=4 or shape[-1]!=5:raise ValueError('B,K,T,5 local relation contract')
        if any(v.shape!=shape for v in (self.relation_valid,self.positive_known,self.negative_known)):
            raise ValueError('Per-channel/time masks required')
        if self.full_repair_valid.shape!=(shape[0],shape[2]):raise ValueError('Separate full-plan mask required')
        if self.full_repair_target.shape!=(shape[0],shape[2],4):raise ValueError('Full output remains six/eight encoded points')
        for channel in (2,4):
            pos=self.positive_known[...,channel];neg=self.negative_known[...,channel]
            if bool((pos&neg).any()) or not torch.equal(pos|neg,self.relation_valid[...,channel]):
                raise ValueError('Known overlap needs either a known positive or a supported negative')
            truth=self.relations[...,channel]
            if bool((pos&(truth!=1)).any()) or bool((neg&(truth!=0)).any()):
                raise ValueError('Event truth and support disagree')
        # Full targets cannot be invented for UNKNOWN, UNSOLVED, or local-only
        # evidence. Status never enters model input or selects an online output.
        eligible=(self.offline_case_status==0)|(self.offline_case_status==1)
        if bool((self.full_repair_valid&~eligible[:,None]).any()):
            raise ValueError('Only independently certified KEEP/REPAIR has a full target')


def local_prediction_loss(raw, targets, *, difference_weight=0., distance_scale_m=5., positive_weights=(1.,1.)):
    if not isinstance(targets,LocalTrainingTargets):raise TypeError('Versioned local facts required')
    # Deliberately does NOT intersect with full_repair_valid. All reliable local
    # facts contribute, including scenes whose strict teacher says UNKNOWN.
    return consequence_loss(raw,targets.relations,targets.relation_valid,
        difference_weight=difference_weight,distance_scale_m=distance_scale_m,positive_weights=positive_weights)


def certified_repair_loss(output, targets):
    if not isinstance(targets,LocalTrainingTargets):raise TypeError('Versioned separated targets required')
    safe=torch.where(targets.full_repair_valid[...,None],targets.full_repair_target,
                     torch.zeros_like(targets.full_repair_target))
    return final_trajectory_loss(output.float(),safe.float(),targets.full_repair_valid)


def verify_feature_base(feature_identity, registered_base):
    """A prior full-data S0 cache can never impersonate the new A-only base."""
    if feature_identity.get('anchor_identity')!=registered_base.get('identity'):
        raise ValueError('Feature cache belongs to a different frozen base')
    if feature_identity.get('checkpoint_sha256')!=registered_base['checkpoint']['sha256']:
        raise ValueError('Feature cache checkpoint differs')
    role=registered_base.get('training_role')
    if role not in ('A_train','historical_S0_new_unseen_logs'):
        raise ValueError('Unregistered natural-error base provenance')
