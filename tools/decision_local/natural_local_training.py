"""Only versioned data/forward hooks; the existing small-head trainer is reused."""
from dataclasses import dataclass
import torch
from starVLA.dataloader.recovery_plan_dataset import repair_inputs_targets,repair_oracle
from starVLA.model.modules.decision_local.natural_error_contract import LocalTrainingTargets,local_prediction_loss,certified_repair_loss
from starVLA.model.modules.decision_local.minimal_repair_contract import execution_view
from starVLA.model.modules.decision_local.consequence_contract import predicted_relations

NATURAL_KINDS=('N_NONE','N_CV','N_FUT','P_ABS','P_DIFF','R_CURRENT','R_ABS_FUTURE','R_DIFF_FUTURE')


@dataclass(frozen=True)
class NaturalTargets(LocalTrainingTargets):
    original_ego_GT: torch.Tensor


def natural_forward(model,kind,batch,predictor=None):
    if kind not in NATURAL_KINDS:raise ValueError('Unsupported natural-error experiment')
    current,plan,queries,old=repair_inputs_targets(batch)
    def joined(key):return torch.cat((batch[key][:,0],batch[key][:,2,...,1:]),-1)
    # Kinematic-invalid variants are not visual task truth. Native input is
    # always retained, including native policy kinematic errors.
    kinetic=batch['kinematic_valid'].clone();kinetic[:,0]=True
    mask=kinetic[...,None,None]
    target=NaturalTargets(old.relations,old.relation_validity,joined('positive_known')&mask,
        joined('negative_known')&mask,old.repair_target,old.repair_target_valid,old.offline_case_status,old.original_ego_GT)
    if kind.startswith('P_'):
        raw=model(execution_view(current,plan),queries)
        loss=local_prediction_loss(raw.float(),target,difference_weight=float(kind=='P_DIFF'))
        return raw,{'total':loss['loss'],'absolute':loss['absolute'],'difference':loss['difference'],**loss['channels']},target
    if kind in ('N_CV','N_FUT'):
        output=model.oracle_probe(current,plan,queries,repair_oracle(batch,kind[2:]))
    elif kind=='N_NONE':output=model.camera_only(current,plan,queries)
    else:
        if predictor is None or predictor.training or any(p.requires_grad for p in predictor.parameters()):
            raise ValueError('R requires its frozen predictor, trained on the other recording role')
        with torch.no_grad():values=predicted_relations(predictor(execution_view(current,plan),queries).float())
        if kind=='R_CURRENT':values=torch.cat((values[...,:3],values[...,1:3]),-1)
        output=model.camera_only(current,plan,queries,values)
    loss,_=certified_repair_loss(output,target)
    return output,{'total':loss,'trajectory':loss},target
