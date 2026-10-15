"""Version-specific data/forward hooks for the existing native small-head trainer."""
import hashlib,json
import torch
from starVLA.dataloader.recovery_plan_dataset import repair_inputs_targets,repair_oracle
from starVLA.model.modules.decision_local.minimal_repair_contract import execution_view
from starVLA.model.modules.decision_local.consequence_contract import predicted_relations
from starVLA.model.modules.decision_local.consequence_losses import consequence_loss
from starVLA.model.modules.structured_world.losses import final_trajectory_loss

REPAIR_KINDS=('LOG_NONE','LOG_CV','LOG_FUT','MIN_NONE','MIN_CV','MIN_FUT',
              'R_MIN_CURRENT','R_MIN_ABS','R_MIN_DIFF')


def label_semantics(meta):
    fields={k:meta[k] for k in ('schema','recovery','constraints','solver','query_config','observed_free_support','source_files')}
    return hashlib.sha256(json.dumps(fields,sort_keys=True).encode()).hexdigest()


def repair_forward(model,kind,batch,predictor=None):
    current,plan,queries,target=repair_inputs_targets(batch)
    if kind in ('P_ABS','P_DIFF'):
        raw=model(execution_view(current,plan),queries)
        losses=consequence_loss(raw.float(),target.relations,target.relation_validity,difference_weight=float(kind=='P_DIFF'))
        return raw,{'total':losses['loss'],'absolute':losses['absolute'],'difference':losses['difference'],**losses['channels']},target
    if kind not in REPAIR_KINDS:raise ValueError('No silent fallback from unknown repair experiment')
    if kind.endswith(('_CV','_FUT')):
        output=model.oracle_probe(current,plan,queries,repair_oracle(batch,kind.split('_')[-1]))
    elif kind.endswith('_NONE'):
        output=model.camera_only(current,plan,queries)
    else:
        if predictor is None or predictor.training or any(p.requires_grad for p in predictor.parameters()):
            raise ValueError('R must receive its frozen fold-disjoint predictor from its first update')
        with torch.no_grad():relations=predicted_relations(predictor(execution_view(current,plan),queries).float())
        if kind=='R_MIN_CURRENT':relations=torch.cat((relations[...,:3],relations[...,1:3]),-1)
        output=model.camera_only(current,plan,queries,relations)
    truth=target.original_ego_GT if kind.startswith('LOG_') else target.repair_target
    loss,_=final_trajectory_loss(output.float(),truth.float(),target.repair_target_valid)
    return output,{'total':loss,'trajectory':loss},target
