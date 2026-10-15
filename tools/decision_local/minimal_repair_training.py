"""Version-specific data/forward hooks for the existing native small-head trainer."""
import hashlib,json,math
import torch
from starVLA.dataloader.recovery_plan_dataset import repair_inputs_targets,repair_oracle
from starVLA.model.modules.decision_local.minimal_repair_contract import execution_view
from starVLA.model.modules.decision_local.consequence_contract import predicted_relations
from starVLA.model.modules.decision_local.consequence_losses import consequence_loss
from starVLA.model.modules.structured_world.losses import final_trajectory_loss
from starVLA.model.modules.decision_local.minimal_repair_contract import CaseStatus

REPAIR_KINDS=('LOG_NONE','LOG_CV','LOG_FUT','MIN_NONE','MIN_CV','MIN_FUT',
              'R_MIN_CURRENT','R_MIN_ABS','R_MIN_DIFF')


def label_semantics(meta):
    fields={k:meta[k] for k in ('schema','recovery','constraints','solver','query_config','observed_free_support','source_files')}
    return hashlib.sha256(json.dumps(fields,sort_keys=True).encode()).hexdigest()


def load_repair_class_balance(path, kind, train):
    """Explicit training-only follow-up; never infer weights from dev outcomes.

    The receipt binds fixed weights to actual training exposures. It does not
    change the target population or normalize the two classes per microbatch.
    """
    from tools.decision_local.common import read, digest, identity
    if kind not in ('MIN_NONE','MIN_CV','MIN_FUT','R_MIN_CURRENT','R_MIN_ABS','R_MIN_DIFF'):
        raise ValueError('Class balance is a separately registered MIN follow-up only')
    receipt=read(path)
    if receipt.get('schema')!='minimal_repair_class_balance_v1':
        raise ValueError('Unknown class-balance contract')
    if (receipt['labels']!=train.label_identity['identity'] or
        receipt['label_semantics']!=label_semantics(train.label_identity) or
        receipt['fold']!=train.fold_identity or
        receipt['training_index_sha256']!=identity(train.index)):
        raise ValueError('Class weights belong to another training population')
    evidence=receipt['training_exposure_source']
    if digest(evidence)!=receipt['training_exposure_source_sha256']:
        raise ValueError('Training exposure evidence changed')
    counts=read(evidence)['exposure_counts']
    if receipt['counts']!={k:counts[k] for k in ('KEEP','REPAIR','target_eligible')}:
        raise ValueError('Weights must derive from the recorded training exposures')
    keep,repair,total=(counts[k] for k in ('KEEP','REPAIR','target_eligible'))
    if min(keep,repair)<=0 or keep+repair!=total:
        raise ValueError('Two nonempty exhaustive eligible classes required')
    weights=tuple(receipt['weights'][k] for k in ('KEEP','REPAIR'))
    for actual,expected in zip(weights,(total/(2*keep),total/(2*repair))):
        if not math.isfinite(actual) or not math.isclose(actual,expected,rel_tol=1e-12,abs_tol=0):
            raise ValueError('Fixed inverse-frequency weights do not match exposure counts')
    return receipt,weights


def repair_trajectory_loss(output, target, class_weights=None):
    """Native per-scene reduction, with an optional fixed class multiplier.

    The trainer still divides summed microbatch scene losses by the original
    optimizer batch's eligible count. Empty classes contribute exact zero.
    Offline case status affects this loss only, never model inputs/forward.
    """
    loss,_=final_trajectory_loss(output.float(),target.repair_target.float(),target.repair_target_valid)
    if class_weights is None:
        return {'total':loss,'trajectory':loss}
    status=target.offline_case_status[:,None]
    keep,_=final_trajectory_loss(output.float(),target.repair_target.float(),
        target.repair_target_valid & (status==CaseStatus.KEEP))
    repair,_=final_trajectory_loss(output.float(),target.repair_target.float(),
        target.repair_target_valid & (status==CaseStatus.REPAIR))
    weighted=class_weights[0]*keep+class_weights[1]*repair
    return {'total':weighted,'trajectory':weighted,'trajectory_unweighted':loss,
            'trajectory_keep':keep,'trajectory_repair':repair}


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
    if kind.startswith('LOG_'):
        if getattr(model,'repair_class_weights',None) is not None:
            raise ValueError('Class-balanced MIN loss cannot silently change LOG controls')
        loss,_=final_trajectory_loss(output.float(),target.original_ego_GT.float(),target.repair_target_valid)
        losses={'total':loss,'trajectory':loss}
    else:
        losses=repair_trajectory_loss(output,target,getattr(model,'repair_class_weights',None))
    return output,losses,target
