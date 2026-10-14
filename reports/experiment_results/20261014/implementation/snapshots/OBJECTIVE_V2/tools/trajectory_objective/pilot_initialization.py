"""A strict, explicit warm start permitted only in objective_pilot scope."""
import json
from pathlib import Path
from omegaconf import OmegaConf
from starVLA.model.modules.vehicle_joint.initialization import identity_hash,file_sha256


def validate(path,config,scope):
    if not path:return None
    if scope!='objective_pilot':raise ValueError('Learned pilot initialization is forbidden in full/formal training')
    record=json.loads(Path(path).read_text())
    if (record.get('schema')!='trajectory_objective_matched_pilot_initialization_v1' or
        record.get('identity')!=identity_hash({k:v for k,v in record.items() if k!='identity'}) or
        file_sha256(record['weights'])!=record['weights_sha256']):
        raise ValueError('Incomplete or changed pilot initialization')
    cfg=OmegaConf.to_container(config,resolve=True)
    cfg.pop('trajectory_objective',None)
    parent=dict(record['parent_config']);parent.pop('trajectory_objective',None)
    # Finite pilots may tune optimizer hyperparameters, never architecture,
    # data routing, initialization seed or auxiliary-loss configuration.
    cfg=json.loads(json.dumps(cfg));parent=json.loads(json.dumps(parent))
    for item in (cfg,parent):
        trainer=item.get('trainer',{})
        for key in ('learning_rate','gradient_clipping','max_grad_norm'):
            trainer.pop(key,None)
        if 'scheduler_specific_kwargs' in trainer:
            trainer['scheduler_specific_kwargs'].pop('min_lr',None)
    if cfg!=parent:raise ValueError('Pilot must preserve the exact parent architecture/config except ego objective')
    return record
