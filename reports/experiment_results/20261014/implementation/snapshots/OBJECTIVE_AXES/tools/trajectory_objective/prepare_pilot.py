"""Freeze mature FP32 weights for matched short objective pilots only."""
import argparse
import fcntl
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch

from tools.foresight.checkpoints import checkpoint_identity
from starVLA.model.modules.vehicle_joint.initialization import file_sha256, identity_hash
from tools.ddpolicy_vehicle.prepare_data import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--training-run',required=True);p.add_argument('--tag',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'EXPORT.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2)
    training,checkpoint=checkpoint_identity(a.training_run,a.tag)
    manifest=a.output/'initialization.json';weights=a.output/'fp32_model.pt'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old['checkpoint']!=checkpoint or file_sha256(weights)!=old['weights_sha256']:
            raise ValueError('Existing pilot initialization changed')
        print(json.dumps(old,indent=2));return
    if weights.exists():raise FileExistsError('Incomplete export; inspect before retry')
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    state=get_fp32_state_dict_from_zero_checkpoint(str(Path(a.training_run)/'checkpoints'),tag=a.tag)
    if any(not torch.isfinite(v).all() for v in state.values() if v.is_floating_point()):
        raise FloatingPointError('Nonfinite pilot parent weights')
    temporary=weights.with_suffix('.tmp');torch.save(state,temporary);temporary.replace(weights)
    record=dict(schema='trajectory_objective_matched_pilot_initialization_v1',checkpoint=checkpoint,
        parent_config=training['config'],parent_run=str(Path(a.training_run).resolve()),
        weights=str(weights.resolve()),weights_sha256=file_sha256(weights),
        purpose='matched training-domain objective pilots only; full runs must use original generic initialization')
    record['identity']=identity_hash(record);atomic_json(manifest,record)
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
