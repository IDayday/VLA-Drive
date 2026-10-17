"""Finalize metadata only after the separately admitted factory has succeeded."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from tools.st_world_policy.teacher_provider import pin, pinned, validate_profile_recipe
from tools.st_world_policy.training_config import validate_config
from starVLA.dataloader.st_world_targets import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--factory-root',required=True)
    p.add_argument('--checkpoint-output-root',required=True)
    args=p.parse_args()
    root=Path(args.factory_root).resolve()
    checkpoint=Path(args.checkpoint_output_root).resolve()
    if root==checkpoint or root in checkpoint.parents or checkpoint in root.parents:
        p.error('Metadata and checkpoint namespaces must be separate')
    original=pin(root/'configured/training_config.json')
    cfg=pinned(original)
    old_output=cfg['runtime']['output_root']
    cfg=deepcopy(cfg)
    cfg['runtime']['output_root']=str(checkpoint/'runs')
    d=pinned(cfg['data']['teacher_provider']['descriptor'])
    validate_profile_recipe(d['profile_admission'],cfg)
    validate_config(cfg)
    atomic_json(root/'TRAINING_CONFIG.json',cfg,immutable=True)
    atomic_json(root/'EXECUTION_CONFIG_LINEAGE.json',{
        'parent':original,'child':pin(root/'TRAINING_CONFIG.json'),
        'changed_field':'runtime.output_root','before':old_output,'after':cfg['runtime']['output_root'],
        'checkpoint_directory_created':False,'training_launched':False,'formal_approved':False},immutable=True)
    print(json.dumps({'training_config':pin(root/'TRAINING_CONFIG.json'),'training_started':False}))


if __name__=='__main__':
    main()
