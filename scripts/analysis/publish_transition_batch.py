"""Publish a fixed-layout epoch only after native semantic validation succeeds.

Consumes completed real artifacts; no test-fixture generation or status overrides.
The epoch record is published last, and is the only formal-training entry point.
"""
import argparse
import json
import os
from pathlib import Path
import uuid
from starVLA.rl.flow_grpo.batch_profile import kernel_identity, validate_batch_profile
from starVLA.rl.flow_grpo.full_epoch import enforce_full_epoch, epoch_budget
from starVLA.rl.flow_grpo.config import resolve_config
from starVLA.rl.flow_grpo.loading import file_sha
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.acceptance import acceptance_context
from starVLA.rl.flow_grpo.transactions import atomic_json


def pointer(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':file_sha(path)}


def publish(config, root):
    root=Path(root).resolve();cfg,sft=resolve_config(config);configure_numerics()
    pilot=root/'pilot';resume=root/'resumed'
    context=json.loads((pilot/'execution_context.json').read_text())
    os.environ['WORLD_SIZE']=str(context['world_size'])
    if acceptance_context(cfg,context['resume_identity'])!=context:
        raise ValueError('publisher/current source/config differs from native pilot')
    if json.loads((resume/'execution_context.json').read_text())!=context:
        raise ValueError('resume did not use the target profile')
    cache=json.loads((root/'cache_probe/probe.json').read_text())
    batch={'schema_version':1,'status':'QUALIFIED_FIXED_LAYOUT','context':context,'kernels':kernel_identity(),
        'scope':'G16/K10 fixed saved-chain batch160; serial rollout chunk1; BF16 storage, FP32 action arithmetic/ZeRO partitions; no cross-layout equivalence claim',
        'trainable_names':cache['trainable_names'],
        'fp32_gradient_comparisons':[pointer(root/f'math_comparison/grads_rank{i}.json') for i in (1,2)],
        'bf16_rounding_comparison':pointer(root/'bf16_rounding.json'),
        'native_adam':pointer(root/'adam.json'),'export':pointer(root/'export_check/export_equivalence.json'),
        'historical_serial_comparison':pointer(root.parent/'native_gradient_comparison/summary.json')}
    for key,folder in [('fp32_serial_run','fp32_serial'),('fp32_batch_run','fp32_flat'),('fp32_rounding_oracle','fp32_oracle'),('bf16_batch_run','bf16_flat')]:
        batch[key]=pointer(root/folder/'report.json')
    batch_path=Path(cfg['runtime']['batch_profile_evidence'])
    if batch_path.exists():
        if json.loads(batch_path.read_text())!=batch:raise ValueError('conflicting batch publication')
    else:atomic_json(batch_path,batch)
    try:validate_batch_profile(pointer(batch_path),cfg,context)
    except Exception:
        batch_path.rename(batch_path.with_name(batch_path.name+'.rejected-'+uuid.uuid4().hex));raise
    rows=[json.loads(s) for s in (pilot/'training.jsonl').read_text().splitlines()]
    dtype_path=root/'pilot_dtype.json';atomic_json(dtype_path,{'ranks':rows[-1]['ranks']})
    legacy=json.loads(Path('/mnt/project/DriveDreamer-Policy-action-rl/runs/action_head/world16/epoch_evidence.json').read_text())
    record={'schema_version':1,'status':'AUTHORIZED_FULL_DATA_EXPERIMENT','scope':batch['scope'],
        'context':context,'budget':epoch_budget(103288),'split_sha256':file_sha(cfg['paths']['split_manifest']),
        'pilot_context':pointer(pilot/'execution_context.json'),'resume_context':pointer(resume/'execution_context.json'),
        'pilot_config':pointer(pilot/'rl_config.json'),'pilot_control':pointer(root/'pilot_control/result.json'),
        'resume_control':pointer(root/'resumed_control/result.json'),'pilot_launch':pointer(root/'pilot_spec.json'),
        'resume_launch':pointer(root/'resumed_spec.json'),'pilot_training':pointer(pilot/'training.jsonl'),
        'pilot_dtype':pointer(dtype_path),'exact_resume':pointer(root/'exact_resume.json'),
        'pilot_immutable':[pointer(pilot/f'immutable_update000002_rank{i}.json') for i in range(context['world_size'])],
        'velocity_graph':legacy['velocity_graph'],'action_head_cache':pointer(root/'cache_probe/probe.json'),
        'transition_batch':pointer(batch_path)}
    enforce_full_epoch(cfg,context,record=record)
    target=Path(cfg['runtime']['epoch_evidence'])
    if target.exists():
        if json.loads(target.read_text())!=record:raise ValueError('conflicting epoch publication')
    else:atomic_json(target,record)
    print(json.dumps({'status':record['status'],'world_size':context['world_size'],'executable_sha256':context['executable_sha256'],'record':str(target)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--root',required=True)
    a=p.parse_args();publish(a.config,a.root)
