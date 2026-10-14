"""Verify every registered parameter against actual reconstructed FP32 masters."""
import argparse
import json
from pathlib import Path
import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from starVLA.rl.constrained_flow_noise.contracts import require_dispatch, file_sha, SFT_SHA256
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--calibration',required=True);p.add_argument('--output',required=True)
    p.add_argument('--SFT',default='/mnt/project/DriveDreamer-Policy-flow-grpo/artifacts/action-only-checkpoints-v1/frozen_visual/pytorch_model.pt')
    a=p.parse_args();host=require_dispatch();torch.set_num_threads(4)
    root=Path(a.checkpoint);identity=json.loads((root/'identity.json').read_text())
    if not (root/'COMPLETE').exists() or identity['SFT_sha256']!=SFT_SHA256 or file_sha(a.SFT)!=SFT_SHA256:raise ValueError('Checkpoint/SFT identity mismatch')
    if Path(a.output).exists():raise FileExistsError(a.output)
    original=torch.load(a.SFT,weights_only=True,map_location='cpu',mmap=True)
    original=original.get('module',original)
    expected={'policy.'+k:v.float() for k,v in original.items() if k.startswith('action_model.')}
    cal=json.loads(Path(a.calibration).read_text())
    if identity['variant']=='N_FLOW_NOISE':
        initial=torch.load(cal['noise_initial_path'],map_location='cpu',weights_only=True)
        expected.update({'noise_head.'+k:v.to(torch.bfloat16).float() for k,v in initial.items() if k in identity['closure']['noise_names']})
    actual=get_fp32_state_dict_from_zero_checkpoint(str(root),tag='pytorch_model')
    fixed_buffers=set(actual)-set(expected)
    if not fixed_buffers<={'noise_head.logvar_min','noise_head.logvar_max'}:raise ValueError('Unexpected non-parameter master entries')
    for name in fixed_buffers:
        if not torch.equal(actual[name],initial[name.removeprefix('noise_head.')].to(torch.bfloat16).float()):raise ValueError('Fixed noise bounds changed')
        actual.pop(name)
    if set(actual)!=set(expected):raise ValueError('Master-weight parameter closure differs from original head plus noise MLP')
    entries={}
    for name,before in expected.items():
        after=actual[name]
        entries[name]=dict(dtype=str(after.dtype),finite=bool(torch.isfinite(after).all()),changed_elements=int(torch.count_nonzero(after!=before)),
                           max_abs_change=float((after-before).abs().max()))
    passed=all(v['dtype']=='torch.float32' and v['finite'] and v['changed_elements']>0 for v in entries.values())
    report=dict(status='PASS' if passed else 'FAIL',host=host,checkpoint_identity=identity['sha256'],variant=identity['variant'],
                tensors=len(entries),all_master_tensors_actually_changed=passed,entries=entries,
                reconstruction='Installed DeepSpeed public get_fp32_state_dict_from_zero_checkpoint, all eight saved partitions')
    atomic_json(Path(a.output),report)
    if not passed:raise ValueError('A registered FP32 master parameter had no legitimate change')


if __name__=='__main__':main()
