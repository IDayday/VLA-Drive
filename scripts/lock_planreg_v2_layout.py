"""Create a layout lock only from an executed V2 full-path profiling report."""
import argparse
import json
from pathlib import Path
from navsim.agents.EpisodeDrive.planreg_v2.agent import file_sha256

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--microbatch',type=int,required=True);p.add_argument('--accumulate',type=int,required=True)
    p.add_argument('--gpus-per-node',type=int,required=True);p.add_argument('--workers',type=int,required=True)
    args=p.parse_args()
    root=Path(args.run)
    report=json.loads((root/'validation.json').read_text())
    metadata=json.loads((root/'run_metadata.json').read_text())
    config=json.loads((root/'resolved_config.json').read_text())
    if report['optimizer_steps']<32 or report['peak_allocated_gib']>=72 or not all(report['horizons_valid']):
        raise ValueError('Insufficient full-WM, finite 32-step memory profile')
    if not config['world_model_enabled'] or not report['fp32_trainable']:
        raise ValueError('Profile silently disabled part of V2')
    if args.microbatch*args.accumulate*metadata['world_size']!=metadata['global_batch']:
        raise ValueError('Do not extrapolate a measured profile into a different layout')
    if (args.microbatch,args.accumulate,args.workers)!=(metadata['microbatch'],metadata['accumulate'],metadata['num_workers']):
        raise ValueError('Same global batch is insufficient: exact profiled microbatch/accumulation/workers required')
    if metadata['world_size']%args.gpus_per_node:
        raise ValueError('Invalid GPU/node topology')
    if metadata['global_batch']!=128:
        raise ValueError('Formal target GB128 requires measured accumulated layout, not a 1-GPU unit smoke')
    lock=dict(architecture_version=config['architecture_version'],passed=True,
        global_batch=metadata['global_batch'],world_size=metadata['world_size'],gpus_per_node=args.gpus_per_node,
        microbatch=args.microbatch,accumulate=args.accumulate,workers=args.workers,
        peak_allocated_gib=report['peak_allocated_gib'],source_sha256=file_sha256(root/'validation.json'))
    if Path(args.output).exists():raise FileExistsError('New immutable layout lock required')
    Path(args.output).write_text(json.dumps(lock,indent=2))
