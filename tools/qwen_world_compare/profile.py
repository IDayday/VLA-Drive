"""Profile the existing complete-objective train loop or a sealed deployment."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

if __package__ in(None,''):
    script_dir=Path(__file__).resolve().parent
    sys.path[:]=[x for x in sys.path if Path(x or os.curdir).resolve()!=script_dir]
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def training_command(args):
    if not args.config or not 4<=args.updates<=20:
        raise ValueError('Training profiles need a recipe and a finite4..20 update window')
    command=['--config',str(Path(args.config).resolve()),'--output',str(Path(args.output).resolve()/'run'),
        '--qualification','--joint','--stop-after',str(args.updates),'--profile-no-save',
        '--backend',args.backend,'--profile-dir',str(Path(args.output).resolve()/'phases'),
        '--profile-steps',str(args.updates-2)]
    if args.micro_batch is not None:command+=['--micro-batch',str(args.micro_batch)]
    if args.evidence_override:command+=['--evidence-override',str(Path(args.evidence_override).resolve())]
    return command


def main():
    parser=argparse.ArgumentParser(__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--config');mode.add_argument('--export')
    parser.add_argument('--output',required=True);parser.add_argument('--backend',choices=['zero2','ddp'],default='zero2')
    parser.add_argument('--micro-batch',type=int);parser.add_argument('--updates',type=int,default=4)
    parser.add_argument('--evidence-override');parser.add_argument('--emit-command',action='store_true')
    parser.add_argument('--data');parser.add_argument('--scene-token');parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args();out=Path(args.output)
    if args.config:
        from starVLA.model.modules.qwen_world_compare.factory import load_config
        load_config(args.config)
        command=training_command(args)
        if args.emit_command:
            print(json.dumps(dict(module='tools.qwen_world_compare.train_il',args=command,
                launcher='torchrun --standalone --nproc-per-node=8 -m tools.qwen_world_compare.profile',
                qualification_only=True,formal_initialization_reused=False)))
            return
        if int(os.environ.get('WORLD_SIZE',1))!=8:
            raise ValueError('The DDP/ZeRO2 comparison requires an already leased eight-rank torchrun')
        from tools.qwen_world_compare import train_il
        sys.argv=['tools.qwen_world_compare.train_il',*command]
        train_il.main()
        return
    if args.emit_command:
        raise ValueError('--emit-command applies only to training profiles')
    if not args.data:raise ValueError('Deployment profiling requires a sealed current-input TRAIN data root')
    import torch
    from starVLA.rl.qwen_world_compare.contracts import require_dispatch,file_sha
    from starVLA.rl.qwen_world_compare.deployment import load_export
    from starVLA.rl.qwen_world_compare.profiling import profile_inference,_atomic
    from starVLA.rl.qwen_world_postcot.data import WorldSceneData
    from starVLA.rl.qwen_world_postcot.batching import to_device
    from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
    from starVLA.rl.simwam_recipe.contracts import seed_for
    from starVLA.rl.flow_grpo.reproducibility import configure_numerics
    host=require_dispatch();configure_numerics();torch.cuda.set_device(0)
    out.mkdir(parents=True,exist_ok=True)
    result_path=out/'INFERENCE_PROFILE.json'
    if result_path.exists():raise FileExistsError('Inference profiles use a fresh output invocation')
    tick=time.monotonic();model,meta=load_export(args.export,'cuda');load_seconds=time.monotonic()-tick
    data=WorldSceneData(args.data,role='TRAIN');token=args.scene_token or data.tokens[0]
    if token not in data.tokens:raise ValueError('Inference profile scene must belong to sealed TRAIN')
    observation=data.observation(token)
    processor=QwenObservationProcessor(str(Path(args.export)/'processor'))
    current=to_device(processor([observation]),torch.device('cuda'))
    ego=torch.tensor([observation['proprio']],device='cuda',dtype=torch.float32)
    noise=torch.randn(1,8,3,generator=torch.Generator().manual_seed(
        seed_for('native_qwen_sourceflow_ODE',2101,token))).cuda()
    torch.cuda.reset_peak_memory_stats()
    result=profile_inference(model,current,ego,noise,repeats=args.repeats,warmup=1)
    result.update(host=host,architecture_id=meta['architecture_id'],model_sha256=meta['model_sha256'],
        export_identity_sha256=file_sha(Path(args.export)/'export.json'),
        qualification=meta.get('qualification'),model_load_seconds=load_seconds,
        data_manifest_sha256=data.sha,seed=2101,scene_token=token,performance_score=None)
    _atomic(result_path,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()

