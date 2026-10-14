"""Thin orchestration for a fixed, bounded research campaign."""
from iqe.contracts import require


def dispatch(config,args):
    name=args.command
    if name=='train-expert-cell':
        from .training import train_expert
        require(args.cell is not None,'--cell is required')
        return train_expert(config,args.cell,args.device,args.resume,args.stop_after)
    if name=='cache-score-context':
        from .context import build
        return build(config,args.device,args.max_samples,args.shard,args.shards)
    if name in ('evaluate-expert-matrix','extend-specialists','freeze-candidate-pool'):
        from .experiments import evaluate_experts,extend,freeze_pool
        return {'evaluate-expert-matrix':evaluate_experts,'extend-specialists':extend,'freeze-candidate-pool':freeze_pool}[name](config,args.device)
    if name in ('build-deployment-bank','build-rich-bank'):
        from .bank import build_deploy,build_rich
        return (build_deploy if name=='build-deployment-bank' else build_rich)(config,args.device)
    if name=='train-scorer-cell':
        from .scorer_training import train_scorer
        require(args.cell is not None,'--cell is required')
        return train_scorer(config,args.cell,args.device,args.resume,args.stop_after)
    if name in ('evaluate-scorer-matrix','calibrate-selection','evaluate-final-system','benchmark-system'):
        from .evaluation import evaluate_scorers,calibrate,evaluate_final,benchmark
        return {'evaluate-scorer-matrix':evaluate_scorers,'calibrate-selection':calibrate,
                'evaluate-final-system':evaluate_final,'benchmark-system':benchmark}[name](config,args.device)
    if name=='run-specialist-scorer-study':
        from .experiments import run_study
        return run_study(config,args.device,args.resume)
    raise ValueError('unsupported campaign command: '+name)
