"""Bounded fixed-pool selection study. Errors have nonzero exit and JSON status."""
import argparse
import json
import sys
from .config import load


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('bind','split-inner','audit-existing','lock-refit-plan','audit-process-labels'):
        q=sub.add_parser(name);q.add_argument('--config',required=True)
    for name in ('train-diagnostic','train-refit','train-overfit','resume'):
        q=sub.add_parser(name);q.add_argument('--config',required=True);q.add_argument('--kind',choices=('M_scene','M_set'),required=True)
        q.add_argument('--device',default='cuda:0');q.add_argument('--resume',nargs='?',const='auto')
        q.add_argument('--stop-after',type=int)
        if name=='resume':q.add_argument('--phase',choices=('diagnostic','refit','overfit'),required=True)
    for name in ('evaluate-external','calibrate','evaluate-dev','benchmark','run-current-state-probe','run-study'):
        q=sub.add_parser(name);q.add_argument('--config',required=True);q.add_argument('--device',default='cuda:0')
        if name=='benchmark':
            q.add_argument('--samples',type=int,default=100);q.add_argument('--warmup',type=int,default=10)
        if name=='run-current-state-probe':
            q.add_argument('--prepare-only',action='store_true');q.add_argument('--workers',type=int,default=4)
            q.add_argument('--resume',nargs='?',const='auto')
        if name=='run-study':q.add_argument('--resume',action='store_true')
    q=sub.add_parser('export-research-bundle');q.add_argument('--config',required=True)
    q.add_argument('--kind',choices=('M_scene','M_set'),required=True)
    args=p.parse_args(argv)
    try:
        config=load(args.config)
        if args.command=='bind':
            from .binding import bind
            r=bind(config);summary=dict(status='COMPLETE',binding_hash=r['hash'])
        elif args.command=='split-inner':
            from .data import split_inner
            r=split_inner(config);summary=dict(status='COMPLETE',fit_scenes=r['fit']['scenes'],fit_logs=r['fit']['logs'],
                                              val_scenes=r['validation']['scenes'],val_logs=r['validation']['logs'],hash=r['hash'])
        elif args.command=='audit-existing':
            from .audit import run
            summary=run(config)
        elif args.command=='lock-refit-plan':
            from .training import lock_refit_plan
            r=lock_refit_plan(config);summary=dict(status='COMPLETE',hash=r['hash'],updates={k:v['updates'] for k,v in r['models'].items()})
        elif args.command=='audit-process-labels':
            from .process_audit import run
            summary=run(config)
        elif args.command in ('evaluate-external','evaluate-dev'):
            from .evaluation import evaluate_role
            role='stage_val' if args.command=='evaluate-external' else 'dev_report'
            r=evaluate_role(config,role,args.device)
            summary=dict(status='COMPLETE',role=role,scenes=r['requested_scenes'],report=str(config.root/'evaluation'/role/'RESULTS.json'))
        elif args.command=='calibrate':
            from .evaluation import calibrate
            r=calibrate(config,args.device);summary=dict(status='COMPLETE',hash=r['hash'],models={k:dict(status=v['status'],gamma=v['gamma'],always_Base=v['always_Base']) for k,v in r['models'].items()})
        elif args.command=='export-research-bundle':
            from .bundle import export
            summary=dict(status='COMPLETE',bundle=export(config,args.kind),ACTIVE_updated=False)
        elif args.command=='benchmark':
            from .benchmark import run
            r=run(config,args.device,args.samples,args.warmup);summary=dict(status=r['status'],report=str(config.root/'latency/BENCHMARK.json'))
        elif args.command=='run-current-state-probe':
            from .privileged_probe import run,prepare
            r=prepare(config,args.workers) if args.prepare_only else run(config,args.device,args.resume,args.workers)
            summary=dict(status=r['status'],report=str(config.root/'current_probe'),export_allowed=False)
        elif args.command=='run-study':
            from .workflow import run
            summary=run(config,args.device,args.resume)
        else:
            from .training import train_model
            phase=args.phase if args.command=='resume' else args.command.removeprefix('train-')
            summary=train_model(config,args.kind,phase,args.device,args.resume or ('auto' if args.command=='resume' else None),args.stop_after)
        print(json.dumps(summary,allow_nan=False));return 0
    except Exception as e:
        print(json.dumps(dict(status='FAILED',error=type(e).__name__,reason=str(e))),file=sys.stderr);return 1


if __name__=='__main__':sys.exit(main())
