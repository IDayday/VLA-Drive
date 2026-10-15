"""Bounded fixed-pool selection study. Errors have nonzero exit and JSON status."""
import argparse
import json
import sys
from .config import load


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('bind','split-inner','audit-existing','lock-refit-plan'):
        q=sub.add_parser(name);q.add_argument('--config',required=True)
    for name in ('train-diagnostic','train-refit','train-overfit','resume'):
        q=sub.add_parser(name);q.add_argument('--config',required=True);q.add_argument('--kind',choices=('M_scene','M_set'),required=True)
        q.add_argument('--device',default='cuda:0');q.add_argument('--resume',nargs='?',const='auto')
        q.add_argument('--stop-after',type=int)
        if name=='resume':q.add_argument('--phase',choices=('diagnostic','refit','overfit'),required=True)
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
        else:
            from .training import train_model
            phase=args.phase if args.command=='resume' else args.command.removeprefix('train-')
            summary=train_model(config,args.kind,phase,args.device,args.resume or ('auto' if args.command=='resume' else None),args.stop_after)
        print(json.dumps(summary,allow_nan=False));return 0
    except Exception as e:
        print(json.dumps(dict(status='FAILED',error=type(e).__name__,reason=str(e))),file=sys.stderr);return 1


if __name__=='__main__':sys.exit(main())
