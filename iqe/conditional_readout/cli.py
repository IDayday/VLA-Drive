import argparse
import json
import sys
from .config import load


def main(argv=None):
    parser=argparse.ArgumentParser(description='One bounded candidate-conditioned visual-readout follow-up')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('bind','diagnostic','train'):
        p=sub.add_parser(name);p.add_argument('--config',required=True)
        if name!='bind':
            p.add_argument('--device',default='cuda:0')
            p.add_argument('--resume',nargs='?',const='auto')
            p.add_argument('--stop-after',type=int)
        if name=='diagnostic':p.add_argument('--mode',required=True,choices=('global','conditioned'))
    args=parser.parse_args(argv)
    try:
        from iqe.specialist_study.resource_guard import check
        from .experiment import bind,train_cell
        config=load(args.config);check(config,'train-scorer-cell' if args.command!='bind' else 'research-preflight')
        result=bind(config) if args.command=='bind' else train_cell(config,args.device,args.resume,args.stop_after,
                    args.mode if args.command=='diagnostic' else None)
        print(json.dumps(dict(status='OK',command=args.command,result=result),allow_nan=False))
    except Exception as error:
        print(json.dumps(dict(status='FAIL',error_type=type(error).__name__,error=str(error))),file=sys.stderr)
        raise


if __name__=='__main__':main()
