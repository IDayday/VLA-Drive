#!/usr/bin/env python3
"""Read-only full-D_S fit evidence for already completed, selected Scorer cells."""
import argparse
import json

from iqe.specialist_study.config import load_config
from iqe.specialist_study.diagnostics import scorer_fit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--cells',nargs='+',choices=('S00','S01','S10','S11'),required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    config=load_config(args.config)
    for cell in args.cells:
        result=scorer_fit(config,cell,args.device)
        print(json.dumps({'cell':cell,**result}),flush=True)


if __name__=='__main__':
    main()
