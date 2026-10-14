"""Thin staged research entrypoint; no implicit navtest or deployment."""
import argparse
import json
import traceback
from pathlib import Path
from iqe.io import atomic_json
from iqe.contracts import require
from .config import load_config

COMMANDS=('research-preflight','bind-base','prepare-context-source','audit-historical-subsets','audit-target-opportunities','select-pilot',
          'build-paired-expert-data','train-expert-cell','evaluate-expert-matrix','extend-specialists','freeze-candidate-pool',
          'cache-score-context','build-deployment-bank','build-rich-bank','train-scorer-cell','evaluate-scorer-matrix',
          'calibrate-selection','evaluate-final-system','benchmark-system','run-specialist-scorer-study')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in COMMANDS:
        p=sub.add_parser(name);p.add_argument('--config',required=True)
        if name=='prepare-context-source':p.add_argument('--action',choices=('bind','cache','score'),required=True)
        p.add_argument('--device',default='cuda:0');p.add_argument('--cell')
        p.add_argument('--resume');p.add_argument('--stop-after',type=int)
        p.add_argument('--max-samples',type=int);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=parser.parse_args();c=load_config(a.config)
    try:
        trainers=('train-expert-cell','train-scorer-cell')
        cache_commands=('cache-score-context','prepare-context-source')
        require((a.cell is not None)==(a.command in trainers),'--cell is required only for a training cell')
        require(a.stop_after is None or a.command in trainers,'--stop-after only applies to a training cell')
        require(a.resume is None or a.command in trainers or (a.command=='run-specialist-scorer-study' and a.resume=='auto'),
                'explicit resume belongs to a training cell; workflow supports --resume auto only')
        require(a.max_samples is None or a.command in (*cache_commands,'audit-target-opportunities'),'--max-samples unsupported for this stage')
        require((a.shard==0 and a.shards==1) or a.command in cache_commands,'shard options only apply to context extraction')
        if a.command=='prepare-context-source':
            require(a.action=='cache' or (a.max_samples is None and a.shard==0 and a.shards==1),'source shard/sample limits only apply to cache')
        from .resource_guard import check
        check(c,a.command)
        if a.command=='prepare-context-source':
            from .source_preparation import prepare
            result=prepare(c,a.action,a.device,a.shard,a.shards,a.max_samples)
        elif a.command in ('research-preflight','bind-base'):
            from .binding import bind
            result=bind(c)
        elif a.command=='audit-historical-subsets':
            from .historical import audit
            result=audit(c)
        elif a.command=='audit-target-opportunities':
            from .data import audit_opportunities
            result=audit_opportunities(c,a.max_samples)
        elif a.command=='select-pilot':
            from .data import select_pilot
            result=select_pilot(c)
        elif a.command=='build-paired-expert-data':
            from .data import build_expert_data
            result=build_expert_data(c)
        else:
            from .workflow import dispatch
            result=dispatch(c,a)
        atomic_json(c.root/'command_status'/(a.command+'.json'),{'status':'COMPLETE','result':result})
        summary={k:v for k,v in result.items() if k not in ('record_hashes','artifacts','eligibility','all_endpoints','rows','subsets','numeric_comparison')} if isinstance(result,dict) else result
        print(json.dumps(summary,default=str,ensure_ascii=False))
    except Exception as error:
        atomic_json(c.root/'command_status'/(a.command+'.json'),{'status':'FAILED','error':repr(error),'traceback':traceback.format_exc()})
        raise

if __name__=='__main__':main()
