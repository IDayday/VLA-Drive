"""Explicit F-only research: training and checkpoint evaluation run independently.

Both train CLI budgets are validated before this controller is launched. Each
arm has its own evaluation slots and serial evaluation queue; no GPU waiting
script or mutation of a completed checkpoint is used.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
from scripts.cluster_flow_grpo.cluster import run, write_json, exclusive_controller


def orchestrate(spec, train, evaluate, *, pause=time.sleep):
    """Exercise this production controller with injectable execution in CPU tests."""
    root=Path(spec['control_dir']);root.mkdir(parents=True,exist_ok=True)
    cancel=threading.Event()
    state={'status':'RUNNING','training':{},'evaluations':{},'started':time.time()}
    pool=ThreadPoolExecutor(max_workers=2)
    queues={label:ThreadPoolExecutor(max_workers=1) for label in spec['arms']}
    trains={};evaluations={}
    try:
        for label,arm in spec['arms'].items():
            trains[label]=pool.submit(train,arm['train_spec'],cancel)
        while True:
            for label,arm in spec['arms'].items():
                future=trains[label]
                if future.done():
                    result=future.result()
                    if result.get('status')!='PASS':raise RuntimeError('training failed: '+label)
                    state['training'][label]=result
                for update in spec['evaluate_updates']:
                    key=f'{label}/{update}'
                    boundary=Path(arm['run'])/f'checkpoints/update_{update:06d}'
                    if key not in evaluations and (boundary/'COMPLETE').is_file():
                        saved=json.loads((boundary/'trainer_state.json').read_text())
                        if saved['update']!=update or saved['world_size']!=8:
                            raise ValueError('checkpoint boundary identity mismatch')
                        evaluations[key]=queues[label].submit(evaluate,label,arm,update,cancel)
                        state['evaluations'][key]={'status':'QUEUED_OR_RUNNING'}
                    if key in evaluations and evaluations[key].done():
                        result=evaluations[key].result()
                        if result.get('status')!='COMPLETE':raise RuntimeError('evaluation failed: '+key)
                        state['evaluations'][key]=result
                    if future.done() and key not in evaluations:
                        raise RuntimeError('completed training missing evaluation boundary: '+key)
            state['updated']=time.time();write_json(root/'progress.json',state)
            if len(state['training'])==len(trains) and all(f.done() for f in evaluations.values()):break
            pause(5)
        state.update(status='COMPLETE',finished=time.time())
        write_json(root/'result.json',state)
        return state
    except BaseException as exc:
        cancel.set();state.update(status='FAIL',error=repr(exc),finished=time.time())
        write_json(root/'result.json',state)
        raise
    finally:
        cancel.set()
        pool.shutdown(wait=True,cancel_futures=True)
        for queue in queues.values():queue.shutdown(wait=True,cancel_futures=True)


def evaluate_arm(label,arm,update,cancel):
    from starVLA.rl.flow_grpo.checkpoint import export_checkpoint
    from scripts.cluster_flow_grpo.parallel_evaluation import evaluate_parallel
    run_root=Path(arm['run']);boundary=run_root/f'checkpoints/update_{update:06d}'
    exported=run_root/f'export_update{update}'
    export_checkpoint(boundary,exported)
    common=arm['evaluation']
    result=evaluate_parallel(arm['config'],str(exported),str(Path(common['root'])/f'{label}_{update}'),
        'rl_dev',common['tokens'],common['data_root'],common['metric_cache'],42,
        common['slots'],cancel_event=cancel)
    return {k:result[k] for k in ('status','scene_count','epdms','seconds')}


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--spec',required=True)
    args=parser.parse_args();spec=json.loads(Path(args.spec).read_text())
    if spec['evaluate_updates']!=[32,64] or set(spec['arms'])!={'uniform','discount'}:
        raise ValueError('this controller serves only the registered 32/64 F comparison')
    slots=[(s['host'],s['gpu']) for a in spec['arms'].values() for s in a['evaluation']['slots']]
    if len(slots)!=8 or len(set(slots))!=8:raise ValueError('each arm needs four distinct evaluation slots')
    with exclusive_controller(spec['control_dir']):
        print(json.dumps(orchestrate(spec,lambda p,c:run(p,cancel_event=c),evaluate_arm)))


if __name__=='__main__':main()
