"""Disjoint single-PPU inference workers and concurrent CPU scoring/export."""
import concurrent.futures as cf
from collections import Counter
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import time
from common import ROOT, read, write, file_hash, poses_valid
from pipeline import verify, score_row


def gt_reference(record):
    return dict(token=record['token'],log=record['log'],identity=record['identity'],
                source_paths=record['source_paths'],gt_trajectory=record['gt_trajectory'],
                gt_score=record['gt_score'],base_failure=record['guidance']['eligibility']['base_failure'],
                target_quality=record['guidance']['eligibility'])


def main():
    cfg,identity=verify();rows=read(ROOT/'POPULATION.json')
    benchmark=read(ROOT/'INFERENCE_BENCHMARK.json')
    assert benchmark['exact_match'] and benchmark['compute_speedup']>1
    world=cfg['execution']['inference_processes'];workers=cfg['execution']['score_workers']
    deadline=cfg['execution']['wall_deadline_unix']
    assert 1<world<=4 and workers<=4 and time.time()<deadline
    lock=(ROOT/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    predictions=ROOT/'predictions';predictions.mkdir(exist_ok=True)
    output=ROOT/'evaluation';output.mkdir(exist_ok=True)
    run_id=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());logs=ROOT/'execution'/run_id;logs.mkdir(parents=True)
    write(logs/'REQUEST.json',dict(identity=identity['identity'],pid=os.getpid(),world_size=world,
          score_workers=workers,deadline_unix=deadline,started_unix=time.time(),runner_sha256=file_hash(__file__)))
    # Records are authoritative. Rebuild streams after any interrupted append.
    done=set();failures={};reasons=Counter();quality=Counter();metric_sums=Counter();accepted=0
    streams={name:(output/(name+'.tmp')).open('w') for name in ('ALL_GT.jsonl','TRAINING_PAIRS.jsonl','PAIRS.jsonl')}

    def consume(record):
        nonlocal accepted
        token=record['token'];assert token not in done and record['identity']==identity['identity']
        done.add(token);eligibility=record['guidance']['eligibility'];reasons[eligibility['reason']]+=1
        for key,value in record['base_score']['metrics'].items():metric_sums[key]+=value
        for key in ('gt_safety_pass','gt_pdms_improves','gt_verified_repair'):
            quality[key]+=int(eligibility[key])
        streams['ALL_GT.jsonl'].write(json.dumps(gt_reference(record),ensure_ascii=False,allow_nan=False)+'\n')
        if record['training_ready']:
            accepted+=1
            streams['TRAINING_PAIRS.jsonl'].write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
            path=output/'records'/f'{token}.json'
            streams['PAIRS.jsonl'].write(json.dumps(dict(token=token,log=record['log'],record=str(path),
                   record_sha256=file_hash(path),training_ready=True))+'\n')
        for stream in streams.values():stream.flush()

    for row in rows:
        path=output/'records'/f'{row["token"]}.json'
        if path.exists():
            record=read(path)
            assert file_hash(record['prediction_path'])==record['prediction_sha256']
            consume(record)
    for name,stream in streams.items():
        stream.close();os.replace(output/(name+'.tmp'),output/name)
    streams={name:(output/name).open('a',buffering=1) for name in streams}
    write(output/'REQUEST.json',dict(identity=identity['identity'],tokens=[r['token'] for r in rows],
          target_policy='all_gt; pairs selected only by Base failure',historical_smoke=False),replace=True)
    children=[];handles=[];started=time.time();last_report=0.;checked=False;pending={};consecutive_errors=0
    initial_predictions=sum(all(c in '0123456789abcdef' for c in p.stem) for p in predictions.glob('????????????????.json'))
    shards=[rows[rank::world] for rank in range(world)];cursors=[0]*world
    worker_identity={key:identity[key] for key in ('identity','checkpoint_sha256')}
    worker_identity['score_protocol']={'sha256':identity['score_protocol']['sha256']}
    pool=cf.ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'))

    def report(stage):
        summaries=[]
        for rank in range(world):
            path=predictions/f'STATUS.rank{rank}.json'
            if path.exists():summaries.append(read(path))
        prediction_count=sum(all(c in '0123456789abcdef' for c in p.stem) for p in predictions.glob('????????????????.json'))
        summary=dict(identity=identity['identity'],stage=stage,requested=len(rows),predicted=prediction_count,
                     scored=len(done),coverage=len(done)/len(rows),failed_tokens=sorted(failures),
                     training_ready_pairs=accepted,retained_gt=len(done),pair_eligibility_counts=dict(reasons),
                     target_quality_counts=dict(quality),mean_base_metrics={k:v/len(done) for k,v in metric_sums.items()} if done else {},
                     status='COMPLETE' if len(done)==len(rows) and not failures else 'RUNNING' if stage=='running' else 'INCOMPLETE',
                     historical_smoke=False,updated_unix=time.time(),elapsed_s=time.time()-started,
                     note='All scored GT retained; training_ready means valid failure-to-GT supervision, not guaranteed successful/safe repair.')
        write(output/'SUMMARY.json',summary,replace=True)
        write(ROOT/'MONITOR.json',dict(**summary,run_id=run_id,inference_workers=summaries,
              child_pids=[child.pid for child in children],score_workers=workers),replace=True)
        print(json.dumps(summary),flush=True)
        return summary

    try:
        for rank in range(world):
            handle=(logs/f'infer_rank{rank}.log').open('w');handles.append(handle)
            children.append(subprocess.Popen(['bash',str(ROOT/'run.sh'),'infer','--prediction-name','predictions',
                    '--rank',str(rank),'--world-size',str(world)],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True))
        while len(done)+len(failures)<len(rows) or pending:
            if time.time()>deadline:raise TimeoutError('Original task wall-clock deadline reached')
            for child in children:
                if child.poll() is not None and child.returncode!=0:raise RuntimeError(f'Inference PID {child.pid} exited {child.returncode}')
            for rank in range(world):
                while cursors[rank]<len(shards[rank]) and len(pending)<workers*2:
                    row=shards[rank][cursors[rank]];token=row['token']
                    if token in done or token in failures:cursors[rank]+=1;continue
                    if not (predictions/f'{token}.json').exists():break
                    future=pool.submit(score_row,(row,cfg,worker_identity,str(predictions),str(output),False))
                    pending[future]=token;cursors[rank]+=1
            if pending:
                finished,_=cf.wait(pending,timeout=1,return_when=cf.FIRST_COMPLETED)
                for future in finished:
                    token=pending.pop(future);result=future.result()
                    if result['status']=='ok':consume(read(output/'records'/f'{token}.json'));consecutive_errors=0
                    else:
                        failures[token]=result['error'];consecutive_errors+=1
                        if consecutive_errors>=3:raise RuntimeError('Three consecutive scoring errors')
            else:time.sleep(1)
            if all(child.poll() is not None for child in children) and not pending:
                missing=[row['token'] for row in rows if row['token'] not in done and row['token'] not in failures and not (predictions/f'{row["token"]}.json').exists()]
                if missing:raise RuntimeError(f'Inference finished with {len(missing)} missing predictions')
            if time.time()-last_report>=60:
                summary=report('running');last_report=time.time()
                if time.time()-started>=300 and not checked:
                    checked=True
                    assert summary['predicted']>initial_predictions and summary['scored']>0
                    sample=read(next((output/'records').glob('*.json')));poses_valid(sample['input_trajectory']);poses_valid(sample['gt_trajectory'])
                    assert sample['identity']==identity['identity']
                    write(logs/'FIVE_MINUTE_CHECK.json',summary)
        for child in children:
            if child.wait(timeout=120)!=0:raise RuntimeError('Inference exited unsuccessfully')
        rank_complete=[read(predictions/f'COMPLETE.rank{rank}.json') for rank in range(world)]
        assert sum(item['completed'] for item in rank_complete)==len(rows)
        summary=report('complete')
        if failures:raise RuntimeError('Some scoring errors remain; see evaluation/errors')
        write(logs/'COMPLETE.json',summary)
    except BaseException as error:
        summary=report('interrupted')
        write(logs/'ERROR.json',dict(error=repr(error),summary=summary))
        raise
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        pool.shutdown(wait=True,cancel_futures=True)
        for stream in streams.values():stream.close()
        for handle in handles:handle.close()


if __name__=='__main__':main()
