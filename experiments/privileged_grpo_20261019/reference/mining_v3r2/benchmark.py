"""Controlled 1/2-process inference on identical tokens with a ready barrier."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import numpy as np
from common import ROOT, read, write
from pipeline import verify


def trial(world_size, limit=96):
    output=ROOT/f'benchmark_infer{world_size}'
    output.mkdir(exist_ok=False)
    children=[];streams=[];started=time.time()
    try:
        for rank in range(world_size):
            stream=(output/f'worker{rank}.log').open('w');streams.append(stream)
            child=subprocess.Popen(['bash',str(ROOT/'run.sh'),'infer','--limit',str(limit),
                '--prediction-name',output.name,'--rank',str(rank),'--world-size',str(world_size),'--barrier'],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(child)
        while not all((output/f'READY.rank{rank}.json').exists() for rank in range(world_size)):
            if any(child.poll() is not None for child in children):raise RuntimeError('Worker failed before ready')
            if time.time()-started>300:raise TimeoutError('Model load timeout')
            time.sleep(1)
        write(output/'GO.json',dict(started_unix=time.time()))
        while any(child.poll() is None for child in children):
            if time.time()-started>600:raise TimeoutError('Inference benchmark timeout')
            time.sleep(1)
        assert all(child.returncode==0 for child in children)
        summaries=[read(output/f'COMPLETE.rank{rank}.json') for rank in range(world_size)]
        compute_s=max(x['finished_unix'] for x in summaries)-min(x['started_unix'] for x in summaries)
        tokens=[row['token'] for row in read(ROOT/'POPULATION.json')[:limit]]
        files=[p for p in output.glob('????????????????.json') if all(c in '0123456789abcdef' for c in p.stem)]
        assert len(files)==len(tokens)==sum(x['generated'] for x in summaries)
        assert {p.stem for p in files}==set(tokens)
        return dict(processes=world_size,scenes=limit,compute_s=compute_s,
                    total_s=time.time()-started,scenes_per_s=limit/compute_s,output=str(output))
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        for stream in streams:stream.close()


def main():
    _,identity=verify()
    results=[]
    for count in (1,2):
        if count==1:
            baseline=ROOT.parent/'mining_v3'
            previous=read(baseline/'IDENTITY.json')
            assert previous['checkpoint_sha256']==identity['checkpoint_sha256']
            assert previous['source_hashes']==identity['source_hashes']
            for name in ('pipeline.py','common.py'):
                assert previous['pipeline_hashes'][name]==identity['pipeline_hashes'][name]
            folder=baseline/'benchmark_infer1'
            completed=read(folder/'COMPLETE.rank0.json')
            assert completed['completed']==completed['generated']==96
            files=[p for p in folder.glob('????????????????.json') if all(c in '0123456789abcdef' for c in p.stem)]
            assert len(files)==96
            result=dict(processes=1,scenes=96,compute_s=completed['elapsed_s'],total_s=None,
                        scenes_per_s=96/completed['elapsed_s'],output=str(folder),
                        provenance='Preflight baseline reused; inference code/source/checkpoint identical, bookkeeping-only repair')
        else:
            result=trial(count)
        results.append(result)
        write(ROOT/f'BENCHMARK_{count}.json',result)
        print(json.dumps(result),flush=True)
    checks=[]
    for row in read(ROOT/'POPULATION.json')[:96]:
        token=row['token'];a=read(Path(results[0]['output'])/f'{token}.json');b=read(ROOT/'benchmark_infer2'/f'{token}.json')
        delta=float(np.max(np.abs(np.asarray(a['raw'])-b['raw'])))
        assert a['identity']==previous['identity'] and b['identity']==identity['identity'] and a['seed']==b['seed']
        assert delta==0, (token,delta)
        checks.append(dict(token=token,max_abs_raw_difference=delta))
    result=dict(identity=identity['identity'],trials=results,exact_match=True,coverage=96,
                compute_speedup=results[0]['compute_s']/results[1]['compute_s'],checks=checks,
                note='One ordered 1/2-process trial; model startup excluded from compute speedup. CPU scoring not active.')
    write(ROOT/'INFERENCE_BENCHMARK.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)


if __name__=='__main__':main()
