"""Run the existing original-observation ODE and streaming canonical DEV scorer."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from starVLA.rl.constrained_flow_noise.contracts import require_dispatch, file_sha, fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json
from .analyze_ode import analyze


def child_environment():
    env=os.environ.copy()
    for key in list(env):
        if key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE','MASTER_ADDR','MASTER_PORT') or key.startswith('TORCHELASTIC_'):env.pop(key)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):env[key]='1'
    env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',NO_ALBUMENTATIONS_UPDATE='1',PYTHONUNBUFFERED='1')
    return env


def run_evaluation(export, population, output, baseline_scores, evaluation_override, cache_root, previous_report=None, workers=16):
    host=require_dispatch();out=Path(output);out.mkdir(parents=True,exist_ok=True)
    export=Path(export);weight=file_sha(export/'pytorch_model.pt')
    pop=json.loads(Path(population).read_text())
    identity=dict(export_sha256=weight,population_identity=pop['identity_sha256'],baseline_scores=str(baseline_scores),
                  evaluation_override_sha256=file_sha(evaluation_override),cache_root=str(cache_root),single_seed=2101,world=8,
                  previous_report_sha256=file_sha(previous_report) if previous_report else None,
                  source_sha256={str(p):file_sha(p) for p in [Path(__file__),Path(__file__).with_name('analyze_ode.py')]})
    if (out/'identity.json').exists() and json.loads((out/'identity.json').read_text())!=identity:raise ValueError('Evaluation resume identity changed')
    atomic_json(out/'identity.json',identity)
    if (out/'report.json').exists():return json.loads((out/'report.json').read_text())
    env=child_environment();processes=[];handles=[];started=time.monotonic()
    def start(args,name):
        handle=(out/(name+'.log')).open('a');handles.append(handle)
        process=subprocess.Popen(args,env=env,stdout=handle,stderr=subprocess.STDOUT);processes.append(process)
        atomic_json(out/(name+'_launch.json'),dict(args=args,pid=process.pid,host=host,started_unix=time.time()))
        return process
    try:
        args=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','-m',
              'scripts.analysis.flow_safe_improvement.sample_original_export','--export',str(export),'--export-sha256',weight,
              '--population',str(population),'--output',str(out/'samples'),'--cohort','dev_tokens','--role','evaluation',
              '--evaluation-override',str(evaluation_override)]
        if (out/'samples/manifest.json').exists():args.append('--resume')
        infer=start(args,'inference')
        while not (out/'samples/manifest.json').exists():
            if infer.poll() is not None:raise RuntimeError('ODE inference exited before publishing its identity')
            time.sleep(2)
        args=[sys.executable,'-m','scripts.analysis.flow_safe_improvement.score_samples','--samples',str(out/'samples'),
              '--population',str(population),'--cache-root',str(cache_root),'--output',str(out/'scores'),'--workers',str(workers)]
        if (out/'scores/manifest.json').exists():args.append('--resume')
        scorer=start(args,'scoring')
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('Original inference or canonical scoring failed; no scene exclusion')
            time.sleep(2)
        if any(p.returncode!=0 for p in processes):raise RuntimeError('Evaluation process failed')
        previous=json.loads(Path(previous_report).read_text()) if previous_report else None
        result=analyze(out/'scores',baseline_scores,pop,previous)
        atomic_json(out/'report.json',result)
        atomic_json(out/'cost.json',dict(wall_seconds=time.monotonic()-started,host=host,
                    status='COMPLETE',scoring=json.loads((out/'scores/status.json').read_text()),
                    inference_ranks=[json.loads(p.read_text()) for p in sorted((out/'samples').glob('rank*.json'))]))
        return result
    except Exception as error:
        atomic_json(out/'failure.json',dict(type=type(error).__name__,message=str(error),no_missing_scene_exclusion=True))
        for process in processes:
            if process.poll() is None:process.terminate()
        for process in processes:
            if process.poll() is None:process.wait()
        raise
    finally:
        for handle in handles:handle.close()


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('export','population','output','baseline-scores','evaluation-override','cache-root'):p.add_argument('--'+key,required=True)
    p.add_argument('--previous-report');p.add_argument('--workers',type=int,default=16)
    a=p.parse_args();result=run_evaluation(**vars(a))
    print(json.dumps(dict(status='COMPLETE',PDMS=100*result['metrics']['score']['current'],scenes=result['scenes'],pause=result['severe_safety_pause_components'])),flush=True)


if __name__=='__main__':main()
