"""Complete scene population and invocation-cost accounting, without rerunning inference."""
import argparse
import json
from pathlib import Path
import numpy as np
from starVLA.rl.qwen_sourceflow.contracts import fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--evaluation',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.evaluation);identity=json.loads((root/'identity.json').read_text())
    rows=[json.loads(p.read_text()) for p in (root/'scenes').glob('*.json')]
    if (len(rows)!=identity['count'] or len({r['token'] for r in rows})!=len(rows) or
        any(r['identity']!=identity['sha256'] or r['status']!='OK' for r in rows)):
        raise ValueError('Incomplete/invalid evaluation population; no missing denominators allowed')
    # Token order is irrelevant, but exact membership is verified below.
    from starVLA.rl.qwen_sourceflow.contracts import POPULATION
    pop=json.loads(POPULATION.read_text());key={'TRAIN':'train_tokens','DEV':'dev_tokens','NAVTEST':'navtest_tokens'}[identity['cohort']]
    expected=pop[key][:identity['count']] if identity['qualification'] else pop[key]
    if set(expected)!={r['token'] for r in rows}:raise ValueError('Wrong evaluation scene membership')
    numeric=rows[0]['feedback']['metrics'].keys()
    means={k:float(np.mean([r['feedback']['metrics'][k] for r in rows])) for k in numeric
        if isinstance(rows[0]['feedback']['metrics'][k],(int,float))}
    inv=[json.loads(p.read_text()) for p in (root/'invocations').glob('*.json')]
    if any(v['status']=='RUNNING' for v in inv):raise ValueError('Evaluation workers still running')
    start=min(v['start_unix']-v['model_loading_seconds'] for v in inv);end=max(v['end_unix'] for v in inv)
    result=dict(status='COMPLETE',checkpoint=identity['checkpoint'],cohort=identity['cohort'],
        seed=identity['seed'],metric=identity['metric'],inference=identity['inference'],
        qualification=identity['qualification'],full_population=identity['full_population'],
        scenes=len(rows),logs=len({r['log'] for r in rows}),means=means,PDMS=100*means['score'],
        source_identity=identity['sha256'],logical_single_candidates=len(rows),
        physical_inference_including_retries=sum(v['physical_inference'] for v in inv),
        reused_scenes=sum(v['reused'] for v in inv),model_loading_seconds=sum(v['model_loading_seconds'] for v in inv),
        worker_wall_seconds=sum(v['wall_seconds'] for v in inv),elapsed_span_seconds=end-start,
        forward_worker_seconds=sum(v['forward_seconds'] for v in inv),
        scoring_wait_worker_seconds=sum(v['scoring_wait_seconds'] for v in inv),
        GPU_cost_semantics='Worker occupancy on authorized GPUs, possibly shared; not exclusive GPU compute seconds')
    result['sha256']=fingerprint(result);atomic_json(a.output,result);print(json.dumps(result))


if __name__=='__main__':main()
