"""Complete scene population and invocation-cost accounting, without rerunning inference."""
import argparse
import json
from pathlib import Path
import numpy as np
from starVLA.rl.qwen_sourceflow.contracts import fingerprint,file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json
from scripts.analysis.qwen_sourceflow.evaluation_integrity import verify_identity,verify_scene


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--evaluation',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.evaluation);identity=verify_identity(json.loads((root/'identity.json').read_text()))
    rows=[verify_scene(json.loads(p.read_text()),identity) for p in (root/'scenes').glob('*.json')]
    if (len(rows)!=identity['count'] or len({r['token'] for r in rows})!=len(rows) or
        any(r['identity']!=identity['sha256'] or r['status']!='OK' for r in rows)):
        raise ValueError('Incomplete/invalid evaluation population; no missing denominators allowed')
    # Token order is irrelevant, but exact membership is verified below.
    from starVLA.rl.qwen_sourceflow.contracts import POPULATION
    pop=json.loads(POPULATION.read_text());key={'TRAIN':'train_tokens','DEV':'dev_tokens','NAVTEST':'navtest_tokens'}[identity['cohort']]
    expected=pop[key][:identity['count']] if identity['qualification'] else pop[key]
    if set(expected)!={r['token'] for r in rows}:raise ValueError('Wrong evaluation scene membership')
    if (fingerprint(expected)!=identity['token_sha256'] or
            any(r['log']!=pop['token_logs'][r['token']] for r in rows)):
        raise ValueError('Wrong evaluation population identity or scene/log association')
    numeric=rows[0]['feedback']['metrics'].keys()
    means={k:float(np.mean([r['feedback']['metrics'][k] for r in rows])) for k in numeric
        if isinstance(rows[0]['feedback']['metrics'][k],(int,float))}
    inv=[json.loads(p.read_text()) for p in (root/'invocations').glob('*.json')]
    if any(v['status']=='RUNNING' for v in inv):raise ValueError('Evaluation workers still running')
    # Legacy start_unix was written after scorer/processor construction. The
    # completed total monotonic wall includes that work; recover its real start
    # rather than subtracting only model loading from the later timestamp.
    start=min(v['end_unix']-v['wall_seconds'] for v in inv);end=max(v['end_unix'] for v in inv)
    export=identity.get('rl_export')
    result=dict(schema='native_evaluation_summary_v3',status='COMPLETE',
        integrity_analysis_sha256=file_sha(Path(__file__).with_name('evaluation_integrity.py')),
        integrity_checks=dict(identity_seal=True,score_protocol_seal=True,inline_scene_seals=True,
            scene_feedback_token_log_protocol=True,finite_native_8x3_requests=True,scored_request_equals_policy_request=True),
        policy_type='RL_MERGED' if export else 'IL',evaluation=str(root.resolve()),
        evaluation_role='USER_REQUESTED_INTERIM' if identity.get('user_requested_interim_navtest') else 'REGISTERED',
        user_requested_interim_navtest=identity.get('user_requested_interim_navtest'),
        checkpoint=export['checkpoint'] if export else identity['checkpoint'],
        IL_parent_checkpoint=identity['checkpoint'],
        model_sha256=export['model_sha256'] if export else identity['parent_model_sha256'],
        rl_export=export,cohort=identity['cohort'],
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
    if Path(a.output).exists():raise FileExistsError('Keep previous cost/report schema; use a new output')
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    result['sha256']=fingerprint(result);atomic_json(a.output,result);print(json.dumps(result))


if __name__=='__main__':main()
