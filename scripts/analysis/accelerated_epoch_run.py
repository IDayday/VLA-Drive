"""Accelerated training while reusing the already-running original SFT evaluation.

The old controller retains ownership of its six evaluation GPUs. Only completed,
identity-bound five-seed results can be consumed here. No duplicate baseline jobs.
"""
import argparse
import json
from pathlib import Path
import time
from scripts.analysis.full_epoch_run import train_epoch,evaluate_policy,orchestrate,summarize
from scripts.cluster_flow_grpo.cluster import exclusive_controller
from starVLA.rl.flow_grpo.loading import file_sha


def await_baseline(control, cancel, *, timeout=259200, poll=15):
    deadline=time.monotonic()+timeout
    while not cancel.is_set():
        path=Path(control)/'progress.json'
        state=json.loads(path.read_text()) if path.exists() else {}
        result=state.get('evaluations',{}).get('sft')
        if result is not None:
            if result.get('status')!='COMPLETE':raise ValueError('external baseline is incomplete')
            return result
        failure=state.get('baseline_error') or state.get('failures',{}).get('sft_evaluation')
        if failure:raise RuntimeError('external baseline failed: '+str(failure))
        if time.monotonic()>=deadline:raise TimeoutError('active baseline producer exceeded bounded deadline')
        cancel.wait(poll)
    raise RuntimeError('baseline wait cancelled')


def validate_baseline(result, binding, spec):
    from starVLA.rl.flow_grpo.evaluation_transaction import completed_evaluation
    from starVLA.rl.flow_grpo.config import resolve_config
    from starVLA.rl.flow_grpo.contracts import digest
    from omegaconf import OmegaConf
    import pandas as pd
    raw=Path(binding['producer_spec']['path']).read_bytes()
    if file_sha(binding['producer_spec']['path'])!=binding['producer_spec']['sha256']:
        raise ValueError('external baseline producer spec changed')
    producer=json.loads(raw)
    if producer['baseline_checkpoint']!=spec['baseline_checkpoint']:
        raise ValueError('external baseline checkpoint differs')
    common=spec['evaluation'];old=producer['evaluation']
    for key in ('tokens','data_root','metric_cache','raw_logs','maps','v1_cache'):
        if old[key]!=common[key]:raise ValueError('external baseline inputs differ: '+key)
    cfg,sft=resolve_config(spec['config'])
    if digest(OmegaConf.to_container(sft,resolve=True))!=binding['identity']['resolved_data_model_config_sha256']:
        raise ValueError('baseline data/model configuration differs')
    for name,sha in binding['inference_sources'].items():
        if file_sha(name)!=sha:raise ValueError('baseline inference implementation changed: '+name)
    tokens=json.loads(Path(common['tokens']).read_text())
    if len(tokens)!=12146 or result.get('scenes_per_seed')!=12146 or set(result.get('seeds',{}))!={'42','43','44','45','46'}:
        raise ValueError('full five-seed official baseline required')
    for seed,row in result['seeds'].items():
        identity=dict(binding['identity'],seed=int(seed))
        report=completed_evaluation(row['v2'],identity,tokens)
        if report is None:raise ValueError('baseline v2 transaction incomplete')
        v1=Path(row['v1']);seal=json.loads((v1/'COMPLETE').read_text())
        if seal['summary_sha256']!=file_sha(v1/'summary.json') or seal['identity_sha256']!=file_sha(v1/'identity.json'):
            raise ValueError('baseline v1 transaction changed')
        summary=json.loads((v1/'summary.json').read_text())['results']['sft']
        if summary['csv_sha256']!=file_sha(v1/'sft.csv'):raise ValueError('baseline v1 scores changed')
        frame=pd.read_csv(v1/'sft.csv',dtype={'token':str})
        if len(frame)!=len(tokens) or frame.token.duplicated().any() or set(frame.token)!=set(tokens):
            raise ValueError('baseline v1 token set differs')
        if row['epdms']!=report['epdms'] or row['pdms']!=summary['means']['score']:
            raise ValueError('baseline summary differs')
    return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--spec',required=True);a=p.parse_args()
    spec=json.loads(Path(a.spec).read_text())
    if spec['controller_sha256']!=file_sha(__file__):raise ValueError('accelerated controller changed')
    binding=spec['external_baseline']
    def evaluate(label,checkpoint,cancel):
        if label=='sft':return validate_baseline(await_baseline(binding['control_dir'],cancel),binding,spec)
        return evaluate_policy(spec,label,checkpoint,cancel)
    with exclusive_controller(spec['control_dir']):
        result=orchestrate(spec,lambda c:train_epoch(spec,c),evaluate)
        summarize(spec,result)

if __name__=='__main__':main()
