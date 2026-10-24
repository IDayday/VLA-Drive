"""Publish only a complete, disjoint set of independent single-output shards."""
import argparse
import json
import os
from pathlib import Path
import shutil
import uuid
from starVLA.rl.qwen_flow_exploration.protocol import read_protocol
from starVLA.rl.qwen_flow_exploration.common_analysis import checked_seal,load_evaluation,validate_root_record
from starVLA.rl.qwen_flow_exploration.evaluation import partition
from scripts.analysis.qwen_flow_exploration.preservation_evaluate import verified_row,METRICS
from starVLA.rl.qwen_sourceflow.contracts import fingerprint,file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def full_population(identity,protocol):
    split=identity['split'];count=identity['scenes'];orders=[protocol['splits'][split]]
    if split in protocol['diagnostic']:orders.append(protocol['diagnostic'][split])
    if identity['qualification']:orders=[order[:count] for order in orders]
    matches=[order for order in orders if len(order)==count and fingerprint(order)==identity['tokens_sha256']]
    if not matches:raise ValueError('Unregistered full scene population')
    return matches[0]


def aggregate(protocol,paths,output):
    output=Path(output)
    if output.exists():raise FileExistsError('Preserve already published aggregate/analysis')
    identity=None;tokens=None;ranks={};rows={};invocations=[];ledger_totals={};shard_receipts=[]
    for folder in map(Path,paths):
        current=checked_seal(json.loads((folder/'identity.json').read_text()),'shard identity')
        complete=checked_seal(json.loads((folder/'SHARD_COMPLETE.json').read_text()),'completed independent shard')
        execution=json.loads((folder/'execution.json').read_text())
        if complete['status']!='COMPLETE' or complete['execution']!=execution or execution['identity_sha256']!=current['sha256']:
            raise ValueError('A shard is incomplete or belongs to a different execution')
        if identity is None:identity=current;tokens=full_population(identity,protocol)
        if current!=identity:raise ValueError('Shards have different model/scientific identities')
        rank,world=execution['rank'],execution['world'];local=partition(tokens,rank,world)
        if rank in ranks or (ranks and world!=next(iter(ranks.values()))):raise ValueError('Duplicate rank or mixed partition world')
        if execution['local_tokens_sha256']!=fingerprint(local):raise ValueError('Changed shard membership')
        ranks[rank]=world;expected={f'{t}_seed{s}.json' for t in local for s in identity['evaluation_seeds']}
        if {p.name for p in (folder/'scenes').glob('*.json')}!=expected:raise ValueError('Missing/extra local records')
        for token in local:
            for seed in identity['evaluation_seeds']:
                path=folder/'scenes'/f'{token}_seed{seed}.json';key=(token,seed)
                if key in rows:raise ValueError('Duplicate scene/seed output')
                row=verified_row(path,identity,token,seed);validate_root_record(row,identity)
                rows[key]=(path.resolve(),row)
        shard_receipts.append(dict(path=str(folder.resolve()),seal_sha256=file_sha(folder/'SHARD_COMPLETE.json')))
        for p in (folder/'invocations').glob('*/invocation.json'):
            invocation=json.loads(p.read_text())
            if invocation['status'] in ('LOADING','RUNNING'):raise ValueError('A worker invocation remains unreviewed/running')
            invocations.append(invocation)
            ledger=p.parent/'ledger.json'
            if ledger.exists():
                for name,value in json.loads(ledger.read_text()).items():
                    if isinstance(value,(int,float)) and not isinstance(value,bool):ledger_totals[name]=ledger_totals.get(name,0)+value
    if identity is None:raise ValueError('No completed shards')
    world=next(iter(ranks.values()))
    if set(ranks)!=set(range(world)) or len(rows)!=len(tokens)*len(identity['evaluation_seeds']):
        raise ValueError('Incomplete global rank/scene/seed population')
    means={name:sum(sum(rows[(t,s)][1]['native_subscores'][name] for s in identity['evaluation_seeds'])/len(identity['evaluation_seeds']) for t in tokens)/len(tokens) for name in METRICS}
    result=dict(status='COMPLETE',identity_sha256=identity['sha256'],scenes=len(tokens),seed_count=len(identity['evaluation_seeds']),
        logical_single_output_evaluations=len(rows),PDMS=100*means['score'],means=means,
        physical_inference_attempts_including_retries=sum(v['physical_inference_attempts'] for v in invocations),
        completed_physical_inferences=sum(v['completed_inferences'] for v in invocations),
        new_scored_rows_over_invocations=sum(v['new'] for v in invocations),reused_rows_over_invocations=sum(v['reused'] for v in invocations),
        cumulative_worker_wall_seconds=sum(v.get('wall_seconds',0) for v in invocations),
        model_loading_and_initialization_worker_seconds=sum(v.get('model_loading_seconds',0) for v in invocations),
        job_elapsed_span_seconds=(max(v['end_unix'] for v in invocations)-min(v['start_unix'] for v in invocations)) if invocations else None,
        worker_invocations=len(invocations),completed_shards=world,physical_ledger_totals=ledger_totals,
        cost_semantics='Independent worker occupancy may share physical GPUs; sum is not exclusive GPU time or end-to-end elapsed span',
        qualification=identity['qualification'],shards=shard_receipts)
    attempt=output.with_name(output.name+'.incomplete.'+uuid.uuid4().hex);(attempt/'scenes').mkdir(parents=True)
    atomic_json(attempt/'identity.json',identity)
    for (token,seed),(path,_) in rows.items():(attempt/'scenes'/f'{token}_seed{seed}.json').symlink_to(path)
    atomic_json(attempt/'COMPLETE.json',result)
    # The usual full-population verifier is the final publication gate.
    load_evaluation(attempt,protocol,allow_qualification=identity['qualification'])
    os.rename(attempt,output)
    return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--protocol',required=True)
    p.add_argument('--shard',action='append',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=aggregate(read_protocol(a.protocol),a.shard,a.output)
    print({key:result[key] for key in ['status','scenes','PDMS','completed_shards']})


if __name__=='__main__':main()
