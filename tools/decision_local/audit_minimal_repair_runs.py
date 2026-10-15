"""Verify matched repair runs from their native identities and exposure logs.

This audit never resumes, changes, or selects a model. Partial runs are reported
as partial; equal current updates are compared, not unequal wall-clock progress.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import read,atomic_json,digest


COMMON=('features','labels','anchor_identity','query_config','recovery_config',
        'constraint_config','epochs','batch','microbatch','lr','dim','weight_decay',
        'scheduler','clip','compute_precision','TF32','trainable_parameters',
        'training_index_sha256','repair_label_semantics','repair_fold')


def audit(registration,require_complete=False):
    rows=[];by_seed={};pending=[];contracts=[]
    for job in registration['jobs']:
        root=Path(job['output'])
        if not (root/'identity.json').exists():pending.append(job['id']);continue
        info=read(root/'identity.json')
        if info['protocol']!='minimal_repair_v1' or info['kind']!=job['kind'] or info['seed']!=job['seed']:
            raise ValueError('Registered arm differs from native run')
        if info['source_sha']!=job['training_source_sha']:raise ValueError('Report/source identity substituted')
        contract={key:info[key] for key in COMMON};contracts.append(contract)
        previous=by_seed.setdefault(info['seed'],info['initial_model_sha256'])
        if previous!=info['initial_model_sha256']:raise ValueError('Unmatched common parameter initialization')
        logs=[]
        if (root/'train.jsonl').exists():
            snapshot=(root/'train.jsonl').read_text()
            lines=snapshot.splitlines()
            # The writer commits a boundary record with its trailing newline;
            # audit one stable snapshot, not a second read racing its append.
            if snapshot and not snapshot.endswith('\n'):lines=lines[:-1]
            logs=[json.loads(line) for line in lines]
        if any(row['update']!=i+1 for i,row in enumerate(logs)):
            raise ValueError('Duplicated or missing optimizer-boundary log')
        complete=read(root/'COMPLETE.json') if (root/'COMPLETE.json').exists() else None
        if complete:
            population=info['repair_fold']['scenes']
            if complete['epoch']!=24 or complete['exposure']!=24*population:
                raise ValueError('Incomplete original-scene exposure at common endpoint')
            counts=complete['exposure_counts']
            if counts['native']!=12*population or counts['lateral']!=6*population or counts['time']!=6*population:
                raise ValueError('Plan-pool mixture differs from registration')
            if counts['KEEP']+counts['REPAIR']!=counts['target_eligible']:
                raise ValueError('Invalid target exposure accounting')
        else:pending.append(job['id'])
        rows.append({'job':job['id'],'seed':info['seed'],'kind':info['kind'],
                     'source_sha':info['source_sha'],'initial_model_sha256':info['initial_model_sha256'],
                     'identity_sha256':digest(root/'identity.json'),'completed':complete,
                     'updates_observed':len(logs),'scope':info['scope'],'logs':logs})
    if contracts and any(c!=contracts[0] for c in contracts):raise ValueError('Scientific or optimization recipe mismatch')
    paired={}
    for seed in sorted(by_seed):
        members=[r for r in rows if r['seed']==seed];common=min(len(r['logs']) for r in members)
        keys=('update','epoch','offset','exposure','lr_next','eligible_this_batch',
              'effective_optimizer_updates','exposure_counts')
        for i in range(common):
            expected={k:members[0]['logs'][i][k] for k in keys}
            if any({k:r['logs'][i][k] for k in keys}!=expected for r in members):
                raise ValueError('Unequal data, masks, scheduler, or exposure at the same update')
        paired[str(seed)]={'arms':[r['kind'] for r in members],
                           'matched_optimizer_boundaries':common,'exact_accounting':True}
    for row in rows:row.pop('logs')
    if require_complete and pending:raise ValueError('Registered common endpoints missing: '+','.join(pending))
    return {'runs':rows,'pending':pending,'common_recipe':contracts[0] if contracts else None,
            'same_seed_initialization_and_exposure':paired,
            'not_checked':'No claim that trained weights match across different targets/information sources',
            'no_training_state_changes':True}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registry',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--require-complete',action='store_true')
    a=p.parse_args();result=audit(read(a.registry),a.require_complete)
    result.update(registry_sha256=digest(a.registry),tool_sha256=digest(__file__))
    atomic_json(a.output,result)
    print(json.dumps({'runs':len(result['runs']),'pending':result['pending'],
                      'paired':result['same_seed_initialization_and_exposure']}))


if __name__=='__main__':main()
