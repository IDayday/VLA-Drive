"""Full-population comparison for an explicit fixed-checkpoint Navtest request.

Keeps this single-seed measurement separate from the final five-run protocol.
Uses the existing scene collapse and log-cluster bootstrap implementation.
"""
import argparse
import json
from pathlib import Path

from tools.foresight.score_pdms import atomic_json, digest, identity_hash
from tools.foresight.summarize_experiments import (
    collapse_sampling_runs, paired_difference, sampling_contract, write_csv,
)
from tools.local_interaction_mask_v2.compare_pdms import read


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--lock',required=True)
    p.add_argument('--score-dirs',required=True,nargs='+')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    lock=json.loads(Path(a.lock).read_text())
    if (lock['schema']!='foresight_navtest_checkpoint_probe_lock_v1' or
            lock['identity']!=identity_hash({k:v for k,v in lock.items() if k!='identity'}) or
            lock['evaluation_purpose']!='user_requested_fixed_checkpoint'):
        raise ValueError('Explicit fixed-checkpoint lock required')
    groups={};exports=[];evaluators=[];sources={};seen=set()
    for directory in a.score_dirs:
        folder=Path(directory);summary=json.loads((folder/'summary.json').read_text())
        export=summary['export_identity'];checkpoint=export['checkpoint'];sha=checkpoint['sha256']
        if (summary['schema']!='foresight_official_pdms_v1' or not summary['valid'] or
                summary['failed'] or not summary['full_navtest'] or summary['diagnostic'] or
                summary['evaluation_purpose']!=lock['evaluation_purpose'] or
                export['protocol']['precision']!='FP32' or
                export['evaluation_source']!=lock['evaluation_source_sha'] or
                checkpoint!=lock['checkpoint_records'].get(sha) or sha in seen or
                export['current_identity']['identity']!=lock['current_data_identity'] or
                export['protocol']['sampling_seed'] not in lock['sampling_seeds']):
            raise ValueError('Score identity, checkpoint, or protocol differs from request')
        rows=read(folder/'scenes.csv')
        if (len(rows)!=12146 or len({r['log'] for r in rows.values()})!=136 or
                summary['scenes']!=12146 or summary['logs']!=136 or
                any(r['status']!='ok' for r in rows.values())):
            raise ValueError('Complete failure-free official Navtest population required')
        seed=export['protocol']['sampling_seed']
        collapsed,group=collapse_sampling_runs({seed:rows},[seed])
        group['checkpoint']=checkpoint;groups[checkpoint['arm']]=(collapsed,group)
        seen.add(sha);exports.append(export);evaluators.append(summary['evaluator_identity'])
        sources[str(folder/'scenes.csv')]=digest(folder/'scenes.csv')
    if (seen!=set(lock['checkpoints']) or any(e!=evaluators[0] for e in evaluators) or
            any(sampling_contract(e)!=sampling_contract(exports[0]) for e in exports)):
        raise ValueError('Missing model or mixed evaluator/current/sampling contracts')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    report={'schema':'ddp_full_foresight_checkpoint_probe_planning_v1','split':'navtest',
            'requested_update':lock['requested_update'],'evaluation_purpose':lock['evaluation_purpose'],
            'final_endpoint_comparison':False,'sampling_seeds':lock['sampling_seeds'],
            'training_seeds':sorted({e['checkpoint']['training_seed'] for e in exports}),
            'valid':True,'lock_identity':lock['identity'],'source_csv_sha256':sources,
            'groups':{},'comparisons':{}}
    arms=sorted(groups)
    for arm,(rows,group) in groups.items():
        report['groups'][arm]=group;write_csv(out/(arm+'_scenes.csv'),rows)
    for i,base in enumerate(arms):
        for first in arms[i+1:]:
            name=first+'-'+base
            result,scenes,logs=paired_difference(groups[first][0],groups[base][0])
            report['comparisons'][name]=result
            write_csv(out/(name+'_scenes.csv'),scenes);write_csv(out/(name+'_logs.csv'),logs)
    atomic_json(out/'summary.json',report)


if __name__=='__main__':main()
