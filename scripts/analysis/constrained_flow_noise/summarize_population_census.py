"""Aggregate only sealed census records; never label partial coverage as full TRAIN."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

from starVLA.rl.constrained_flow_noise.contracts import require_dispatch, file_sha, fingerprint
from starVLA.rl.constrained_flow_noise.runtime import OrderedScenes
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    require_dispatch()
    root=Path(args.root)
    registration=json.loads((root/'registration.json').read_text())
    pop=json.loads(Path(registration['population']).read_text())
    stream=OrderedScenes(pop['train_tokens'],42)
    expected=registration['population_count']
    seen,files,logs={}, {},Counter()
    counts,ties=Counter(),Counter()
    means,seconds=defaultdict(float),defaultdict(float)
    source=file_sha(__file__)
    while True:
        require_dispatch()
        for path in sorted((root/'scenes').glob('*/*.json')):
            if str(path) in files:
                continue
            row=json.loads(path.read_text());position=row['position']
            if (row['registration_sha256']!=registration['sha256'] or position in seen
                    or not 0<=position<expected or row['token']!=stream.at(position)
                    or pop['token_logs'][row['token']]!=row['log']):
                raise ValueError('Census identity/population/duplicate mismatch')
            summary=row['summary']
            if summary['scenes']!=1 or summary['candidates']!=16:
                raise ValueError('Census must retain every G16 group')
            seen[position]=row;files[str(path)]=file_sha(path);logs[row['log']]+=1
            for key in ('zero_PDMS_advantage_groups','all_zero_PDMS_groups','zero_final_advantage_groups',
                        'progress_only_official_reward_groups','official_safety_cost_varies_groups',
                        'geometric_cost_varies_groups','groups_with_safe_candidate'):
                counts[key]+=summary[key]
            counts['all_PDMS_one']+=int(summary['all_PDMS_one'])
            if summary['tied_PDMS_value'] is not None:
                ties[repr(summary['tied_PDMS_value'])]+=1
            for key,value in summary['official_means'].items():
                means[key]+=value
            means['joint_safe_candidate_rate']+=summary['joint_safe_candidate_rate']
            for key,value in summary['seconds'].items():
                seconds[key]+=value
        complete=len(seen)==expected
        counts['other_nonzero_nonunit_tied_groups']=counts['zero_PDMS_advantage_groups']-counts['all_PDMS_one']-counts['all_zero_PDMS_groups']
        counts['mixed_PDMS_groups']=len(seen)-counts['zero_PDMS_advantage_groups']
        report=dict(schema='cfn_frozen_population_census_summary_v1',status='VERIFYING_FULL_POPULATION' if complete else 'IN_PROGRESS',
                    registered_scenes=expected,completed_scenes=len(seen),remaining_scenes=expected-len(seen),
                    observed_logs=len(logs),registered_logs=len({pop['token_logs'][t] for t in pop['train_tokens']}),
                    counts=dict(counts),tied_PDMS_value_histogram=dict(ties),
                    candidate_means={k:v/len(seen) for k,v in means.items()} if seen else {},
                    committed_logical_candidate_scores=16*len(seen),measured_worker_seconds=dict(seconds),
                    reference='fixed original SFT plus original C_SDE G16; no optimizer updates',
                    not_single_candidate_ODE_deployment=True,
                    final_advantage_semantics='fixed initial multipliers only; current training lambda is not substituted',
                    population_scope='full registered RL_TRAIN' if complete else 'incomplete observed subset; not full-TRAIN statistics',
                    registration_sha256=registration['sha256'],source_sha256=source,observed_unix=time.time(),
                    seal_manifest_sha256=fingerprint(files),
                    retry_cost_scope='committed unique groups; uncommitted failed attempts and qualification calls must be added from launch/failure receipts')
        if complete:
            for name,digest in files.items():
                path=Path(name)
                if file_sha(path)!=digest or file_sha(path.with_suffix('.pt'))!=seen[int(path.stem)]['payload_sha256']:
                    raise ValueError('Census record changed during final verification')
            report['status']='COMPLETE_ALL_TRAIN_G16'
            report['sha256']=fingerprint(report)
            if (root/'full_summary.json').exists():
                raise FileExistsError('Final census summary is immutable')
            atomic_json(root/'full_summary.json',report)
            (root/'COMPLETE').write_text('Every registered TRAIN scene, all G16 candidates, sealed and verified\n')
        else:
            report['sha256']=fingerprint(report)
        atomic_json(root/'progress_summary.json',report)
        if complete or not args.wait:
            print(json.dumps(dict(status=report['status'],completed_scenes=len(seen))),flush=True)
            break
        time.sleep(30)


if __name__=='__main__':
    main()
