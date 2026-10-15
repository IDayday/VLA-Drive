"""Audit legal raw logs and freeze recording-separated roles before outcomes.

The old fold files are read, never overwritten. Fragment names share their
parent recording; adjacent segments cannot become an unseen base-policy fold.
Raw files outside the locked navtrain list are not silently promoted to train.
"""
import argparse
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json, digest, identity, check_host, source_sha


def recording(log):
    name, begin, end = log.rsplit('_', 2)
    if not begin.isdigit() or not end.isdigit() or int(begin)>=int(end):
        raise ValueError('Unrecognized fragment identity: '+log)
    return name


def split_roles(rows, dev_rows):
    """Outcome-free recording split, with A-only calibration and P/R separation."""
    dev_recordings={recording(r['log']) for r in dev_rows}
    groups=defaultdict(list)
    for row in rows: groups[recording(row['log'])].append(row)
    ordered=sorted(set(groups)-dev_recordings,
                   key=lambda s:hashlib.sha256(('natural-error-recording-v1:'+s).encode()).hexdigest())
    # Alternation is at the WHOLE original recording, not individual fragments.
    A=ordered[::2]; B=ordered[1::2]
    calibration=set(A[::20])
    role_by_recording={g:('A_calibration' if g in calibration else 'A_train') for g in A}
    role_by_recording.update({g:('B_predictor' if i%2==0 else 'B_refiner') for i,g in enumerate(B)})
    role_by_recording.update({g:'excluded_dev_recording' for g in dev_recordings})
    return role_by_recording, {role:[r for r in rows if role_by_recording[recording(r['log'])]==role]
                              for role in sorted(set(role_by_recording.values()))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--anchor',type=Path,required=True);p.add_argument('--old-folds',type=Path,required=True)
    p.add_argument('--navtrain-filter',type=Path,required=True);p.add_argument('--raw-logs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();check_host();start=time.monotonic();cpu=time.process_time()
    import yaml
    anchor=json.loads(a.anchor.read_text());data=anchor['data']
    rows=json.loads((Path(data['train']['current_root'])/'index.json').read_text())
    dev=json.loads((Path(data['dev']['current_root'])/'index.json').read_text())
    official=set(yaml.safe_load(a.navtrain_filter.read_text())['log_names'])
    raw={f.stem for f in a.raw_logs.glob('*.pkl')};trained={r['log'] for r in rows};devlogs={r['log'] for r in dev}
    unused=raw&official-trained-devlogs
    roles,indices=split_roles(rows,dev)
    old=json.loads(a.old_folds.read_text());oldgroups=defaultdict(set)
    for log,fold in old['log_fold'].items(): oldgroups[recording(log)].add(fold)
    source=json.loads((Path(anchor['run'])/'identity.json').read_text())
    sources={k:source.get(k) for k in ('data','ego','dino','future_clip','interaction','source_sha','selected_index_hash')}
    report={'schema':'natural_error_base_source_audit_v1','source_sha':source_sha(ROOT),
        'anchor_identity':anchor['identity'],'old_training_identity':source['identity'],
        'official_navtrain_filter_sha256':digest(a.navtrain_filter),'old_fold_sha256':digest(a.old_folds),
        'legal_navtrain_logs':len(official),'raw_trainval_logs':len(raw),
        'S0_training_logs':len(trained),'fixed_development_logs':len(devlogs),
        'legal_available_unseen_log_names':sorted(unused),
        'outside_locked_navtrain':sorted(raw-official),
        'unseen_decision':'A-only shared base necessary' if not unused else 'Audit unused log teacher/tuning/image provenance before use',
        'old_fold_recordings':len(oldgroups),'old_fold_shared_recordings':sum(len(x)>1 for x in oldgroups.values()),
        'fitted_and_generic_source_receipts':sources,
        'GT_MAE_reuse_for_new_base':'FORBIDDEN: original teacher fitted full training population; refit only A_train if retained',
        'generic_DINO_cache':'Deterministic frozen generic features may be read only at A_train tokens; no B gradient/statistic',
        'role_rule':'SHA256(natural-error-recording-v1:parent recording) sort; alternating A/B; every twentieth A recording calibration; B alternates predictor/refiner; exclude all dev parent recordings',
        'roles':roles,
        'role_populations':{k:{'scenes':len(v),'logs':len({r['log'] for r in v}),
                               'recordings':len({recording(r['log']) for r in v})} for k,v in indices.items()},
        'no_results_or_scores_used_to_partition':True,
        'pending_checks':['raw timestamp/image identity duplication across newly assigned roles','A-only teacher training/cache identity','new base profile and budget'],
        'CPU_process_seconds':time.process_time()-cpu,'wall_seconds':time.monotonic()-start}
    a.output.mkdir(parents=True,exist_ok=True)
    for role,index in indices.items():atomic_json(a.output/(role+'.json'),index)
    report['index_sha256']={k:digest(a.output/(k+'.json')) for k in indices}
    report['identity']=identity(report)
    atomic_json(a.output/'SOURCE_AUDIT.json',report)
    print(json.dumps({k:report[k] for k in ('unseen_decision','legal_available_unseen_log_names','old_fold_shared_recordings','role_populations','identity')}))


if __name__=='__main__':main()
