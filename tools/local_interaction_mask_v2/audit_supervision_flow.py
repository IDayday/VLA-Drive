"""Separate unavailable annotations from predicted-slot association/filter losses."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess


def audit_scene(data):
    config=data['graph_provenance']['config']
    active=[node for node in data['nodes'] if node['group']=='A']
    assignments=data['association_LABEL_SIDE_ONLY']['assignments']
    by_slot={item['source_slot']:item for item in assignments}
    if len(by_slot)!=len(assignments):raise ValueError('Repeated full-slot assignment')
    counts=Counter(active_predicted_neighbors=len(active),full_slot_assignments=len(assignments),
        full_assignments_with_future=sum(item['future_valid_points']>0 for item in assignments))
    for node in active:
        item=by_slot.get(node['source_slot_id'])
        if item is None:
            counts['active_without_full_slot_assignment']+=1
            continue
        if item['local_slot']!=node['local_slot']:raise ValueError('Source/local mapping differs')
        class_ok=item['same_class'] or not config['match_require_class']
        distance_ok=item['distance_m']<config['match_max_distance_m']
        accepted=class_ok and distance_ok
        if accepted!=item['association_accepted'] or accepted!=item['used_for_local_loss']:
            raise ValueError('Recorded label-side acceptance disagrees with frozen rules')
        if not accepted:
            reason=('distance_and_class' if not class_ok and not distance_ok else
                    'class_only' if not class_ok else 'distance_only')
            counts['active_rejected_'+reason]+=1
            if item['future_valid_points']>0:counts['rejected_assigned_GT_has_future']+=1
        elif item['future_valid_points']==0:
            counts['active_accepted_current_all_future_invalid']+=1
        else:
            counts['active_accepted_with_future']+=1
            counts['accepted_valid_future_points']+=item['future_valid_points']
    keys=('active_without_full_slot_assignment','active_rejected_distance_and_class',
          'active_rejected_class_only','active_rejected_distance_only',
          'active_accepted_current_all_future_invalid','active_accepted_with_future')
    for key in keys:counts.setdefault(key,0)
    if sum(counts[key] for key in keys)!=len(active):raise ValueError('Supervision partition lost nodes')
    if counts['active_accepted_with_future']+counts['active_accepted_current_all_future_invalid']!=data['row']['local_matched']:
        raise ValueError('Accepted current counts differ from original full audit')
    return dict(counts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.audit);out=Path(args.output)
    with (root/'scenes.csv').open() as stream:source=list(csv.DictReader(stream))
    if not source or len({row['token'] for row in source})!=len(source):raise ValueError('Invalid source population')
    rows=[];totals=Counter();digest=hashlib.sha256()
    for original in source:
        row={'token':original['token'],'log':original.get('log',''),'status':'ok'}
        try:
            if original['status']!='ok':raise ValueError('Original graph audit failed')
            raw=(root/'nodes'/(row['token']+'.json')).read_bytes()
            digest.update(row['token'].encode()+b'\0'+hashlib.sha256(raw).digest())
            counts=audit_scene(json.loads(raw));row.update(counts);totals.update(counts)
        except Exception as error:row.update(status='failed',error=repr(error))
        rows.append(row)
    failed=sum(row['status']!='ok' for row in rows)
    accepted=totals['active_accepted_with_future']+totals['active_accepted_current_all_future_invalid']
    summary={'scenes':len(rows),'failed':failed,'valid':failed==0,'totals':dict(totals),
        'future_available_given_accepted_current':totals['active_accepted_with_future']/accepted if accepted else None,
        'source_audit_summary_sha256':hashlib.sha256((root/'summary.json').read_bytes()).hexdigest(),
        'source_node_records_digest':digest.hexdigest(),
        'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'units':'scene-actor occurrences, not unique physical actors across scenes',
        'interpretation':'Unassigned/rejected predicted slots are NOT missing dataset annotations. Rejected assigned GT may have complete futures. Zero valid future after an accepted association is counted separately.',
        'scope':'Read-only diagnosis of the fixed deployed graph and existing label-side assignment; no threshold, matching, model, task mask or training change.'}
    out.mkdir(parents=True,exist_ok=False)
    with (out/'scenes.csv').open('w') as stream:
        writer=csv.DictWriter(stream,sorted(set().union(*(row.keys() for row in rows))))
        writer.writeheader();writer.writerows(rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    if failed:raise RuntimeError('Supervision audit failures retained; report is incomplete')


if __name__=='__main__':main()
