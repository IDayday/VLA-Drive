"""Read immutable repair receipts; count errors separately from target eligibility.

No model, solver, target construction, or historic checkpoint sweep. Canonical
scores are joined only by explicit plan identity and never used to select rows.
"""
import argparse
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.decision_local.common import atomic_json, digest, check_host, source_sha


def case_flags(case):
    c = case['input_check']
    return {
        'population': True,
        'known_road_violation': c['road_violation'],
        'known_annotated_overlap': c['annotated_overlap'],
        'kinematic_violation': not c['kinematic_ok'],
        'any_known_violation': c['known_violation'],
        'full_4s_annotation_support': c['complete_annotations'],
        'uncertain_incomplete_support': case['status_name'] == 'UNCERTAIN' and not c['complete_annotations'],
        'known_violation_filtered_by_incomplete_support': c['known_violation'] and not c['complete_annotations'],
        'known_violation_without_full_target': c['known_violation'] and not case['target_valid'],
        'solver_unsolved': case['status_name'] == 'UNSOLVED',
        'local_intent_or_scope_uncertain': case['status_name'] in ('OUT_OF_SCOPE','TASK_UNCERTAIN','AMBIGUOUS'),
        'map_out_of_range': c['out_of_range'],
        'KEEP': case['status_name'] == 'KEEP',
        'REPAIR': case['status_name'] == 'REPAIR',
        'target_eligible': case['target_valid'],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    host = check_host(); start = time.monotonic(); cpu = time.process_time()
    manifest = json.loads((a.labels/'identity.json').read_text())
    if manifest['constraints'].get('velocity_boundary') != 'measured_vector_v2':
        raise ValueError('The corrected current-velocity contract is required')
    a.output.mkdir(parents=True, exist_ok=True)
    counts = Counter(); scenes = defaultdict(set); logs = defaultdict(set); cross = Counter()
    def load(row):
        rec = json.loads((a.labels/'records'/(row['token']+'.json')).read_text())
        if rec['identity'] != manifest['identity'] or rec['log'] != row['log']:
            raise ValueError('Foreign or changed input label receipt')
        return rec
    rows_path = a.output/'plans.csv.gz'
    with gzip.open(rows_path, 'wt') as out, ThreadPoolExecutor(a.workers) as pool:
        writer = None
        for rec in pool.map(load, manifest['index']):
            for c in rec['cases']:
                group = 'NATURAL' if c['plan_index'] == 0 else 'SYNTHETIC'
                flags = case_flags(c)
                for name, active in flags.items():
                    if active:
                        key=(group,name); counts[key]+=1; scenes[key].add(rec['token']); logs[key].add(rec['log'])
                cross[(group, c['status_name'], str(c['input_check']['known_violation']),
                       str(c['input_check']['complete_annotations']))] += 1
                row={'token':rec['token'],'log':rec['log'],'anchor_index':c['anchor_index'],
                     'plan_index':c['plan_index'],'distribution':group,'status':c['status_name'],
                     'reason':c.get('reason',''),**flags,**c['input_check']}
                if writer is None:
                    writer=csv.DictWriter(out, fieldnames=list(row)); writer.writeheader()
                writer.writerow(row)
    summary={
        'schema':'natural_error_legacy_eligibility_audit_v1', 'source_sha':source_sha(ROOT),
        'source_label_identity':manifest['identity'],'source_identity_file_sha256':digest(a.labels/'identity.json'),
        'index_scenes':len(manifest['index']),'split':manifest['split'],'resource':host,
        'counts':{g:{name:{'plans':n,'unique_scenes':len(scenes[(g,name)]),'unique_logs':len(logs[(g,name)])}
                     for (gg,name),n in counts.items() if gg==g} for g in ('NATURAL','SYNTHETIC')},
        'cross_table':[dict(zip(('distribution','status','known_violation','complete_annotations'),k),plans=v)
                       for k,v in sorted(cross.items())],
        'row_file_sha256':digest(rows_path),'CPU_process_seconds':time.process_time()-cpu,
        'wall_seconds':time.monotonic()-start,
        'canonical':'Separate plan-key join required; no official error is inferred from geometry',
        'object_time_support':'Not in old receipts; added by the bounded local evidence audit',
        'independence':'scene/log counts deduplicate q0 sampling seeds and synthetic variants',
    }
    atomic_json(a.output/'SUMMARY.json',summary)
    print(json.dumps(summary['counts']))


if __name__ == '__main__':
    main()
