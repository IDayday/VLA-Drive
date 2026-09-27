"""Add role-specific coverage and expected mask supervision to a saved graph audit."""
import argparse
import csv
import json
from pathlib import Path
from tools.local_interaction_mask_v2.audit_graphs import role_coverage
from starVLA.model.modules.joint_world.public_baseline import sha256
from tools.local_interaction_mask_v2.train_foundation import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(); source = Path(args.audit); output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    original = json.loads((source/'summary.json').read_text())
    with (source/'scenes.csv').open() as handle: population = list(csv.DictReader(handle))
    rows = []
    for item in population:
        row = {'token': item['token'], 'status': item['status']}
        if item['status'] == 'ok':
            saved = json.loads((source/'nodes'/(item['token']+'.json')).read_text())
            labels = saved['raw_gt_after_graph_only']; association = saved['association_LABEL_SIDE_ONLY']
            config = saved['graph_provenance']['config']
            row.update(role_coverage(association, saved['nodes'], labels['tracks'], labels['classes'],
                labels['supported'], labels['current_relevance_proxy'], config['predictable_classes']))
            selected = sum(node['group'] == 'A' for node in saved['nodes'])
            supervised = sum(record['used_for_local_loss'] and record['future_valid_points'] > 0
                             for record in association['assignments'])
            row.update(active_neighbors=selected, neighbors_with_any_future=supervised,
                expected_neighbor_task_valid_probability=supervised/selected if selected else None)
        rows.append(row)
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with (output/'scenes.csv').open('w') as handle:
        writer = csv.DictWriter(handle, keys); writer.writeheader(); writer.writerows(rows)
    good = [row for row in rows if row['status'] == 'ok']
    probabilities = [row['expected_neighbor_task_valid_probability'] for row in good if row['active_neighbors']]
    totals = {key: sum(row[key] for row in good) for key in good[0]
              if key not in ('token', 'status', 'expected_neighbor_task_valid_probability')} if good else {}
    atomic_json(output/'summary.json', dict(original_audit_sha256=sha256(source/'summary.json'),
        scope=original['scope'], requested=len(rows), completed=len(good), failed=len(rows)-len(good),
        totals=totals, ego_only_scenes=sum(not row['active_neighbors'] for row in good),
        expected_neighbor_task_valid_probability=sum(probabilities)/len(probabilities) if probabilities else None,
        semantics='Uniform neighbor task on current-eligible nodes; labels are used only to audit expected supervision. No GT-dependent resampling. B context associations are distinct from A trajectory targets. Image retention is not instance recall.',
        source_audit=str(source), formal_algorithm_evidence=original['formal_algorithm_evidence']))


if __name__ == '__main__': main()
