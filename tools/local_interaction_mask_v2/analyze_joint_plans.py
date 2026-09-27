"""Keep actual executed DiT and internal graph ego separate in geometry diagnostics."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('tokens','predictions','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();tokens=json.loads(Path(a.tokens).read_text());rows=[];root=Path(a.predictions)
    if len(set(tokens))!=len(tokens):raise ValueError('Duplicate requested scenes')
    for token in tokens:
        row={'token':token,'status':'ok'}
        try:
            with np.load(root/(token+'.npz'),allow_pickle=False) as f:
                actual=f['trajectory'][:,:2]
                if 'graph_joint_xy' not in f:
                    row['has_joint_graph']=False;rows.append(row);continue
                joint=f['graph_joint_xy'];active=f['graph_active_actor_mask'];sources=f['graph_source_slot_ids']
            if not active[0] or sources[0]!=-1 or joint.shape[1:]!=(8,2) or actual.shape!=(8,2):raise ValueError('Invalid exported graph/DiT contract')
            if not np.isfinite(actual).all() or not np.isfinite(joint[active]).all():raise ValueError('Nonfinite active predictions')
            neighbors=joint[1:][active[1:]];gap=np.linalg.norm(joint[0]-actual,axis=-1)
            row.update(has_joint_graph=True,active_neighbors=len(neighbors),graph_ego_vs_DiT_mean_distance_m=float(gap.mean()),graph_ego_vs_DiT_final_distance_m=float(gap[-1]))
            if len(neighbors):
                for name,ego in [('actual_DiT',actual),('internal_graph',joint[0])]:
                    distances=np.linalg.norm(neighbors-ego[None],axis=-1);i,t=np.unravel_index(distances.argmin(),distances.shape)
                    row[name+'_minimum_synchronous_center_distance_m']=float(distances[i,t])
                    row[name+'_time_of_minimum_s']=float((t+1)*.5)
                    row[name+'_center_distance_below_2m']=bool((distances<2.).any())
        except Exception as error:row.update(status='failed',error=repr(error))
        rows.append(row)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);keys=sorted(set().union(*(r.keys() for r in rows)))
    with (out/'scenes.csv').open('w') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
    summary={'requested':len(tokens),'failed':sum(r['status']=='failed' for r in rows),'joint_graph_scenes':sum(r.get('has_joint_graph',False) for r in rows),
        'with_predicted_neighbors':sum(r.get('active_neighbors',0)>0 for r in rows),
        'interpretation':'Center proximity and internal-vs-executed trajectory disagreement are geometric proxies, not true collision probabilities or counterfactual environment responses.'}
    for key in keys:
        if key.endswith(('_m','_s')):
            values=[r[key] for r in rows if key in r and r['status']=='ok'];summary[key+'_mean']=float(np.mean(values)) if values else None
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
