"""Offline ego ADE/FDE after current-only export; no model is loaded in this process."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from starVLA.dataloader.foresight_dataset import decode_ego
from starVLA.model.modules.vehicle_joint.initialization import file_sha256,identity_hash
from tools.ddpolicy_vehicle.evaluate_ego import trajectory_errors,relative_ego_target
from tools.ddpolicy_vehicle.prepare_data import load_trusted
from .score_pdms import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('predictions','current-root','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--processed-root',help='For Navtest labels only; never read by camera export')
    a=p.parse_args();bank=Path(a.predictions);root=Path(a.current_root);out=Path(a.output)
    export=json.loads((bank/'identity.json').read_text());current=json.loads((root/'identity.json').read_text())
    index=json.loads((root/'index.json').read_text())
    if current!=export['current_identity'] or identity_hash(index)!=current['index_sha256']:raise ValueError('Prediction/current population differs')
    index=index[:export['limit']] if export['limit'] else index
    if not index or len({r['token'] for r in index})!=len(index):raise ValueError('Empty/duplicate evaluation population')
    if current['split']=='navtest' and (export['checkpoint']['scope']!='formal' or export['limit']):raise ValueError('No diagnostic Navtest evaluation')
    if a.processed_root:
        labels={'kind':'processed GT ego poses','root':a.processed_root}
    else:
        labels=json.loads((root/'ego_identity.json').read_text())
        if labels['schema']!='foresight_ego_labels_v1' or labels['index_sha256']!=current['index_sha256']:raise ValueError('Ego label population mismatch')
    signature=identity_hash(export);rows=[]
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'identity.json',{'schema':'foresight_ego_fit_v1','export_identity':export,'label_identity':labels,
                                  'metric_code_sha256':file_sha256(__file__),'scope':'offline trajectory fit, not PDMS'})
    for scene in index:
        row={'token':scene['token'],'log':scene['log'],'failure':None}
        try:
            token=scene['token']
            if a.processed_root:
                path=Path(a.processed_root)/(token+'.pkl');raw=load_trusted(path)
                target=relative_ego_target(np.asarray(raw['glo_status']['global_poses'])[:12])
            else:
                path=root/'ego'/(token+'.pt');label=torch.load(path,map_location='cpu',weights_only=True)
                if label['identity']!=labels['identity'] or label['token']!=token:raise ValueError('Ego label identity differs')
                target=decode_ego(label['ego']).numpy()
            row['label_sha256']=file_sha256(path)
            row.update({'stationary_'+k:v for k,v in trajectory_errors(np.zeros((8,3)),target).items()})
            row['motion_group']='stationary' if np.linalg.norm(target[:,:2],axis=-1).max()<=.5 else 'moving'
            meta=json.loads((bank/'predictions'/(token+'.json')).read_text())
            if meta['identity_sha256']!=signature or meta['token']!=token or meta['log']!=scene['log']:raise ValueError('Prediction identity differs')
            if meta['status']!='ok':raise ValueError(meta.get('error','Failed prediction'))
            prediction=bank/'predictions'/(token+'.npz')
            if file_sha256(prediction)!=meta['proposal_sha256']:raise ValueError('Prediction content differs')
            with np.load(prediction,allow_pickle=False) as sample:row.update(trajectory_errors(sample['trajectory'],target))
        except Exception as error:row['failure']=repr(error)
        rows.append(row)
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with (out/'scenes.csv').open('w') as stream:
        writer=csv.DictWriter(stream,keys);writer.writeheader();writer.writerows(rows)
    summary={'scenes':len(rows),'logs':len({r['log'] for r in rows}),'failed':sum(bool(r['failure']) for r in rows),
             'scope':'ego-only label-side trajectory fit, not PDMS','groups':{}}
    for group in ('all','stationary','moving'):
        part=rows if group=='all' else [r for r in rows if r.get('motion_group')==group]
        failed=sum(bool(r['failure']) for r in part)
        summary['groups'][group]={'scenes':len(part),'failed':failed,'valid':bool(part) and failed==0}
        for key in ('ADE','FDE','yaw_MAE_rad','yaw_endpoint_rad','stationary_ADE','stationary_FDE'):
            # A failed scene invalidates its group; do not report a filtered mean.
            summary['groups'][group][key]=float(np.mean([r[key] for r in part])) if part and not failed else None
    summary['valid']=not summary['failed'];atomic_json(out/'summary.json',summary)
    if not summary['valid']:raise RuntimeError('Failed rows retained; trajectory-fit result invalid')


if __name__=='__main__':main()
