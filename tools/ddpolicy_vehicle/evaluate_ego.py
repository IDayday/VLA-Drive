"""Offline trajectory-fit diagnostics; these errors are NOT official PDMS.

Prediction files are already frozen. Future poses are read only on this label
side, and never enter camera export, graph selection, or checkpoint loading.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .prepare_data import atomic_json, load_trusted
from starVLA.model.modules.vehicle_joint.initialization import file_sha256


def relative_ego_target(global_poses):
    poses = np.asarray(global_poses, dtype=np.float64)
    if poses.shape != (12, 3) or not np.isfinite(poses).all():
        raise ValueError("Expected four observed and eight valid future ego poses")
    current = poses[3]
    c, s = np.cos(current[2]), np.sin(current[2])
    xy = (poses[4:, :2]-current[:2]) @ np.array([[c, -s], [s, c]])
    yaw = (poses[4:, 2]-current[2]+np.pi) % (2*np.pi)-np.pi
    return np.column_stack((xy, yaw))


def trajectory_errors(predicted, target):
    predicted, target = np.asarray(predicted), np.asarray(target)
    if predicted.shape != (8, 3) or target.shape != (8, 3):
        raise ValueError("Ego trajectory must contain eight xy/yaw points")
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite ego trajectory")
    distance = np.linalg.norm(predicted[:, :2]-target[:, :2], axis=-1)
    yaw = np.abs((predicted[:, 2]-target[:, 2]+np.pi) % (2*np.pi)-np.pi)
    return {"ADE": float(distance.mean()), "FDE": float(distance[-1]),
            "yaw_MAE_rad": float(yaw.mean()), "yaw_endpoint_rad": float(yaw[-1])}


def main():
    p = argparse.ArgumentParser(__doc__)
    for key in ("predictions", "processed-root", "index", "output"):
        p.add_argument("--"+key, required=True)
    a = p.parse_args()
    bank, labels, out = Path(a.predictions), Path(a.processed_root), Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    index = json.loads(Path(a.index).read_text())
    if len({r['token'] for r in index}) != len(index): raise ValueError('Duplicate scene')
    atomic_json(out/'identity.json', {
        'prediction_identity_sha256': file_sha256(bank/'identity.json'),
        'index_sha256': file_sha256(a.index), 'metric_code_sha256': file_sha256(__file__),
        'scope': 'label-side fit diagnostics, not PDMS', 'steps': 8, 'interval_s': .5})
    rows = []
    for scene in index:
        row = {k: scene[k] for k in ('token', 'log')}
        row['failure'] = None
        try:
            source = labels/(scene['token']+'.pkl')
            raw = load_trusted(source)
            target = relative_ego_target(np.asarray(raw['glo_status']['global_poses'])[:12])
            row['label_source_sha256'] = file_sha256(source)
            stationary = np.zeros((8, 3))
            row.update({'stationary_'+k: v for k,v in trajectory_errors(stationary,target).items()})
            row['motion_group'] = 'stationary' if np.linalg.norm(target[:,:2],axis=-1).max() <= .5 else 'moving'
            meta = json.loads((bank/'predictions'/(scene['token']+'.json')).read_text())
            if meta['status'] != 'ok': raise ValueError(meta.get('error','Failed prediction'))
            path = bank/'predictions'/(scene['token']+'.npz')
            if file_sha256(path) != meta['proposal_sha256']: raise ValueError('Prediction changed')
            with np.load(path) as values:
                row.update(trajectory_errors(values['trajectory'], target))
        except Exception as error:
            row['failure'] = repr(error)
        rows.append(row)
    fields = sorted(set().union(*(r.keys() for r in rows))) if rows else ['token','failure']
    with (out/'scenes.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)
    summary = {'scenes':len(rows),'failed_scenes':sum(r['failure'] is not None for r in rows),
               'metric_scope':'trajectory-fit diagnosis, not planning performance',
               'cv_reference':'NOT_RUN: this label adapter does not assert a current velocity frame'}
    for group in ('all','stationary','moving'):
        values = rows if group == 'all' else [r for r in rows if r.get('motion_group') == group]
        summary[group]={'scenes':len(values),'failed':sum(r['failure'] is not None for r in values)}
        for key in ('ADE','FDE','yaw_MAE_rad','yaw_endpoint_rad','stationary_ADE','stationary_FDE'):
            present=[r[key] for r in values if r.get(key) is not None]
            summary[group][key]=float(np.mean(present)) if present else None
            summary[group][key+'_denominator']=len(present)
    summary['valid_complete_result'] = bool(rows) and not summary['failed_scenes']
    atomic_json(out/'summary.json',summary)
    if not summary['valid_complete_result']: raise RuntimeError('Incomplete fit evaluation; all failed scene rows retained')


if __name__ == '__main__': main()
