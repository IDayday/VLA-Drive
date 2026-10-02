"""Wrap already computed fixed mid-Qwen W interventions for canonical CPU PDMS.

No model, images, teacher or optimizer is run. Every original requested scene is
retained. This is a fixed development sensitivity diagnostic, not deployment.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from starVLA.model.modules.vehicle_joint.initialization import identity_hash, file_sha256
from tools.ddpolicy_vehicle.prepare_data import atomic_json


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--diagnostic',required=True)
    parser.add_argument('--current-data',required=True)
    parser.add_argument('--variant',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    root=Path(args.diagnostic);out=Path(args.output)
    report=json.loads((root/'SUMMARY.json').read_text())
    rows=json.loads((root/'queries.json').read_text())
    if args.variant not in ['base']+['W_permuted_layer'+str(i) for i in report['layers']]:
        raise ValueError('Only preregistered fixed layers allowed')
    if report['failures'] or report['scenes']!=len(rows) or report['real_optimizer_updates']:
        raise ValueError('Incomplete original intervention')
    for key in ('image_path_changed','sequence_length_changed','position_encoding_changed'):
        if report[key]:
            raise ValueError('Intervention changed an unregistered input path')
    current=json.loads((Path(args.current_data)/'identity.json').read_text())
    all_rows=json.loads((Path(args.current_data)/'index.json').read_text())
    by_token={row['token']:row for row in all_rows}
    if current['split']!='dev' or len({r['token'] for r in rows})!=len(rows) or any(by_token.get(r['token'])!=r for r in rows):
        raise ValueError('Diagnostic population is not the fixed development subset')
    current={**current,'index_sha256':identity_hash(rows),
        'identity':identity_hash({'original':current['identity'],'fixed_query_list':identity_hash(rows)})}
    protocol={'precision':'FP32','tf32':False,'scorer':None,'candidates_per_scene':1,
        'sampling_seed':42,'steps':10,'solver':'original Euler','future_conditioning':False,
        'current_views':['CAM_F0','CAM_L0','CAM_R0'],'history_images':0,'W_retained':144,
        'auxiliary_heads_removed':True,'mid_Qwen_intervention':args.variant,
        'batch':2,'sequence_length_changed':False,'position_encoding_changed':False,
        'purpose':'fixed intermediate-W sensitivity, potentially out of distribution; not causal proof'}
    identity={'checkpoint':report['checkpoint'],'current_identity':current,
        'evaluation_source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'protocol':protocol,'world_size':1,'limit':len(rows),
        'original_diagnostic_summary_sha256':file_sha256(root/'SUMMARY.json')}
    out.mkdir(parents=True,exist_ok=False);(out/'predictions').mkdir()
    atomic_json(out/'identity.json',identity);atomic_json(out/'current_index.json',rows)
    digest=identity_hash(identity)
    for row in rows:
        path=root/args.variant/(row['token']+'.npz')
        with np.load(path,allow_pickle=False) as saved:
            trajectory=saved['trajectory']
        if trajectory.shape!=(8,3) or not np.isfinite(trajectory).all():
            raise ValueError('Invalid original diagnostic trajectory; do not drop this scene')
        dest=out/'predictions'/path.name;shutil.copy2(path,dest)
        atomic_json(dest.with_suffix('.json'),{**row,'status':'ok','identity_sha256':digest,
            'proposal_sha256':file_sha256(dest),'source_proposal_sha256':file_sha256(path)})
    atomic_json(out/'shard_0.json',{'status':'complete','requested':len(rows),'completed':len(rows),
        'failed':0,'identity_sha256':digest})


if __name__=='__main__':
    main()
