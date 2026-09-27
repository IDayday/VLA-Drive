"""Score frozen exported joint trajectories using current geometric association and full GT."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from starVLA.model.modules.structured_world.metrics import prediction_filters, match_geometry
from tools.joint_world.train_graph import load_samples
from tools.structured_world_v1p1.reaudit_metrics import write_csv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['artifacts','cache','targets','data-root','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--runs',nargs='+',required=True)
    a=p.parse_args();root=Path(a.artifacts);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((Path(a.cache)/'manifest.json').read_text())
    tokens={r['token'] for r in manifest['records']};scenes={n:[] for n in a.runs};objects={n:[] for n in a.runs}
    for name in a.runs:
        meta=json.loads((root/name/'manifest_0.json').read_text())
        if meta['failed'] or {p.stem for p in (root/name/'predictions').glob('*.npz')}!=tokens:
            raise ValueError('Need every complete exported scene')
    for record in manifest['records']:
        sample=load_samples(a.cache,a.targets,a.data_root,8,records=[record],manifest_override=manifest)[0][0]
        token=record['token'];target=sample['target'];cache=sample['cache']
        pred={'boxes':cache['current_boxes'][0],'logits':cache['current_logits'][0]}
        _,support,exists=prediction_filters(pred,target);slots=torch.where(support&exists)[0];gt=torch.where(target.current_supervision_mask)[0]
        rows,cols=match_geometry(pred['boxes'][slots,:2],target.current_boxes[gt,:2]);rows,cols=slots[rows],gt[cols]
        matches={int(g):int(s) for s,g in zip(rows,cols)}
        valid=target.future_valid_mask[cols];all_valid=target.future_valid_mask[gt]
        stationary=(pred['boxes'][rows,None,:2]-target.future_xy_in_ego_t0[cols]).norm(dim=-1)
        for name in a.runs:
            with np.load(root/name/'predictions'/(token+'.npz')) as z:
                joint=torch.from_numpy(z['joint_xy']);ego_plan=torch.from_numpy(z['trajectory'][:,:2])
            if joint.shape!=(65,8,2) or not torch.isfinite(joint).all():raise ValueError('Invalid joint export')
            error=(joint[rows+1]-target.future_xy_in_ego_t0[cols]).norm(dim=-1)
            ego=(ego_plan-sample['xy'][0,0]).norm(dim=-1);graph_ego=(joint[0]-sample['xy'][0,0]).norm(dim=-1)
            scenes[name].append({'token':token,'status':'ok','gt':len(gt),'current_predictions':len(slots),'current_matches':len(rows),
                'class_correct_matches':int((pred['logits'][rows].argmax(-1)==target.current_classes[cols]).sum()),
                'centre_error_sum':float((pred['boxes'][rows,:2]-target.current_boxes[cols,:2]).norm(dim=-1).sum()),
                'agent_error_sum':float(error[valid].sum()),'agent_points':int(valid.sum()),
                'agent_final_error_sum':float(error[:,-1][valid[:,-1]].sum()),'agent_final_count':int(valid[:,-1].sum()),
                'valid_gt_points':int(all_valid.sum()),'valid_gt_instances':int(all_valid.any(-1).sum()),
                'matched_motion_instances':int(valid.any(-1).sum()),'stationary_error_sum':float(stationary[valid].sum()),
                'DiT_ego_ADE':float(ego.mean()),'DiT_ego_FDE':float(ego[-1]),'joint_ego_ADE':float(graph_ego.mean())})
            for gi in gt.tolist():
                slot=matches.get(gi);mask=target.future_valid_mask[gi]
                distance=None if slot is None else (joint[slot+1]-target.future_xy_in_ego_t0[gi]).norm(dim=-1)
                objects[name].append({'token':token,'track_id':target.track_ids[gi],'matched_slot':slot,'detected':slot is not None,
                    'valid_future_points':int(mask.sum()),'ADE':None if slot is None or not mask.any() else float(distance[mask].mean()),
                    'FDE':None if slot is None or not mask[-1] else float(distance[-1])})
    summary={}
    for name,rows in scenes.items():
        total=lambda k:sum(r[k] for r in rows)
        ratio=lambda x,y:total(x)/total(y) if total(y) else None
        summary[name]={'scenes':len(rows),'failed':0,'geometric_recall_2m':ratio('current_matches','gt'),
            'geometric_precision_2m':ratio('current_matches','current_predictions'),
            'class_correct_recall_2m':ratio('class_correct_matches','gt'),'current_predictions':total('current_predictions'),
            'false_positives':total('current_predictions')-total('current_matches'),'centre_error_matched':ratio('centre_error_sum','current_matches'),
            'matched_agent_ADE':ratio('agent_error_sum','agent_points'),'matched_agent_FDE':ratio('agent_final_error_sum','agent_final_count'),
            'stationary_same_predicted_centre_ADE':ratio('stationary_error_sum','agent_points'),
            'motion_point_coverage':ratio('agent_points','valid_gt_points'),
            'motion_instance_coverage':ratio('matched_motion_instances','valid_gt_instances'),
            'DiT_ego_ADE':total('DiT_ego_ADE')/len(rows),'DiT_ego_FDE':total('DiT_ego_FDE')/len(rows),
            'joint_ego_ADE':total('joint_ego_ADE')/len(rows)}
        dest=out/name;dest.mkdir(exist_ok=True)
        for filename,data in [('scenes.csv',rows),('objects.csv',objects[name])]:
            path=dest/filename;write_csv(path,data);path.write_text(path.read_text())
    report={'runs':summary,'association':'Independent2m current centre matching, reused for entire track future; all GT object rows retained.',
            'scope':'Exactly the current-only graph and originalDiT trajectories used in full planning exports; no re-sampling.',
            'limitation':'Low detection coverage must accompany matched-agent errors; fullset graph noise differs from the earlier per-token64scene probe.'}
    (out/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
