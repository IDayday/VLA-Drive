"""Private current-camera/joint-sample figures; never use GT for inference."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.structured_world.visualize import corners, project
from .prepare_data import atomic_json
from starVLA.model.modules.vehicle_joint.initialization import file_sha256


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('predictions','current-root','vehicle-root','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--limit',type=int,default=8);a=p.parse_args()
    if a.limit<1:raise ValueError('Positive diagnostic population required')
    bank,current,labels,out=map(Path,(a.predictions,a.current_root,a.vehicle_root,a.output));out.mkdir(parents=True,exist_ok=False)
    index=json.loads((current/'index.json').read_text())[:a.limit]
    records=[];edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for scene in index:
        token=scene['token'];obs=json.loads((current/'current'/(token+'.json')).read_text())
        meta=json.loads((bank/'predictions'/(token+'.json')).read_text());path=bank/'predictions'/(token+'.npz')
        if meta['status']!='ok' or file_sha256(path)!=meta['proposal_sha256']:raise ValueError('Missing/changed prediction cannot be skipped')
        with np.load(path) as z:pred={k:z[k] for k in z.files}
        with torch.serialization.safe_globals([np.core.multiarray.scalar,np.dtype,type(np.dtype('U16')),np.str_]):
            payload=torch.load(labels/'targets'/(token+'.pt'),weights_only=True,map_location='cpu')
        target=payload['targets'];boxes=target['current_boxes'].numpy();future=target['future_xy_in_ego_t0'].numpy();valid=target['future_valid_mask'].numpy()
        candidates=np.zeros(0,dtype=int)
        if 'vehicle_logits' in pred:
            logits=pred['vehicle_logits'];prob=np.exp(logits-logits.max(-1,keepdims=True));prob=prob[:,0]/prob.sum(-1)
            candidates=np.flatnonzero(prob>=.2)
        fig=plt.figure(figsize=(18,11));grid=fig.add_gridspec(2,3)
        calibration=obs['current_calibration']
        for camera in range(3):
            ax=fig.add_subplot(grid[0,camera])
            with Image.open(obs['image_paths'][camera]) as im:
                im=im.convert('RGB');w,h=im.size;cw=min(w,int(h*16/9));ch=min(h,int(w*9/16))
                im=im.crop(((w-cw)//2,(h-ch)//2,(w+cw)//2,(h+ch)//2)).resize((1024,576),Image.Resampling.BICUBIC)
                ax.imshow(im)
            for values,color in ((boxes,'lime'),(pred.get('vehicle_boxes',np.zeros((0,8)))[candidates],'cyan')):
                for box in values:
                    uv,visible=project(corners(box),np.asarray(calibration['intrinsics'][camera]),
                        np.asarray(calibration['extrinsics'][camera]),np.asarray(calibration['distortion'][camera]))
                    for i,j in edges:
                        if visible[i] and visible[j]:ax.plot(uv[[i,j],0],uv[[i,j],1],color=color,linewidth=.6)
            ax.set(xlim=(0,1024),ylim=(576,0),title=f'Current camera {camera}: GT green, predicted cyan');ax.axis('off')
        ax=fig.add_subplot(grid[1,:2]);table=fig.add_subplot(grid[1,2]);table.axis('off')
        for i,box in enumerate(boxes):
            poly=corners(box)[[0,1,2,3,0],:2];ax.plot(poly[:,1],poly[:,0],color='green',linewidth=.7)
            xy=future[i].copy();xy[~valid[i]]=np.nan;ax.plot(xy[:,1],xy[:,0],'.-',color='green',alpha=.45,markersize=3)
        active=pred.get('active_actor_mask',np.array([True]));selected=pred.get('selected_query_indices',np.array([-1]))
        for actor in np.flatnonzero(active[1:])+1:
            query=selected[actor];poly=corners(pred['vehicle_boxes'][query])[[0,1,2,3,0],:2]
            ax.plot(poly[:,1],poly[:,0],'--',color='tab:orange',linewidth=1)
            xy=pred['vehicle_xy'][actor-1];ax.plot(xy[:,1],xy[:,0],'.-',color='tab:orange',markersize=3)
            ax.text(poly[0,1],poly[0,0],f'q{query}',fontsize=7)
        ego=pred['trajectory'];ax.plot(ego[:,1],ego[:,0],'o-',color='red',linewidth=2,label='Executed ego: joint slot0')
        if 'ego_future_xyyaw' in payload:
            ego_gt=payload['ego_future_xyyaw'].numpy().copy();ego_gt[~payload['ego_future_valid_mask'].numpy()]=np.nan
            ax.plot(ego_gt[:,1],ego_gt[:,0],'--',color='black',label='GT ego (label only)')
        ax.plot(0,0,'k^');ax.set(xlim=(22,-22),ylim=(-3,55),xlabel='ego(t0) y [m], left',ylabel='ego(t0) x [m], forward')
        ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(fontsize=8)
        lines=[f'GT vehicle {i}: '+''.join('1' if x else '0' for x in m) for i,m in enumerate(valid)]
        table.text(0,1,f'Current GT vehicles: {len(boxes)}\nPredicted candidates: {len(candidates)}\nSelected joint vehicles: {int(active[1:].sum())}\n\nFuture label masks (1 = valid)\n'+'\n'.join(lines),va='top',fontsize=8,family='monospace')
        fig.suptitle('Camera-only prediction; one joint sample. GT used only for this offline figure.\n'+token)
        fig.tight_layout();fig.savefig(out/(token+'.png'),dpi=120);plt.close(fig)
        records.append({'token':token,'log':scene['log'],'file':token+'.png','selected_vehicles':int(active[1:].sum())})
    atomic_json(out/'manifest.json',{'selection':'first N rows of fixed input index, before reading prediction errors',
        'prediction_identity_sha256':file_sha256(bank/'identity.json'),'scope':'private scene images; do not commit/push',
        'records':records})


if __name__=='__main__':main()
