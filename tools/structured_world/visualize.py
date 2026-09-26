"""Real-scene evidence: current camera boxes, t0 BEV, future track masks and slot matches."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from starVLA.model.modules.structured_world.contracts import WorldTargets
from starVLA.model.modules.structured_world.matching import match_current


def corners(box):
    x,y,z,l,w,h,s,c=box
    yaw=np.arctan2(s,c);r=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1.]])
    signs=np.array([[-1,-1,-1],[-1,1,-1],[1,1,-1],[1,-1,-1],[-1,-1,1],[-1,1,1],[1,1,1],[1,-1,1]])
    return (signs*np.array([l,w,h])/2)@r.T+np.array([x,y,z])


def project(points,k,ext,d):
    inv=np.linalg.inv(ext);p=points@inv[:3,:3].T+inv[:3,3]
    x,y=(p[:,:2]/np.maximum(p[:,2,None],1e-6)).T;r2=np.minimum(x*x+y*y,1e4)
    k1,k2,p1,p2,k3=d;rad=1+k1*r2+k2*r2*r2+k3*r2**3
    distorted=np.stack([x*rad+2*p1*x*y+p2*(r2+2*x*x),y*rad+p1*(r2+2*y*y)+2*p2*x*y,np.ones_like(x)],-1)
    return (distorted@k.T)[:,:2],p[:,2]>.1


def main():
    p=argparse.ArgumentParser()
    for n in ['predictions','target-cache','sensor-root','output']:p.add_argument('--'+n,required=True)
    p.add_argument('--limit',type=int,default=64);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    cache=Path(a.target_cache);records=[]
    for path in sorted(Path(a.predictions).glob('*.npz'))[:a.limit]:
        token=path.stem;arrays=dict(np.load(path));pred={k:torch.from_numpy(arrays[k]) for k in ['logits','boxes','future_xy']}
        target=WorldTargets(**torch.load(cache/'targets'/f'{token}.pt',weights_only=True));obs=dict(np.load(cache/'observations'/f'{token}.npz'))
        rows,cols=match_current(pred,target);boxes=target.current_boxes.numpy()
        fig=plt.figure(figsize=(18,10));grid=fig.add_gridspec(2,3);axes=[fig.add_subplot(grid[0,i]) for i in range(3)]
        for v,ax in enumerate(axes):
            with Image.open(Path(a.sensor_root)/str(obs['image_paths'][v])) as im:
                im=im.convert('RGB');w,h=im.size
                if w/h>16/9:
                    cw=int(h*16/9);im=im.crop(((w-cw)//2,0,(w+cw)//2,h))
                elif w/h<16/9:
                    ch=int(w*9/16);im=im.crop((0,(h-ch)//2,w,(h+ch)//2))
                ax.imshow(im.resize((1024,576),Image.Resampling.LANCZOS))
            for box in boxes:
                uv,valid=project(corners(box),obs['intrinsics'][v],obs['extrinsics'][v],obs['distortion'][v])
                for i,j in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
                    if valid[i] and valid[j]:ax.plot(uv[[i,j],0],uv[[i,j],1],color='lime',linewidth=.7)
            ax.set(xlim=(0,1024),ylim=(576,0),title=str(obs['camera_names'][v]));ax.axis('off')
        bev=fig.add_subplot(grid[1,:2]);table=fig.add_subplot(grid[1,2]);table.axis('off');lines=[]
        for slot,gt in zip(rows.tolist(),cols.tolist()):
            color=plt.cm.tab20(gt%20)
            for box,style in [(boxes[gt],'-'),(arrays['boxes'][slot],'--')]:
                poly=corners(box)[[0,1,2,3,0],:2];bev.plot(poly[:,1],poly[:,0],style,color=color,linewidth=.8)
            xy=target.future_xy_in_ego_t0[gt].numpy().copy();mask=target.future_valid_mask[gt].numpy();xy[~mask]=np.nan
            bev.plot(xy[:,1],xy[:,0],'.-',color=color,markersize=3)
            pred_xy=arrays['future_xy'][slot];bev.plot(pred_xy[:,1],pred_xy[:,0],':',color=color,linewidth=.6)
            bev.text(boxes[gt,1],boxes[gt,0],str(slot),fontsize=5)
            lines.append(f'{slot:02d} -> {target.track_ids[gt][-8:]}   '+''.join('1' if x else '0' for x in mask))
        bev.plot(0,0,'k^');bev.set(xlabel='ego(t0) y [m], left',ylabel='ego(t0) x [m], forward',xlim=(22,-22),ylim=(-2,55),title='GT box solid / predicted box dashed; GT future solid / prediction dotted');bev.set_aspect('equal');bev.grid(alpha=.2)
        table.text(0,1,'slot -> track suffix / future valid mask\n'+'\n'.join(lines[:64]),va='top',family='monospace',fontsize=5.5)
        category={'empty':len(boxes)==0,'missing_future':bool((~target.future_valid_mask).any()),'static_track':bool(((target.future_xy_in_ego_t0[:,-1]-target.current_boxes[:,:2]).norm(dim=-1)<.3).logical_and(target.future_valid_mask[:,-1]).any())}
        fig.suptitle(f'{token} | t0={int(obs["timestamp"])} | GT={len(boxes)}, overflow={target.overflow}');fig.tight_layout();fig.savefig(out/f'{token}.png',dpi=110);plt.close(fig)
        records.append({'token':token,'file':str(out/f'{token}.png'),'categories':category,'matched_slots':len(rows)})
    (out/'index.json').write_text(json.dumps(records,indent=2));print(json.dumps({'rendered':len(records),'categories':{k:sum(r['categories'][k] for r in records) for k in ['empty','missing_future','static_track']}}))

if __name__=='__main__':main()
