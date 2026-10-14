"""Deterministic balanced geometric cases, kept in controlled project storage.

Uses completed post-hoc groups and canonical scores. No private camera images
are read; no selected case alters a policy, checkpoint or training population.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from shapely.geometry import MultiPoint
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,digest,read,source_sha
from starVLA.model.modules.structured_world.queries import body_points


def choose(rows):
    groups=['road_boundary_or_outside','dynamic_enter_at_query','space_release_at_query',
        'reliable_current_stationary_obstacle_near_query','original_progress_below_0p5',
        'original_high_ge_0p9','future_relation_support_below_half']
    selected=[]
    for group in groups:
        for transition in ('zero_to_nonzero','nonzero_to_zero','both_nonzero','both_zero'):
            def belongs(row):
                before,after=row['q0_score'],row['final_score']
                return {'zero_to_nonzero':before==0 and after>0,'nonzero_to_zero':before>0 and after==0,
                        'both_nonzero':before>0 and after>0,'both_zero':before==0 and after==0}[transition]
            eligible=sorted((r for r in rows if r['groups'][group] and belongs(r)),
                key=lambda r:(r['final_score']-r['q0_score'],r['log'],r['token']))
            # Low/middle/high rank, the same deterministic rule for improvements
            # and regressions. Empty strata stay empty; no success substitution.
            for index in sorted({int(np.floor(q*(len(eligible)-1))) for q in (.1,.5,.9)} if eligible else []):
                selected.append({'group':group,'transition':transition,'rank':index,'population':len(eligible),
                    'token':eligible[index]['token'],'log':eligible[index]['log'],
                    'q0_score':eligible[index]['q0_score'],'final_score':eligible[index]['final_score']})
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('analysis','predictions','label-root','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Immutable case selection')
    analysis=read(a.analysis/'summary.json');pm=read(a.predictions/'identity.json')
    if analysis['predictions']!=pm['identity']:raise ValueError('Cases require the same scored predictions')
    rows=[json.loads(s) for s in (a.analysis/'scenes.jsonl').read_text().splitlines()]
    selected=choose(rows);a.output.mkdir(parents=True)
    tokens=sorted({r['token'] for r in selected});extent=(-20,80,-40,40)
    for token in tokens:
        label=a.label_root/'labels'/(token+'.npz');receipt=read(a.label_root/'records'/(token+'.json'))
        if digest(label)!=receipt['label_sha256']:raise ValueError('Original geometric label changed')
        with np.load(label) as f:
            road=f['road_distance'].copy();occ=f['occupancy'].copy();valid=f['occupancy_valid'].copy()
        if occ.shape!=(9,160,200) or road.shape!=(160,200):
            raise ValueError('This case plot supports only the registered NAVSIM grid and eight future times')
        prediction=a.predictions/'predictions'/(token+'.npz')
        if digest(prediction)!=read(a.predictions/'records'/(token+'.json'))['sha256']:
            raise ValueError('Scored trajectory changed')
        with np.load(prediction) as f:q0=f['q0'].copy();qf=f['q_final'].copy()
        chosen=[r for r in selected if r['token']==token]
        fig,axes=plt.subplots(1,3,figsize=(15,6),layout='constrained')
        for ax,t in zip(axes,[0,4,8]):
            # Original strict validity is shown explicitly; the documented
            # current-ground support extension is not rendered as observed data.
            ax.imshow(np.where(valid[t],0.,1.),extent=extent,origin='lower',cmap='Greys',vmin=0,vmax=3,alpha=.4)
            ax.contour(road,levels=[0],extent=extent,origin='lower',colors=['green'],linewidths=.8)
            mask=np.ma.masked_where((occ[t]!=1)|~valid[t],occ[t])
            ax.imshow(mask,extent=extent,origin='lower',cmap='Oranges',vmin=0,vmax=1,alpha=.75)
            ax.plot(np.r_[0,q0[:,0]],np.r_[0,q0[:,1]],'o-',color='royalblue',label='S0 q0',markersize=3)
            ax.plot(np.r_[0,qf[:,0]],np.r_[0,qf[:,1]],'o-',color='crimson',label='q_final',markersize=3)
            for plan,color in [(q0,'royalblue'),(qf,'crimson')]:
                pose=np.zeros(3,dtype=np.float32) if t==0 else plan[t-1]
                points=body_points(torch.from_numpy(pose[None]),receipt['ego_body'])[0].numpy()
                outline=np.asarray(MultiPoint(points).convex_hull.exterior.coords)
                ax.plot(outline[:,0],outline[:,1],color=color,linewidth=1.2)
            ax.scatter([0],[0],marker='s',color='black',s=25)
            ax.set(xlim=(-8,min(80,max(25,float(q0[:,0].max())+12))),ylim=(-20,20),
                   xlabel='ego(t0) x forward / m',ylabel='y left / m',title=f'Logged environment t={t*.5:g}s')
            ax.set_aspect('equal');ax.legend(loc='upper left',fontsize=8)
        fig.suptitle(f"{token}\nPDMS q0={chosen[0]['q0_score']:.4f}, final={chosen[0]['final_score']:.4f}; geometric labels, not official collision GT",fontsize=10)
        fig.savefig(a.output/(token+'.png'),dpi=110);plt.close(fig)
    atomic_json(a.output/'MANIFEST.json',{'source_sha':source_sha(ROOT),'writer_sha256':digest(__file__),
        'analysis_sha256':digest(a.analysis/'summary.json'),'predictions':pm['identity'],
        'selection':'fixed category × four zero/nonzero transitions ×10/50/90% ordered-delta ranks; no empty-stratum substitution',
        'cases':selected,'unique_scenes':len(tokens),'private_RGB_read_or_uploaded':False,
        'storage':'Controlled project artifacts only; do not commit derived scene maps',
        'limitations':['Plots use original strict label support; gray is unknown, not free',
            'Ground projection and raster approximation are not official responsibility/TTC labels',
            'Case groups overlap; selected scenes are illustrative, not independent statistical evidence']})


if __name__=='__main__':main()
