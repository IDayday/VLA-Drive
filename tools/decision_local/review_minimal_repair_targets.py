"""Fixed training-only target review/export; scores never select a solver target."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,digest,identity,read,check_host
from starVLA.dataloader.foresight_dataset import decode_ego
from starVLA.dataloader.minimal_repair_scene import load_annotated_scene
from starVLA.model.modules.decision_local.local_constraint_checker import body_polygons


def main():
    p=argparse.ArgumentParser()
    for k in ('labels','anchor','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--stratified',action='store_true',help='Training-only fixed road/object/time repair and non-GT KEEP coverage')
    a=p.parse_args();check_host();meta=read(a.labels/'identity.json')
    if meta['split']!='train':raise ValueError('Target quality calibration is training-only')
    if a.output.exists():raise FileExistsError('Preserve previous review')
    a.output.mkdir(parents=True);(a.output/'predictions').mkdir();(a.output/'records').mkdir()
    rows=[read(a.labels/'records'/(r['token']+'.json')) for r in meta['index']]
    # Fixed class priority and token order; no PDMS or preference for a larger gain.
    selected={}
    if a.stratified:
        criteria=[('road_repair',lambda c:c['status']==1 and c['input_check']['road_violation']),
                  ('object_repair',lambda c:c['status']==1 and c['input_check']['annotated_overlap']),
                  ('time_repair',lambda c:c['status']==1 and c['plan_index'] in (3,4)),
                  ('native_repair',lambda c:c['status']==1 and c['plan_index']==0),
                  ('non_native_keep',lambda c:c['status']==0 and c['plan_index']!=0),
                  ('uncertain',lambda c:c['status']==2),('unsolved',lambda c:c['status']==4),
                  ('task_uncertain',lambda c:c['status']==6)]
        for group,predicate in criteria:
            count=0
            for row in rows:
                candidates=[c for c in row['cases'] if predicate(c)]
                if candidates and row['token'] not in selected:
                    selected[row['token']]=(row,{**candidates[0],'review_stratum':group});count+=1
                    if count==4:break
    for status in (1,0,2,3,4,5,6):
        for row in rows:
            candidates=[c for c in row['cases'] if c['status']==status]
            if candidates and row['token'] not in selected:
                selected[row['token']]=(row,candidates[0])
                if sum(c['status']==status for _,c in selected.values())>=4:break
    for row in rows:
        if len(selected)>=32:break
        if row['token'] not in selected:selected[row['token']]=(row,row['cases'][0])
    values=list(selected.values())[:32];index=[{'token':r['token'],'log':r['log']} for r,_ in values]
    contract={'schema':'minimal_repair_training_target_quality_export_v1','labels':meta['identity'],'index':index,
              'fixed_stratified_training_cases':a.stratified,
              'purpose':'official simulator check only, no score input to solver or target selection',
              'unresolved_output':'input retained solely for this target diagnostic, not a deployable GT gate'}
    ident=identity(contract);atomic_json(a.output/'identity.json',{'identity':ident,**contract});atomic_json(a.output/'index.json',index)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,3,figsize=(18,15));anchor=read(a.anchor)
    for n,(row,case) in enumerate(values):
        with np.load(a.labels/'scenes'/(row['token']+'.npz'),allow_pickle=False) as f:
            seed,plan=case['anchor_index'],case['plan_index'];qin=f['plan_in_physical'][seed,plan].copy()
            target=decode_ego(torch.from_numpy(f['repair_target'][seed,plan])).numpy() if case['target_valid'] else qin.copy()
            gt=decode_ego(torch.from_numpy(f['ego'])).numpy()
        path=a.output/'predictions'/(row['token']+'.npz')
        with path.open('wb') as stream:np.savez_compressed(stream,q0=qin,q_final=target)
        atomic_json(a.output/'records'/(row['token']+'.json'),{'identity':ident,'token':row['token'],'sha256':digest(path),'case':case})
        if n<9:
            ax=axs.flat[n];scene=load_annotated_scene(anchor,'train',row)
            geoms=[scene.road] if scene.road.geom_type=='Polygon' else list(scene.road.geoms)
            for geom in geoms:
                if geom.geom_type!='Polygon':continue
                xy=np.array(geom.exterior.coords);ax.fill(xy[:,0],xy[:,1],color='#e2e8dd')
                for hole in geom.interiors:
                    xy=np.array(hole.coords);ax.fill(xy[:,0],xy[:,1],color='white')
            obs,unknown,_=scene.at_times(np.array([2.]),'FUTURE')
            for polys,color in ((obs[0],'#ce6256'),(unknown[0],'#ad89c7')):
                for poly in ([polys] if polys.geom_type=='Polygon' else getattr(polys,'geoms',[])):
                    if poly.geom_type=='Polygon':xy=np.array(poly.exterior.coords);ax.fill(xy[:,0],xy[:,1],color=color,alpha=.3)
            ax.plot(qin[:,0],qin[:,1],'-o',label='plan_in');ax.plot(target[:,0],target[:,1],'-x',label='target')
            ax.plot(gt[:,0],gt[:,1],'--',color='gray',label='logged GT')
            for pose in target[1::2]:
                xy=np.array(body_polygons(pose[None],scene.body)[0].exterior.coords);ax.plot(xy[:,0],xy[:,1],color='orange',alpha=.6)
            xy=np.concatenate((qin[:,:2],target[:,:2],gt[:,:2]),0);lo=xy.min(0)-5;hi=xy.max(0)+5
            ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal');ax.set_title(row['token']+' '+case['status_name']+f' plan{plan}')
            ax.legend(fontsize=7);ax.set_xlabel('ego x / m');ax.set_ylabel('ego y / m')
    fig.tight_layout();fig.savefig(a.output/'PRIVATE_TARGET_REVIEW.png',dpi=120);plt.close(fig)
    atomic_json(a.output/'COMPLETE.json',{'identity':ident,'scenes':len(index),'no_score_based_selection':True,
        'selected_status_counts':{str(s):sum(c['status']==s for _,c in values) for s in range(7)}})
    print(json.dumps({'scenes':len(index),'output':str(a.output)}))


if __name__=='__main__':main()
