"""Fixed current-geometry condition ablations and coherent joint K-sample diagnostics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from tools.joint_local_scene_v3.budget import BudgetRun,atomic_json
from tools.joint_local_scene_v3.data import AnnotatedCorpus
from tools.joint_local_scene_v3.runtime import build_queries,batch_scenes,evaluation_noise,target_metrics
from tools.joint_local_scene_v3.train_mechanism import deterministic_settings
from starVLA.model.modules.joint_scene.flow import JointSceneFlow,execution_from_joint

RELATION_RULE={'peer_role':'non-ego traffic actor, distinct from target','strong':'closest peer within15m of current target centre','weak':'different peer at least5m farther than strong; complete-xy-point count differs by<=1; minimize count difference then maximize distance','removed_actors_per_condition':1,'selection_uses_target_future_values':False,'NOT_APPLICABLE':'No valid strong/weak pair; keep query in denominator'}


def relation_queries(corpus,queries):
    result=[]
    for q in queries['queries']:
        s=corpus[q['scene_index']];g=s.graph;points=s.feature_valid[0,:,:,:2].all(-1).sum(-1);target=q['slot']
        peers=[i for i in range(1,g.boxes.shape[1]) if i!=target and bool(g.active_actor_mask[0,i]) and int(points[i])>0]
        distances={i:float((g.boxes[0,i,:2]-g.boxes[0,target,:2]).norm()) for i in peers}
        peers.sort(key=lambda i:(distances[i],i));strong=peers[0] if peers and distances[peers[0]]<=15 else None
        weak=[i for i in peers if strong is not None and i!=strong and distances[i]>=distances[strong]+5 and abs(int(points[i]-points[strong]))<=1]
        weak.sort(key=lambda i:(abs(int(points[i]-points[strong])),-distances[i],i))
        row=dict(q,status='applicable' if weak else 'NOT_APPLICABLE',strong_actor=strong,weak_actor=weak[0] if weak else None)
        if weak:row.update(strong_distance_m=distances[strong],weak_distance_m=distances[weak[0]],strong_xy_points=int(points[strong]),weak_xy_points=int(points[weak[0]]))
        result.append(row)
    payload={'rule':RELATION_RULE,'query_identity':queries['sha256'],'queries':result}
    payload['sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest();return payload


def write_csv(path,rows):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(r.keys() for r in rows))),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def append(path,row):
    with Path(path).open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')


def visualize(s,pred,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    fig,ax=plt.subplots(figsize=(10,8));g=s.graph;b=g.boxes[0].numpy();v=s.feature_valid[0,:,:,:2].all(-1).numpy();y=s.future[0].numpy();active=g.active_actor_mask[0].numpy()
    context=g.context_boxes[0,g.context_mask[0]].numpy();ax.scatter(context[:,0],context[:,1],c='orange',marker='x',alpha=.4,label='current context')
    for i in np.where(active)[0]:
        color='black' if i==0 else plt.get_cmap('tab20')(i%20)
        yaw=np.arctan2(b[i,6],b[i,7]);rot=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
        corners=np.array([[-1,-1],[-1,1],[1,1],[1,-1]])*b[i,3:5]/2
        ax.add_patch(Polygon(corners@rot.T+b[i,:2],fill=False,edgecolor=color));ax.text(*b[i,:2],str(i))
        ax.plot(y[i,v[i],0],y[i,v[i],1],':',color=color)
        ax.plot(pred[i,:,0],pred[i,:,1],'-o',color=color,markersize=2,linewidth=2 if i==0 else 1)
    ax.set_title('Trained structured-current model; joint sample k=0\nsolid: predicted xy; dotted: valid GT; boxes: CURRENT state only')
    ax.set_xlabel('ego(t0) x [m]');ax.set_ylabel('ego(t0) y [m]');ax.axis('equal');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(path);plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    for name in ('data','checkpoint','output','ledger','run-id'):p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,default=20260927);p.add_argument('--samples',type=int,default=8);p.add_argument('--protocol',required=True);a=p.parse_args();out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise FileExistsError('New diagnostics output required')
    out.mkdir(parents=True,exist_ok=True)
    checkpoint=Path(a.checkpoint);identity={'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'module_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'output':str(out.resolve()),'seed':a.seed,'samples':a.samples,'protocol_sha256':hashlib.sha256(Path(a.protocol).read_bytes()).hexdigest()}
    with BudgetRun(a.ledger,a.run_id,identity,1) as budget:
        deterministic_settings();saved=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=saved['identity']['config']
        corpus=AnnotatedCorpus(a.data);queries=build_queries(corpus);protocol=relation_queries(corpus,queries)
        if protocol!=json.loads(Path(a.protocol).read_text()):raise ValueError('Fixed relation-query protocol mismatch')
        atomic_json(out/'protocol.json',protocol);atomic_json(out/'identity.json',identity)
        model=JointSceneFlow(**cfg['model']).cuda();model.load_state_dict(saved['model'],strict=True);model.eval()
        hook=model.register_forward_pre_hook(lambda *unused:budget.note('real',forwards=1))
        rows=[];kscenes=[];kqueries=[]
        plot_indices=sorted(range(len(corpus)),key=lambda i:hashlib.sha256(corpus.records[i]['token'].encode()).digest())[:8]
        with torch.no_grad():
            applicable=[q for q in protocol['queries'] if q['status']=='applicable']
            for start in range(0,len(applicable),16):
                budget.check();chunk=applicable[start:start+16];scenes=[corpus[q['scene_index']] for q in chunk];g,y,v=batch_scenes(scenes,'cuda');noise=evaluation_noise(scenes,a.seed)
                masks=v.clone()
                for j,q in enumerate(chunk):masks[j,q['slot']]=False
                batch_rows=[dict(q,status='ok') for q in chunk]
                for name,field in [('correct',None),('remove_strong','strong_actor'),('remove_weak','weak_actor')]:
                    budget.check();known=masks.clone()
                    if field is not None:
                        for j,q in enumerate(chunk):known[j,q[field]]=False
                    try:
                        joint=model.sample_conditional(noise,g,y,known,sampling_steps=cfg['sampling_steps'])
                        for j,q in enumerate(chunk):
                            if not torch.isfinite(joint[j]).all():raise ValueError('Nonfinite condition-ablation sample')
                            slot=q['slot'];metric=target_metrics(joint[j,slot],y[j,slot],v[j,slot],g.boxes[j,slot])
                            batch_rows[j][name+'_ADE_m']=metric['xy_ADE_m'];batch_rows[j][name+'_error_sum_m']=metric['xy_error_sum_m']
                    except (RuntimeError,ValueError) as exc:
                        for row in batch_rows:row.update(status='failed',error=str(exc))
                for row in batch_rows:
                    if row['status']=='ok':row.update(strong_minus_correct_m=row['remove_strong_ADE_m']-row['correct_ADE_m'],weak_minus_correct_m=row['remove_weak_ADE_m']-row['correct_ADE_m'],strong_minus_weak_m=row['remove_strong_ADE_m']-row['remove_weak_ADE_m'])
                    rows.append(row);append(out/'condition_progress.jsonl',row)
            for q in protocol['queries']:
                if q['status']=='NOT_APPLICABLE':rows.append(dict(q));append(out/'condition_progress.jsonl',q)
            bank={i:[None]*a.samples for i in range(len(corpus))}
            jobs=[(i,k) for i in range(len(corpus)) for k in range(a.samples)]
            for start in range(0,len(jobs),16):
                budget.check();chunk=jobs[start:start+16];scenes=[corpus[i] for i,k in chunk];g,_,_=batch_scenes(scenes,'cuda')
                noises=torch.cat([evaluation_noise([scene],a.seed+k) for scene,(i,k) in zip(scenes,chunk)])
                sample=model.sample(noises,g,sampling_steps=cfg['sampling_steps'])
                if not torch.isfinite(sample).all():raise ValueError('Nonfinite K-sample output')
                for j,(i,k) in enumerate(chunk):bank[i][k]=sample[j].cpu()
            for i in range(len(corpus)):
                budget.check();s=corpus[i];g=s.graph;y=s.future;v=s.feature_valid
                valid=v[0,:,:,:2].all(-1);truth=torch.where(v,y,0.)[0,:,:,:2];scene_rows=[];samples=torch.stack(bank[i])
                for k,joint in enumerate(samples):
                    error=(joint[...,:2]-truth).norm(dim=-1);active=g.active_actor_mask[0]
                    pair=valid[:,None]&valid[None,:]&torch.triu(torch.ones(len(valid),len(valid),dtype=torch.bool),diagonal=1)[...,None]
                    relative_error=((joint[:,None,:,:2]-joint[None,:,:,:2])-(truth[:,None]-truth[None,:])).norm(dim=-1)
                    distance=(joint[1:,:,:2]-joint[0,None,:,:2]).norm(dim=-1)
                    scene_row={'token':s.token,'log':s.log,'k':k,'seed':a.seed+k,'joint_error_sum_m':float(error[valid].sum()),'joint_valid_points':int(valid.sum()),'joint_relative_error_sum_m':float(relative_error[pair].sum()),'joint_relative_valid_points':int(pair.sum()),'same_joint_ego_neighbor_min_distance_m':float(distance[active[1:]].min()) if active[1:].any() else None,'execution_ego_xy_difference_m':float((execution_from_joint(joint[None])['executed_ego_xyyaw'][0,:,:2]-joint[0,:,:2]).abs().max())}
                    kscenes.append(scene_row);scene_rows.append(scene_row);append(out/'kscene_progress.jsonl',scene_row)
                    for q in [q for q in queries['queries'] if q['scene_index']==i]:
                        slot=q['slot'];metric=target_metrics(joint[slot],y[0,slot],v[0,slot],g.boxes[0,slot],g.ego_state[0,1:3] if slot==0 else None)
                        item=dict(q,k=k,seed=a.seed+k,**{key:val for key,val in metric.items() if key not in q});kqueries.append(item);append(out/'kquery_progress.jsonl',item)
                diversity=(samples[:,None,...,:2]-samples[None,...,:2]).norm(dim=-1);upper=torch.triu(torch.ones(a.samples,a.samples,dtype=torch.bool),diagonal=1)
                oracle=min(range(a.samples),key=lambda k:scene_rows[k]['joint_error_sum_m'])
                diag={'token':s.token,'log':s.log,'joint_oracle_k':oracle,'oracle_non_deployment':True,'joint_oracle_ADE_m':scene_rows[oracle]['joint_error_sum_m']/scene_rows[oracle]['joint_valid_points'],'mean_pair_sample_distance_m':float(diversity[upper][:,valid].mean()) if a.samples>1 else None}
                append(out/'joint_diversity_oracle.jsonl',diag)
                if i in plot_indices:visualize(s,samples[0].numpy(),out/f'private_scene_{i}.png')
            torch.save(bank,out/'private_joint_samples.pt')
        hook.remove();write_csv(out/'condition_queries.csv',rows);write_csv(out/'k_scenes.csv',kscenes);write_csv(out/'k_queries.csv',kqueries)
        summary={'complete':True,'expected_queries':len(protocol['queries']),'applicable':sum(r['status']=='ok' for r in rows),'not_applicable':sum(r['status']=='NOT_APPLICABLE' for r in rows),'failed':sum(r['status']=='failed' for r in rows),'samples':a.samples,'real_optimizer_updates':0,'scope':'GT-current structured mechanism; privileged future ablations are diagnostics, not causal proof or PDMS','identity':identity}
        atomic_json(out/'summary.json',summary);print(json.dumps(summary,indent=2))
        if summary['failed']:raise RuntimeError('Diagnostic failures retained')


if __name__=='__main__':main()
