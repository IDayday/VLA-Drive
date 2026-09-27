"""Batched structured-mechanism evaluation, with complete denominators."""
import hashlib
import torch
from starVLA.model.modules.joint_scene.contracts import stack_graphs
from starVLA.model.modules.joint_scene.flow import execution_from_joint


def batch_scenes(scenes,device='cuda'):
    return stack_graphs([s.graph for s in scenes],device),torch.cat([s.future for s in scenes]).to(device),torch.cat([s.feature_valid for s in scenes]).to(device)


def evaluation_noise(scenes,seed,device='cuda'):
    rows=[]
    for s in scenes:
        h=int.from_bytes(hashlib.sha256(f'{seed}:{s.token}'.encode()).digest()[:8],'little')%(2**63-1)
        rows.append(torch.randn(s.future.shape,generator=torch.Generator().manual_seed(h)))
    return torch.cat(rows).to(device)


def metric_sums(prediction,truth,valid,current,ego_velocity):
    truth=torch.where(valid,truth,0.);xy_valid=valid[...,:2].all(-1)
    error=(prediction[...,:2]-truth[...,:2]).norm(dim=-1)
    stationary=(current[:,:,None,:2]-truth[...,:2]).norm(dim=-1)
    moving=((truth[...,:2]-current[:,:,None,:2]).norm(dim=-1).masked_fill(~xy_valid,0.).amax(-1)>=1.)&xy_valid.any(-1)
    result={}
    for name,sl in [('ego',slice(0,1)),('neighbor',slice(1,None))]:
        for group,selection in [('all',xy_valid[:,sl]),('dynamic',xy_valid[:,sl]&moving[:,sl,None]),('static',xy_valid[:,sl]&~moving[:,sl,None])]:
            key=name+'_'+group;d=error[:,sl];base=stationary[:,sl]
            result.update({key+'_error_sum':float(d[selection].sum()),key+'_points':int(selection.sum()),key+'_stationary_sum':float(base[selection].sum()),
                key+'_fde_sum':float(d[:,:,-1][selection[:,:,-1]].sum()),key+'_fde_count':int(selection[:,:,-1].sum())})
    time=torch.arange(1,prediction.shape[2]+1,device=prediction.device)*.5
    cv=ego_velocity[:,None,:]*time[None,:,None]
    cv_error=(cv-truth[:,0,:,:2]).norm(dim=-1)
    result['ego_cv_error_sum']=float(cv_error[xy_valid[:,0]].sum());result['ego_cv_points']=int(xy_valid[:,0].sum())
    result['neighbor_cv_available']=False
    return result


@torch.no_grad()
def evaluate(model,corpus,seed=20260927,sampling_steps=20,batch=8,limit=None,device='cuda'):
    prior=model.training;model.eval();rows=[];indices=list(range(len(corpus)))[:limit]
    for offset in range(0,len(indices),batch):
        scenes=[corpus[i] for i in indices[offset:offset+batch]];g,y,v=batch_scenes(scenes,device);noise=evaluation_noise(scenes,seed,device)
        plans={'all_hidden':model.sample(noise,g,sampling_steps=sampling_steps)}
        known=v.clone();known[:,0]=False
        plans['ego_hidden']=model.sample_conditional(noise,g,y,known,sampling_steps)
        # Every scene with at least one labeled neighbor supplies a useful diagnostic task.
        selected=[];neighbor_known=v.clone();neighbor_valid=torch.zeros_like(v)
        for i in range(len(scenes)):
            eligible=torch.where(v[i,1:,:,:2].any((-1,-2)))[0]+1
            slot=int(eligible[0]) if len(eligible) else -1;selected.append(slot)
            if slot>=0:neighbor_known[i,slot]=False;neighbor_valid[i,slot]=v[i,slot]
        plans['neighbor_hidden']=model.sample_conditional(noise,g,y,neighbor_known,sampling_steps)
        for i,s in enumerate(scenes):
            row={'token':s.token,'log':s.log,'status':'ok','selected_neighbors':int(g.active_actor_mask[i,1:].sum()),
                 'neighbors_with_labels':int(v[i,1:,:,:2].any((-1,-2)).sum()),'conditional_neighbor_available':selected[i]>=0,
                 'scope':'structured_current_GT_state_mechanism_not_camera_PDMS'}
            for name,p in plans.items():
                selected_valid=v[i:i+1].clone()
                if name=='ego_hidden':selected_valid[:,1:]=False
                elif name=='neighbor_hidden':selected_valid=neighbor_valid[i:i+1]
                metrics=metric_sums(p[i:i+1],y[i:i+1],selected_valid,g.boxes[i:i+1],g.ego_state[i:i+1,1:3])
                row.update({name+'_'+k:value for k,value in metrics.items()})
            result=execution_from_joint(plans['all_hidden'][i:i+1])
            row['joint_ego_execution_max_error_m']=float((result['executed_ego_xyyaw'][...,:2]-plans['all_hidden'][i:i+1,0,:,:2]).abs().max())
            # Same sample, same timestamps. Radius-distance proxy, not oriented-box collision.
            positions=plans['all_hidden'][i,:,:,:2];distance=(positions[0,None]-positions[1:]).norm(dim=-1)
            active=g.active_actor_mask[i,1:]
            row['same_sample_min_ego_neighbor_distance_m']=float(distance[active].min()) if active.any() else None
            rows.append(row)
    model.train(prior);return rows


def summarize(rows):
    result={'scenes':len(rows),'failed':sum(r['status']!='ok' for r in rows),'scope':'structured_current_GT_state_mechanism_not_camera_PDMS'}
    for task in ('all_hidden','ego_hidden','neighbor_hidden'):
        for actor in ('ego','neighbor'):
            for group in ('all','dynamic','static'):
                k=f'{task}_{actor}_{group}';count=sum(r[k+'_points'] for r in rows);terminal=sum(r[k+'_fde_count'] for r in rows)
                result[k+'_ADE_m']=sum(r[k+'_error_sum'] for r in rows)/count if count else None
                result[k+'_stationary_ADE_m']=sum(r[k+'_stationary_sum'] for r in rows)/count if count else None
                result[k+'_valid_points']=count;result[k+'_FDE_m']=sum(r[k+'_fde_sum'] for r in rows)/terminal if terminal else None
        count=sum(r[task+'_ego_cv_points'] for r in rows)
        result[task+'_ego_CV_ADE_m']=sum(r[task+'_ego_cv_error_sum'] for r in rows)/count if count else None
    result['neighbor_CV']='NOT_AVAILABLE: no current neighbor velocity in verified target cache'
    result['selected_neighbors']=sum(r['selected_neighbors'] for r in rows);result['neighbors_with_labels']=sum(r['neighbors_with_labels'] for r in rows)
    result['conditional_neighbor_scenes']=sum(r['conditional_neighbor_available'] for r in rows)
    result['execution_max_difference_m']=max(r['joint_ego_execution_max_error_m'] for r in rows)
    return result
