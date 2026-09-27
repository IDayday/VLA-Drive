"""Local graph corpus and metrics: frozen current inputs first, targets only on loss side."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import torch
from starVLA.model.modules.joint_world.local_cache import load_payload,pack_payload
from starVLA.model.modules.joint_world.local_graph import stack_graphs
from starVLA.model.modules.joint_world.local_targets import local_targets
from starVLA.model.modules.joint_world.local_masks import stable_noise,task_masks
from starVLA.model.modules.structured_world.contracts import WorldTargets
from tools.local_interaction_mask_v2.foundation import ego_label


def cached_worker_init(worker_id):
    # A local sample contains many small contract tensors. file_descriptor IPC
    # otherwise exhausts RLIMIT_NOFILE even with bounded resident/prefetch sizes.
    torch.multiprocessing.set_sharing_strategy('file_system')


class LocalCorpus(torch.utils.data.Dataset):
    def __init__(self,cache,targets,meta_root,resident=64):
        self.root=Path(cache);self.targets=Path(targets);self.meta=Path(meta_root);self.resident=resident;self.memo=OrderedDict()
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        if not self.manifest['complete'] or self.manifest['failed']:raise ValueError('Local training needs a complete current cache')
        self.records=self.manifest['records'];self.identity=self.manifest['identity']
        if self.identity['schema_version']!=2 or self.identity['kind']!='local_current_graph_v2':raise ValueError('Legacy graph cache forbidden')
        if not 1<=resident<=256:raise ValueError('Use bounded resident memory')
        if len({r['token'] for r in self.records})!=len(self.records):raise ValueError('Duplicate local scenes')
        self.label_fingerprint=hashlib.sha256()
        for r in self.records:
            token=r['token']
            for p in (self.targets/'targets'/(token+'.pt'),self.meta/(token+'.pkl')):
                self.label_fingerprint.update(token.encode()+hashlib.sha256(p.read_bytes()).digest())
        self.label_fingerprint=self.label_fingerprint.hexdigest()
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        if i not in self.memo:
            record=self.records[i];payload=load_payload(self.root,record,self.manifest)
            current=pack_payload(payload,self.identity)
            # The graph has already been fixed and validated before label files open.
            token=record['token'];target=WorldTargets(**torch.load(self.targets/'targets'/(token+'.pt'),weights_only=True))
            if target.overflow:raise ValueError('Need full ROI targets')
            action=ego_label(self.meta/(token+'.pkl'))
            ego_xy=(action[:,:2]*torch.tensor([8.805105,2.277741])+torch.tensor([10.172484,.360762]))[None]
            selector=self.identity['selector']
            xy,valid,mapping=local_targets(payload['current_prediction'],current['local_graph'],[target],ego_xy,
                selector['match_max_distance_m'],selector['match_require_class'])
            # Label-side anchor ONLY for posthoc dynamic/static grouping. A
            # detection centre error must not turn a stationary GT into a mover.
            target_current_xy=torch.zeros_like(current['current_xy'])
            for association in mapping[0]['assignments']:
                if association['used_for_local_loss']:
                    target_current_xy[0,association['local_slot']]=target.current_boxes[association['gt_index'],:2]
            self.memo[i]={'token':token,'payload':payload,'current':current,'xy':xy,'valid':valid,'mapping':mapping[0],
                          'action':action,'target_current_xy':target_current_xy}
            if len(self.memo)>self.resident:self.memo.popitem(last=False)
        self.memo.move_to_end(i);return self.memo[i]


def batch_current(samples,device='cuda'):
    current={key:torch.cat([s['current'][key] for s in samples]).to(device) for key in ('actor_features','context','context_mask','current_xy','existence')}
    current['local_graph']=stack_graphs([s['current']['local_graph'] for s in samples],device)
    return current


def task_supervision_statistics(graph, hidden, valid, tasks):
    """Label-side accounting only; never changes graph, mask sampling or loss."""
    eligible = graph.active_actor_mask & graph.predictable_actor_mask
    selected = valid & (hidden & eligible)[..., None]
    result = {}
    for code, name in enumerate(('all_hidden', 'ego_hidden', 'neighbor_hidden')):
        take = tasks['actual'] == code
        supervision = selected[take]
        result[name] = {'scenes': int(take.sum()), 'hidden_actors': int((hidden[take] & eligible[take]).sum()),
            'actors_with_any_label': int(supervision.any(-1).sum()),
            'valid_xy_coordinates': int(supervision.sum())*2,
            'ego_xy_coordinates': int(supervision[:, :1].sum())*2,
            'neighbor_xy_coordinates': int(supervision[:, 1:].sum())*2,
            'scenes_without_valid_hidden_target': int((~supervision.flatten(1).any(-1)).sum())}
    return result


def trajectory_metrics(pred,truth,valid,current_xy,target_current_xy):
    truth=torch.where(valid[...,None],truth,0.)
    prediction=torch.where(valid[...,None],pred,0.)
    distance=(prediction-truth).norm(dim=-1);stationary=(current_xy[:,:,None]-truth).norm(dim=-1)
    result={}
    for name,start,end in [('ego',0,1),('agents',1,valid.shape[1])]:
        v=valid[:,start:end];d=distance[:,start:end];s=stationary[:,start:end]
        result.update({name+'_error_sum':float(d[v].sum()),name+'_valid_points':int(v.sum()),name+'_stationary_error_sum':float(s[v].sum()),
            name+'_fde_sum':float(d[:,:,-1][v[:,:,-1]].sum()),name+'_fde_count':int(v[:,:,-1].sum()),
            name+'_stationary_fde_sum':float(s[:,:,-1][v[:,:,-1]].sum())})
        anchor=torch.where(v.any(-1)[...,None],target_current_xy[:,start:end],0.)
        movement=(truth[:,start:end]-anchor[:,:,None]).norm(dim=-1)
        moving=(movement.masked_fill(~v,0.).amax(-1)>=1.)&v.any(-1) # Posthoc GT labels only.
        for group,mask in [('dynamic',moving),('static',~moving)]:
            take=v&mask[:,:,None]
            result[name+'_'+group+'_error_sum']=float(d[take].sum());result[name+'_'+group+'_valid_points']=int(take.sum())
            terminal=v[:,:,-1]&mask
            result[name+'_'+group+'_fde_sum']=float(d[:,:,-1][terminal].sum())
            result[name+'_'+group+'_fde_count']=int(terminal.sum())
            result[name+'_'+group+'_stationary_error_sum']=float(s[take].sum())
    return result


@torch.no_grad()
def evaluate_graph(model,corpus,seed=2037,indices=None):
    model.eval();rows=[];indices=range(len(corpus)) if indices is None else indices
    for i in indices:
        sample=corpus[i];current=batch_current([sample]);graph=current['local_graph']
        noise=stable_noise([sample['token']],graph.source_slot_ids,8,seed,device='cuda')
        xy=sample['xy'].cuda();valid=sample['valid'].cuda();target_current_xy=sample['target_current_xy'].cuda()
        row={'token':sample['token'],'status':'ok','active_neighbors':int(graph.active_actor_mask[:,1:].sum()),
            'matched_neighbors':sample['mapping']['selected_with_accepted_association'],
            'full_current_roi_targets':sample['mapping']['full_current_gt'],'full_future_points':sample['mapping']['full_valid_future_points']}
        prediction,_=model.sample(noise,**current,sampling_steps=10)
        row.update({'deploy_'+k:v for k,v in trajectory_metrics(prediction,xy,valid,current['current_xy'],target_current_xy).items()})
        # Privileged conditional diagnostics kept separate from all-hidden deployment.
        eligible=graph.trajectory_condition_mask
        known=valid&eligible[:,:,None];known[:,0]=False
        prediction,_=model.sample_conditional(noise,**current,known_xy=xy,known_mask=known,sampling_steps=10)
        ego_only=valid.clone();ego_only[:,1:]=False
        row.update({'conditional_ego_'+k:v for k,v in trajectory_metrics(prediction,xy,ego_only,current['current_xy'],target_current_xy).items()})
        neighbors=torch.where(eligible[0])[0];neighbors=neighbors[neighbors!=0]
        # Pick by stable current source identity, never by future completeness.
        if len(neighbors):
            index=neighbors[graph.source_slot_ids[0,neighbors].argmin()]
            known=valid&eligible[:,:,None];known[:,index]=False
            prediction,_=model.sample_conditional(noise,**current,known_xy=xy,known_mask=known,sampling_steps=10)
            one=torch.zeros_like(valid);one[:,index]=valid[:,index]
            row.update({'conditional_neighbor_'+k:v for k,v in trajectory_metrics(prediction,xy,one,current['current_xy'],target_current_xy).items()})
            row['conditional_neighbor_source']=int(graph.source_slot_ids[0,index]);row['conditional_neighbor_no_labels']=not bool(one.any())
        rows.append(row)
    model.train();return rows


def summarize(rows):
    result={'scenes':len(rows),'failed':sum(r['status']!='ok' for r in rows)}
    for prefix in ('deploy','conditional_ego','conditional_neighbor'):
        for actor in ('ego','agents'):
            name=prefix+'_'+actor
            total=sum(r.get(name+'_error_sum',0) for r in rows);count=sum(r.get(name+'_valid_points',0) for r in rows)
            stationary=sum(r.get(name+'_stationary_error_sum',0) for r in rows)
            result[name+'_ADE_m']=total/count if count else None
            result[name+'_stationary_ADE_m']=stationary/count if count else None
            result[name+'_valid_points']=count
            fde_count=sum(r.get(name+'_fde_count',0) for r in rows)
            result[name+'_FDE_m']=sum(r.get(name+'_fde_sum',0) for r in rows)/fde_count if fde_count else None
            for group in ('dynamic','static'):
                group_name=name+'_'+group;n=sum(r.get(group_name+'_valid_points',0) for r in rows)
                result[group_name+'_valid_points']=n
                result[group_name+'_ADE_m']=sum(r.get(group_name+'_error_sum',0) for r in rows)/n if n else None
                result[group_name+'_stationary_ADE_m']=sum(r.get(group_name+'_stationary_error_sum',0) for r in rows)/n if n else None
                terminal=sum(r.get(group_name+'_fde_count',0) for r in rows)
                result[group_name+'_FDE_m']=sum(r.get(group_name+'_fde_sum',0) for r in rows)/terminal if terminal else None
    result['active_neighbors']=sum(r['active_neighbors'] for r in rows);result['matched_neighbors']=sum(r['matched_neighbors'] for r in rows)
    result['full_current_roi_targets']=sum(r['full_current_roi_targets'] for r in rows)
    result['coverage_note']='Full raw denominators are in the independent graph audit, not silently equated to ROI targets.'
    result['motion_group_rule']='>=1m displacement from matched GT current centre at any valid future; label-side analysis only, never graph/mask selection'
    return result
