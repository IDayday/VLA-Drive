"""Teacher-only data/metrics. Never imported by camera inference."""
import json
from pathlib import Path
import torch
from starVLA.model.modules.trajectory_mae.tokenizer import TeacherInputs
from starVLA.model.modules.vehicle_joint.initialization import identity_hash


class TeacherDataset:
    def __init__(self,root,split='train',limit=0):
        self.root=Path(root);self.identity=json.loads((self.root/'identity.json').read_text())
        if self.identity['schema']!='foresight_gt_vehicle_mae_v1':raise ValueError('Teacher data schema mismatch')
        self.index=json.loads((self.root/(split+'_index.json')).read_text())
        if limit:self.index=self.index[:limit]
        if not self.index:raise ValueError('Empty teacher dataset')
        self.index_hash=identity_hash(self.index)
    def __len__(self):return len(self.index)
    def __getitem__(self,i):
        item=torch.load(self.root/'records'/(self.index[i]['token']+'.pt'),map_location='cpu',weights_only=True)
        if item['identity']!=self.identity['identity'] or item['token']!=self.index[i]['token']:raise ValueError('Teacher record identity mismatch')
        return item['record']
    def batch(self,ids,device):
        rows=[self[i] for i in ids]
        return {k:torch.stack([r[k] for r in rows]).to(device) for k in rows[0]}


def inputs(batch,target,visible):
    return TeacherInputs(batch['current'],batch['active'],batch['future'],visible,target,batch['navigation'],batch['ego_state'])


def fixed_queries(dataset):
    queries=[]
    for i,row in enumerate(dataset.index):
        r=dataset[i]
        for actor in torch.where(r['active'] & r['point_valid'].any(-1))[0].tolist():
            peer=r['point_valid'].clone();peer[actor]=False
            queries.append({**row,'index':i,'target':actor,'known_peer_points':int(peer.sum()),
                            'valid_points':int(r['point_valid'][actor].sum()),'role':'ego' if actor==0 else 'vehicle'})
    return queries


@torch.no_grad()
def evaluate_teacher(model,dataset,queries,device,batch_size=64):
    model.eval();rows=[]
    for start in range(0,len(queries),batch_size):
        q=queries[start:start+batch_size];batch=dataset.batch([x['index'] for x in q],device)
        target=torch.tensor([x['target'] for x in q],device=device);arange=torch.arange(len(q),device=device)
        visible=batch['point_valid'].clone();visible[arange,target]=False
        full=model(inputs(batch,target,visible))['xy']
        empty=model(inputs(batch,target,torch.zeros_like(visible)))['xy']
        labels=batch['future'][arange,target];valid=batch['point_valid'][arange,target]
        current=batch['current'][arange,target,:2]
        for k,query in enumerate(q):
            v=valid[k];count=int(v.sum());indices=torch.where(v)[0]
            movement=(labels[k,v]-current[k]).norm(dim=-1).max().item()
            row={**query,'label_group':'complete' if count==8 else 'partial',
                 'motion_group':'stationary' if movement<=.5 else 'moving','failure':None}
            for name,p in [('full_peer',full),('current_only',empty),('stationary',current[:,None].expand(-1,8,-1))]:
                distance=(p[k,v]-labels[k,v]).norm(dim=-1)
                if not torch.isfinite(distance).all():raise FloatingPointError('Nonfinite teacher metric')
                row[name+'_ADE']=float(distance.mean());row[name+'_last_valid_FDE']=float(distance[-1])
                row[name+'_endpoint_FDE']=float(distance[-1]) if v[-1] else None
            rows.append(row)
    summary={'queries':len(rows),'scenes':len(dataset),'failures':0,'groups':{}}
    for role in ('all','ego','vehicle'):
      for condition in ('all','with_peer','without_peer'):
        subset=[r for r in rows if (role=='all' or r['role']==role) and
                (condition=='all' or (r['known_peer_points']>0)==(condition=='with_peer'))]
        key=role+'_'+condition
        summary['groups'][key]={'queries':len(subset)}
        for name in ('full_peer_ADE','current_only_ADE','stationary_ADE'):
            summary['groups'][key][name]=sum(r[name] for r in subset)/len(subset) if subset else None
    return rows,summary
