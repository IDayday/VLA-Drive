"""Local facts and strict repair targets over one registered natural-error pool."""
import hashlib
from pathlib import Path
import torch
from .recovery_plan_dataset import RecoveryPlanData
from .decision_local import read
from starVLA.model.modules.decision_local.natural_error_contract import SCHEMA,verify_feature_base


class NaturalLocalData(RecoveryPlanData):
    def __init__(self,features,labels,*,reader='mmap_tensors',require_complete=True):
        super().__init__(features,labels,reader=reader,require_complete=require_complete,label_schema=SCHEMA)
        meta=self.label_identity
        if meta['selection']!='80%native10%lateral10%time_hash_v1':raise ValueError('Changed registered plan exposure')
        self.fold_identity={'role':meta['role'],'source_audit_identity':meta['source_audit_identity'],
            'training_index_identity':meta['index_identity'],'scope':meta['scope']}
        if meta['scope']!='engineering_historical_anchor_only':
            anchor=read(meta['anchor_record']);verify_feature_base(self.feature_identity,anchor)
            source=read(meta['source_audit']);registered=read(Path(meta['source_audit']).parent/(meta['role']+'.json')) if meta['role']!='dev' else None
            if source['identity']!=meta['source_audit_identity']:raise ValueError('Population roles changed')
            if registered is not None and registered!=self.index:raise ValueError('The entire registered role population is required')

    def selection(self,ids,epoch=0,evaluation=False,distribution='NATURAL'):
        if evaluation:return super().selection(ids,epoch,evaluation,distribution)
        rows=[]
        for i in ids:
            token=self.index[i]['token'];l=self._scene_labels(token);n=l['plan_in_physical'].shape[0]
            key=int(hashlib.sha256(f'{token}:{epoch}:natural-input-v1'.encode()).hexdigest()[:16],16)
            anchor=epoch%n;slot=key%10
            plan=0 if slot<8 else (1+(key//10)%2 if slot==8 else 3+(key//10)%2)
            rows.append((token,anchor,plan,bool(l['target_eligible'][anchor,plan]),int(l['case_status'][anchor,plan])))
        return rows

    def batch(self,ids,device,*,epoch=0,evaluation=False,distribution='NATURAL'):
        batch=super().batch(ids,device,epoch=epoch,evaluation=evaluation,distribution=distribution)
        selected=self.selection(ids,epoch,evaluation,distribution)
        for key in ('positive_known','negative_known'):
            value=torch.stack([self._scene_labels(token)[key][anchor,plan] for token,anchor,plan,_,_ in selected])
            if torch.device(device).type=='cuda':value=value.pin_memory()
            batch[key]=value.to(device,non_blocking=True)
        batch['local_evidence_protocol']=True
        return batch
