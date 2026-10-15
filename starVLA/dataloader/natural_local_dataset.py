"""Local facts and strict repair targets over one registered natural-error pool."""
import hashlib,json
from pathlib import Path
import torch
from .recovery_plan_dataset import RecoveryPlanData
from .decision_local import read
from starVLA.model.modules.decision_local.natural_error_contract import SCHEMA,verify_feature_base


class NaturalLocalData(RecoveryPlanData):
    def __init__(self,features,labels,*,reader='mmap_tensors',require_complete=True):
        super().__init__(features,labels,reader=reader,require_complete=require_complete,label_schema=SCHEMA)
        self.repair_review=None
        meta=self.label_identity
        if meta['selection']!='80%native10%lateral10%time_hash_v1':raise ValueError('Changed registered plan exposure')
        self.fold_identity={'role':meta['role'],'source_audit_identity':meta['source_audit_identity'],
            'training_index_identity':meta['index_identity'],'scope':meta['scope']}
        if meta['scope']!='engineering_historical_anchor_only':
            anchor=read(meta['anchor_record']);verify_feature_base(self.feature_identity,anchor)
            source=read(meta['source_audit']);registered=read(Path(meta['source_audit']).parent/(meta['role']+'.json')) if meta['role']!='dev' else None
            if source['identity']!=meta['source_audit_identity']:raise ValueError('Population roles changed')
            if registered is not None and registered!=self.index:raise ValueError('The entire registered role population is required')

    def bind_repair_execution_review(self,path):
        """Label-only overlay; preserve old bytes, local facts and scene order."""
        if self.repair_review is not None:raise ValueError('Execution review already bound')
        if (Path(path).parent/'CORRECTNESS_HOLD.json').exists():raise ValueError('Preserved execution review has a correctness hold')
        review=read(path);payload={k:v for k,v in review.items() if k!='identity'}
        actual=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if (review.get('schema')!='natural_repair_execution_review_v1' or review['identity']!=actual or
            review['labels_identity']!=self.label_identity['identity'] or
            review.get('all_label_repairs_checked') is not True or
            review.get('all_admitted_repairs_independently_checked') is not True):
            raise ValueError('Complete independent review for these exact labels required')
        wanted={r['token'] for r in self.index}
        if any(r['token'] not in wanted for r in review['cases'].values()):raise ValueError('Foreign review scene')
        self.repair_review=review;self._review_by_token={}
        for key,row in review['cases'].items():self._review_by_token.setdefault(row['token'],{})[key]=row
        self._labels.clear()

    def _scene_labels(self,token):
        fresh=token not in self._labels
        result=super()._scene_labels(token)
        if fresh and self.repair_review is not None:
            reviewed=self._review_by_token.get(token,{})
            expected={f'{token}:{a}:{p}' for a,p in torch.nonzero((result['case_status']==1)&result['target_eligible']).tolist()}
            if set(reviewed)!=expected:raise ValueError('Review omitted or invented a strict REPAIR')
            record=read(self.labels/'records'/(token+'.json'))
            for key,row in reviewed.items():
                a,p=row['anchor_index'],row['plan_index']
                if key!=f'{token}:{a}:{p}' or row['label_sha256']!=record['sha256'] or type(row['admitted']) is not bool:
                    raise ValueError('Review target identity mismatch')
                if row['admitted']!=all(row['checks'].values()):raise ValueError('Review checks disagree with admission')
                if not row['admitted']:result['target_eligible'][a,p]=False
        return result

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
