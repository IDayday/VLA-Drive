"""Versioned plan/target rows over the unchanged compact S0 feature bank."""
from collections import OrderedDict
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from starVLA.model.modules.decision_local.minimal_repair_contract import CurrentFeatures,PlanInputs,RepairTargets,SCHEMA
from starVLA.model.modules.decision_local.consequence_contract import PredictionQueries,OracleRelations
from starVLA.model.modules.decision_local.recovery_plans import RecoveryConfig,select_plan
from .decision_local import FrozenConsequenceData,read


class RecoveryPlanData(FrozenConsequenceData):
    def __init__(self,features,labels,*,fold_partition=None,fold=None,require_complete=True,reader='mmap_tensors',label_schema=SCHEMA):
        self.features,self.labels=Path(features),Path(labels);self.reader=reader
        if (self.labels/'CORRECTNESS_HOLD.json').exists():raise ValueError('This preserved label version has a correctness hold')
        self.feature_identity=read(self.features/'identity.json');self.label_identity=read(self.labels/'identity.json')
        if self.label_identity['schema']!=label_schema or self.label_identity['feature_identity']!=self.feature_identity['identity']:
            raise ValueError('Foreign versioned repair labels or frozen feature identity')
        if require_complete and not all((p/'COMPLETE.json').is_file() for p in (self.features,self.labels)):
            raise ValueError('Complete unchanged population required')
        self.index=self.label_identity['index'];allfeatures={r['token']:r for r in self.feature_identity['index']}
        if any(allfeatures.get(r['token'])!=r for r in self.index):raise ValueError('Feature/label token or log mismatch')
        self.locations={};wanted={r['token'] for r in self.index}
        if len(wanted)!=len(self.index):raise ValueError('Duplicate scene')
        for path in sorted((self.features/'records').glob('*.json')):
            rec=read(path)
            if rec['identity']!=self.feature_identity['identity']:raise ValueError('Foreign feature chunk')
            for offset,token in enumerate(rec['tokens']):
                if token in wanted:self.locations[token]=(path.stem,offset)
        if set(self.locations)!=wanted:raise ValueError('Missing features')
        self.fold_identity=None
        if fold_partition is not None:
            partition=read(fold_partition)
            if fold not in (0,1) or partition['features']!=self.feature_identity['identity']:raise ValueError('Invalid frozen log partition')
            expected=[r for r in self.feature_identity['index'] if partition['log_fold'][r['log']]==fold]
            if self.index!=expected:raise ValueError('Formal repair population must equal the entire registered fold in original order')
            self.fold_identity={'fold':fold,'partition_sha256':hashlib.sha256(Path(fold_partition).read_bytes()).hexdigest(),
                                'scenes':len(expected),'logs':len({r['log'] for r in expected})}
        self.handles=OrderedDict();self.tensor_chunks={};self._labels={}
        self.recovery=RecoveryConfig(**self.label_identity['recovery'])

    def _scene_labels(self,token):
        if token not in self._labels:
            record=read(self.labels/'records'/(token+'.json'))
            if record['identity']!=self.label_identity['identity']:raise ValueError('Foreign target record')
            path=self.labels/'scenes'/(token+'.npz')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Changed repair targets')
            with np.load(path,allow_pickle=False) as f:self._labels[token]={k:torch.from_numpy(f[k].copy()) for k in f.files}
        return self._labels[token]

    def selection(self,ids,epoch=0,evaluation=False,distribution='NATURAL'):
        if distribution not in ('NATURAL','RECOVERY'):raise ValueError('Unknown evaluation distribution')
        rows=[]
        for i in ids:
            token=self.index[i]['token'];l=self._scene_labels(token);n=l['plan_in_physical'].shape[0]
            if evaluation:
                anchor=0
                plan=0 if distribution=='NATURAL' else 1+(int(hashlib.sha256((token+':fixed-recovery-v1').encode()).hexdigest()[:8],16)%4)
            else:anchor,plan=select_plan(token,epoch,n,self.recovery)
            rows.append((token,anchor,plan,bool(l['target_eligible'][anchor,plan]),int(l['case_status'][anchor,plan])))
        return rows

    def batch(self,ids,device,*,epoch=0,evaluation=False,distribution='NATURAL'):
        rows=[]
        for token,anchor,plan,eligible,status in self.selection(ids,epoch,evaluation,distribution):
            chunk,offset=self.locations[token];f=self._chunk(self.features,chunk,self.feature_identity['identity']);l=self._scene_labels(token)
            row={k:f[k][offset] for k in ('W','H_A','state')}
            for k in ('plan_in_physical','plan_in_encoded','poses','in_grid','kinematic_valid','relations','valid','common_valid','repair_target','target_eligible','case_status'):
                row[k]=l[k][anchor,plan]
            row.update({k:l[k] for k in ('ego','future_valid','body')})
            row.update(anchor_index=torch.tensor(anchor),plan_index=torch.tensor(plan))
            rows.append(row)
        result={}
        for k in rows[0]:
            t=torch.stack([r[k] for r in rows])
            if torch.device(device).type=='cuda':t=t.pin_memory()
            result[k]=t.to(device,non_blocking=True)
        return result


def repair_inputs_targets(batch):
    current=CurrentFeatures.from_mapping({k:batch[k] for k in CurrentFeatures.__dataclass_fields__})
    physical=batch['plan_in_physical'];T=physical.shape[1]
    times=torch.arange(1,T+1,device=physical.device)*.5
    plan=PlanInputs(physical,batch['plan_in_encoded'],times)
    queries=PredictionQueries(batch['poses'],times,batch['in_grid'],batch['kinematic_valid'])
    relation=torch.cat((batch['relations'][:,0],batch['relations'][:,2,...,1:]),-1)
    valid=torch.cat((batch['valid'][:,0],batch['valid'][:,2,...,1:]),-1)
    kinetic=batch['kinematic_valid'].clone();kinetic[:,0]=True
    valid=valid&kinetic[...,None,None]
    targets=RepairTargets(batch['ego'],batch['repair_target'],batch['future_valid']&batch['target_eligible'][:,None],
                          batch['case_status'],relation,valid,batch['target_eligible'])
    return current,plan,queries,targets


def repair_oracle(batch,source):
    if source not in ('CV','FUT'):raise ValueError('Explicit privileged source required')
    slot=1 if source=='CV' else 2
    values=torch.cat((batch['relations'][:,0],batch['relations'][:,slot,...,1:]),-1)
    valid=torch.cat((batch['common_valid'],batch['common_valid'][...,1:]),-1)
    kinetic=batch['kinematic_valid'].clone();kinetic[:,0]=True
    return OracleRelations(values,valid&kinetic[...,None,None],'CV' if source=='CV' else 'FUTURE')
