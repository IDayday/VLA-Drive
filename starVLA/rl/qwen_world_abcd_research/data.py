"""Full existing four-framework TRAIN, with its original heldout-log embargo.

Only a small token manifest is created. The current observation records/images
remain at their existing sealed TRAIN root; no old103k NAVTRAIN reader is used.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re

from starVLA.rl.qwen_world_compare.contracts import file_sha,fingerprint
from starVLA.rl.simwam_recipe.vendor.samplers import ResumableEpochSampler

SCHEMA='qwen_world_abcd_full_TRAIN_tokens_v1'
EXPECTED_TRAIN_COUNT=90890
EXPECTED_TRAIN_PHYSICAL_LOGS=147
POPULATION=Path('/mnt/project/flow-safe-improvement-20261013/population/manifest.json')


def _contract(train_root,partition_path,population_path):
    train_root=Path(train_root).resolve();partition_path=Path(partition_path).resolve()
    population_path=Path(population_path).resolve()
    train=json.loads((train_root/'manifest.json').read_text())
    seal=json.loads((train_root/'COMPLETE').read_text())
    roles=json.loads(partition_path.read_text());pop=json.loads(population_path.read_text())
    if (train['cohort']!='TRAIN' or seal['manifest_sha256']!=file_sha(train_root/'manifest.json')
            or roles['population_sha256']!=file_sha(population_path)
            or train['population_sha256']!=roles['population_sha256']
            or train['role_partition_sha256']!=fingerprint(roles)):
        raise ValueError('Original sealed TRAIN/partition/population changed')
    tokens=train['tokens'];legal=set(roles['train_tokens'])
    if (len(tokens)!=EXPECTED_TRAIN_COUNT or train['count']!=len(tokens)
            or len(set(tokens))!=len(tokens) or set(tokens)!=legal):
        raise ValueError('Full legal TRAIN90890 census/order required')
    forbidden=set(roles['dev_tokens'])|set(roles['navtest_tokens'])|set(roles['embargoed_train_tokens'])
    physical=lambda t:re.sub(r'_\d+_\d+$','',pop['token_logs'][t])
    train_logs={physical(t) for t in tokens};forbidden_logs={physical(t) for t in forbidden}
    if (legal&forbidden or train_logs&forbidden_logs
            or len(train_logs)!=EXPECTED_TRAIN_PHYSICAL_LOGS):
        raise ValueError('DEV/embargoed/NAVTEST token or physical log entered full TRAIN')
    return tokens,pop['token_logs'],dict(schema=SCHEMA,count=len(tokens),tokens=tokens,
        train_root=str(train_root),train_manifest_sha256=file_sha(train_root/'manifest.json'),
        train_complete_sha256=file_sha(train_root/'COMPLETE'),
        role_partition=str(partition_path),role_partition_sha256=fingerprint(roles),
        role_partition_file_sha256=file_sha(partition_path),population_manifest=str(population_path),
        population_sha256=file_sha(population_path),seed=42,mode='uniform',
        physical_log_count=len(train_logs),DEV_tokens_excluded=len(roles['dev_tokens']),
        embargoed_train_tokens_excluded=len(roles['embargoed_train_tokens']),
        NAVTEST_tokens_excluded=len(roles['navtest_tokens']),
        actor_inputs='current L/F/R images,ego,navigation only; WorldSceneData.observation',
        observations_copied=False,old103288_NAVTRAIN_not_used=True,
        ordering='existing sealed TRAIN order; source ResumableEpochSampler seed42+epoch and exact world padding')


def write_full_train_manifest(output,train_root,*,partition_path=None,population_path=POPULATION):
    train_root=Path(train_root)
    partition_path=Path(partition_path) if partition_path else train_root.parent/'DATA_PARTITION.json'
    _,_,body=_contract(train_root,partition_path,population_path)
    result=dict(body,sha256=fingerprint(body));output=Path(output)
    if output.exists():
        if json.loads(output.read_text())!=result:
            raise FileExistsError('Existing immutable fullTRAIN manifest differs')
        return result
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_name(output.name+'.writing-'+str(os.getpid()))
    tmp.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,output)
    return result


class FullTRAINTokenStream:
    """WorldRLTokenStream-compatible, fullTRAIN-only recoverable cursor stream."""
    def __init__(self,manifest,world):
        self.path=Path(manifest);self.manifest=json.loads(self.path.read_text())
        body=dict(self.manifest);seal=body.pop('sha256',None)
        if seal!=fingerprint(body) or body.get('schema')!=SCHEMA:
            raise ValueError('FullTRAIN stream manifest checksum/schema changed')
        tokens,logs,expected=_contract(body['train_root'],body['role_partition'],body['population_manifest'])
        if body!=expected:
            raise ValueError('FullTRAIN stream does not match its original sealed sources')
        self.world=int(world)
        if self.world<=0:raise ValueError('Positive world required')
        self.tokens=tokens;self.logs=logs;self.sha=file_sha(self.path)
        self.epoch=-1;self.order=[]
        self.sampler=ResumableEpochSampler(tokens,seed=42,batch_size=1,num_processes=self.world)
        self.epoch_size=math.ceil(len(tokens)/self.world)*self.world
        self.identity=dict(schema=SCHEMA,manifest_sha256=self.sha,scene_count=len(tokens),world=self.world,
            train_manifest_sha256=body['train_manifest_sha256'],role_partition_sha256=body['role_partition_sha256'],
            physical_log_count=body['physical_log_count'],physical_log_disjoint=True,
            original_source_epoch_padding=True,ordering=body['ordering'],
            DEV_and_embargoed_and_NAVTEST_excluded=True,old103288_NAVTRAIN_not_used=True)

    def _order(self,epoch):
        if epoch!=self.epoch:
            self.sampler.set_epoch(epoch);order=list(self.sampler)
            self.order=(order*math.ceil(self.epoch_size/len(order)))[:self.epoch_size]
            self.epoch=epoch
        return self.order

    def at(self,global_cursor,rank):
        if global_cursor<0 or global_cursor%self.world or not 0<=rank<self.world:
            raise ValueError('Invalid global scene cursor or rank')
        epoch,offset=divmod(global_cursor,self.epoch_size);index=offset+rank
        return self.tokens[self._order(epoch)[index]],dict(epoch=epoch,index=index,padding=index>=len(self.tokens))

    def token_at(self,position):
        if position<0:raise ValueError('Negative scene position')
        epoch,index=divmod(position,self.epoch_size)
        return self.tokens[self._order(epoch)[index]]

    def token(self,update,rank):
        return self.at(update*self.world,rank)[0]
