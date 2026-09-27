"""Frozen corpus subsets and one-time mechanism campaign preparation."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from tools.joint_local_scene_v3.data import AnnotatedCorpus
from tools.joint_local_scene_v3.runtime import build_queries
from tools.joint_local_scene_v3.budget import initialize_ledger,atomic_json


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


class CorpusSubset(torch.utils.data.Dataset):
    def __init__(self,corpus,path):
        self.corpus=corpus;record=json.loads(Path(path).read_text());check=dict(record);identity=check.pop('identity_sha256')
        if digest(check)!=identity or record['source_identity']!=corpus.manifest['identity_sha256']:raise ValueError('Subset source/identity mismatch')
        self.indices=record['indices']
        if not self.indices or len(set(self.indices))!=len(self.indices) or any(i<0 or i>=len(corpus) for i in self.indices):raise ValueError('Invalid subset indices')
        self.manifest=dict(corpus.manifest,identity_sha256=identity,subset=record)
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):return self.corpus[self.indices[i]]


def subset(corpus,path):return CorpusSubset(corpus,path) if path else corpus


def main():
    p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--holdout',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if (out/'registration.json').exists():raise FileExistsError('Campaign already registered')
    out.mkdir(parents=True,exist_ok=True)
    train=AnnotatedCorpus(a.train);holdout=AnnotatedCorpus(a.holdout)
    if len(train)!=7284 or len(holdout)!=64:raise ValueError('Unexpected fixed population')
    if {r['token'] for r in train.records}&{r['token'] for r in holdout.records} or {r['log'] for r in train.records}&{r['log'] for r in holdout.records}:raise ValueError('Split overlap')
    groups={'ego_only':[],'static_neighbors':[],'dynamic_neighbors':[]}
    for i in range(len(train)):
        s=train[i];g=s.graph;v=s.feature_valid[0,1:,:,:2].all(-1)
        distance=(s.future[0,1:,:,:2]-g.boxes[0,1:,None,:2]).norm(dim=-1)
        key='ego_only' if not g.active_actor_mask[0,1:].any() else 'dynamic_neighbors' if ((distance>=1.)&v).any() else 'static_neighbors'
        groups[key].append(i)
    for i in range(len(holdout)):holdout[i].validate()
    for group in groups.values():group.sort(key=lambda i:hashlib.sha256(('fixed_train64:'+train.records[i]['token']).encode()).digest())
    chosen=[]
    for key,count in [('ego_only',16),('static_neighbors',16),('dynamic_neighbors',32)]:chosen+=groups[key][:count]
    if len(chosen)!=64:raise ValueError('Insufficient diagnostic strata; do not silently change quotas')
    chosen.sort(key=lambda i:hashlib.sha256(('order:'+train.records[i]['token']).encode()).digest())
    manifest={'source_identity':train.manifest['identity_sha256'],'rule':'hash within ego_only16/static_neighbors16/dynamic_neighbors32, dynamic iff a valid neighbor displacement >=1m; data-only diagnostic selection, never graph selection','indices':chosen,'tokens':[train.records[i]['token'] for i in chosen]}
    manifest['identity_sha256']=digest(manifest);atomic_json(out/'train64.json',manifest)
    diag=CorpusSubset(train,out/'train64.json')
    for name,data in [('train64',diag),('holdout',holdout)]:atomic_json(out/(name+'_queries.json'),build_queries(data))
    # 4 formal runs *64 epochs + paired small-set512 + 8 real resume + two paired repair allowances and startup repairs.
    update_cap=4*14592+2*512+8+2*(2*512)+16
    initialize_ledger(out/'budget_ledger.json',gpu_hours=48,synthetic_updates=0,real_updates=update_cap)
    registration={'base_sha':'afa599762de491d4a9f87310aae41a4fc8d5e76b','gpu_hour_cap':48,'real_update_cap':update_cap,'real_update_cap_formula':'4*14592 + 2*512 + 8 + 2*(2*512) + 16','train_scenes':len(train),'holdout_scenes':len(holdout),'train_identity':train.manifest['identity_sha256'],'holdout_identity':holdout.manifest['identity_sha256'],'train64_identity':manifest['identity_sha256'],'train64_strata':{'ego_only':16,'static_neighbors':16,'dynamic_neighbors':32},'token_overlap':0,'log_overlap':0,'all_source_artifact_hashes_verified':True,'raw_log_population_available':False,'source_population_filtered':True,'batch':32,'updates_per_epoch':228,'initial_epochs':32,'initial_updates':7296,'max_epochs':64,'scheduler_updates':14592,'epoch_milestones':[0,1,2,4,8,16,24,32,48,64],'small_batch':16,'small_updates':512,'small_milestones':[0,64,128,256,512],'training_seeds':[42,43],'sampling_seed':20260927,'sampling_steps':20,'endpoint_K':8,'precision':'float32, TF32 disabled, deterministic algorithms/math attention','extension_rule':'At epoch32, extend BOTH if at least one preregistered metric in either arm improves >1% from24 to32 AND >1% from16 to24, without confirmed overfit. Overfit proxy: any all-hidden ego/neighbor holdout ADE worsens >10% in both intervals while same train64 metric improves. Primary metrics: ego/neighbor all-hidden ADE, ego/neighbor conditional ADE on queries with known_other_xy_points>0. Same fixed single-sample protocol. Budget feasibility required; no PDMS use.','selection':'Common endpoint only, no arm-specific best checkpoint','second_seed':'Same final length and full64-epoch scheduler; run after primary pair and diagnostics if budget permits','condition_projection':'frozen, no Qwen features','legacy_campaign':'sealed unchanged; real0 synthetic16','legacy_v2':'sealed step1887','visual_training':'NOT_AUTHORIZED','navtest':'NOT_AUTHORIZED'}
    atomic_json(out/'registration.json',registration);print(json.dumps(registration,indent=2))


if __name__=='__main__':main()
