"""Finite two-method safety campaign; reuse existing native dispatch/evaluation.

Training is independent fresh50k from IL53k. The original campaign, training
and inference snapshots stay untouched. No automatic failed-worker retry.
"""
import argparse
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import time
import traceback
from starVLA.rl.qwen_sourceflow.safe_recipe import ROOT
from starVLA.rl.qwen_sourceflow.safe_admission import TASK, require_dispatch
from starVLA.rl.qwen_sourceflow.resources import allowed_hosts, available
from starVLA.rl.qwen_sourceflow.loading import inspect_il
from starVLA.rl.qwen_sourceflow.contracts import file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json
from scripts.analysis.qwen_sourceflow.campaign import (Campaign, read, immutable_source,
    formal_endpoint, scene_stream_hash, MILESTONES)


class SafeCampaign(Campaign):
    def admitted_hosts(self):
        policy=read('/mnt/project/server_dispatch_policy.json')
        return [h for h in allowed_hosts() if h in policy['task_authorizations'][TASK]['allowed_hosts']]

    def reserved(self):
        # One evaluation slot can share a qualified39GB training process.
        # Never assign a third own process to a device, or cohost two new RLs.
        uuids=set()
        for task in self.tasks():
            if not self.done(task['name']):
                for spec in task['workers']:
                    if not spec['module'].endswith('train_safe_rl'):uuids.update(spec['gpu_uuids'])
        return uuids

    def admitted_routes(self,min_free_MiB):
        routes,inventory=super().admitted_routes(min_free_MiB)
        allowed=set(self.admitted_hosts())
        return [r for r in routes if r['host'] in allowed],inventory

    def ensure_formal(self,key,parent):
        name=key+'_formal50k';run=self.root/'runs'/('RL_'+key.upper()+'_seed42')
        if self.task_path(name).exists():return self.done(name),run
        blocked=set(self.reserved())
        for task in self.tasks():
            if not self.done(task['name']):
                for spec in task['workers']:blocked.update(spec['gpu_uuids'])
        routes,inventory=available(blocked,min_free_MiB=41500)
        groups=defaultdict(list)
        for r in routes:
            if r['host'] in self.admitted_hosts():groups[r['host']].append(r)
        full=[v for v in groups.values() if len(v)==8]
        atomic_json(self.control/('admission_'+key+'.json'),dict(at=time.time(),inventory=inventory,full_hosts=[v[0]['host'] for v in full]))
        if not full:return False,run
        preferred='training-vla-zt' if key=='wla_adv' else 'training-vlawm-tsinghua-zt4'
        chosen=max(full,key=lambda rows:(rows[0]['host']==preferred,sum(r['free_MiB'] for r in rows)))
        args=['--config',f'configs/qwen_sourceflow/rl_{key}.yaml',
            '--manifest','/mnt/project/simwam-source-process-20261018/source_manifest.json',
            '--il-checkpoint',parent['path'],'--parent-selection',str(self.selection),'--output',str(run)]
        if key=='scadv3':args+=['--safety-budget',str(self.root/'calibration/SFT53k_public_hard512_v1/budget.json')]
        self.submit(name,'scripts.analysis.qwen_sourceflow.train_safe_rl',args,chosen,self.training,
                    distributed=True,minimum_free_MiB=41500)
        return False,run

    def tick(self):
        require_dispatch();reg=read(self.root/'experiment_registration.json')
        proof=read(self.root/'qualification/QUALIFIED.json')
        if proof['status']!='PASS' or proof['formal_training_source_sha']!=self.training['sha']:
            raise ValueError('Training source lacks finite qualification/equivalence receipt')
        parent=inspect_il(reg['parent'],parent_selection=self.selection)
        complete={};dev={};runs={};held={}
        for key in reg['variants']:
            done,run=self.ensure_formal(key,parent);runs[key]=run;dev[key]={}
            marker=self.control/'HELD'/f'{key}.json'
            if done and marker.exists():
                held[key]=read(run/'progress.json')
                complete[key]=dict(status='HELD_BY_PREREGISTERED_RULE',updates=held[key]['updates'],checkpoint=held[key]['checkpoint'])
            elif done:complete[key]=formal_endpoint(run)
            for update in MILESTONES:
                export=self.ensure_export(key,update,run)
                if export is None:continue
                summary=self.ensure_evaluation(f'RL_{key}_update{update:06d}_DEV',parent,'DEV',export=export)
                if summary is not None:
                    dev[key][update]=dict(summary=str(summary),summary_sha256=file_sha(summary),PDMS=read(summary)['PDMS'],export=str(export))
            # Preregistered severe-degradation guard, based on full DEV only.
            previous=False;alarms=[]
            from scripts.analysis.qwen_sourceflow.analyze_paired import read as read_evaluation
            for update,item in sorted(dev[key].items()):
                alarm_path=self.control/'DEV_guard'/f'{key}_{update}.json'
                if alarm_path.exists():alarm=read(alarm_path)
                else:
                    import numpy as np
                    _,_,_,current=read_evaluation(self.root/'evaluations'/f'RL_{key}_update{update:06d}_DEV')
                    _,_,_,base=read_evaluation('/mnt/project/qwen-sourceflow-native-20261018/evaluations/selected_IL_DEV')
                    delta=float(100*(current[:,0]-base[:,0]).mean())
                    safety_delta=float(100*((current[:,1:4]==1).all(1).mean()-(base[:,1:4]==1).all(1).mean()))
                    alarm=dict(update=update,PDMS_delta=delta,joint_safety_delta_points=safety_delta,severe=delta<=-5 or safety_delta<=-5)
                    atomic_json(alarm_path,alarm)
                if previous and alarm['severe']:alarms.append(alarm)
                previous=alarm['severe']
            if alarms and key not in complete:
                marker=self.control/'HELD'/f'{key}.json'
                if not marker.exists():
                    atomic_json(marker,dict(reason='two consecutive complete DEV severe degradation',evidence=alarms))
                    for spec in read(self.task_path(key+'_formal50k'))['workers']:
                        (Path(spec['receipt_dir'])/'STOP_AFTER_BOUNDARY').touch()
        required_dev={k:[u for u in MILESTONES if k not in held or u<=held[k]['updates']] for k in reg['variants']}
        if len(complete)!=2 or any(set(required_dev[k])-set(dev[k]) for k in reg['variants']):
            return dict(status='FULL_TRAINING_AND_DEV',complete_variants=list(complete),complete_DEV={k:list(v) for k,v in dev.items()})
        locked=self.control/'ENDPOINTS_AND_DEV_LOCKED.json'
        choices={k:max(v,key=lambda u:(v[u]['PDMS'],u)) for k,v in dev.items()}
        if not locked.exists():
            streams={k:scene_stream_hash(run) for k,run in runs.items() if k not in held}
            if not held and len({tuple(v) for v in streams.values()})!=1:raise ValueError('Safety methods consumed different scene order')
            atomic_json(locked,dict(endpoints=complete,DEV=dev,selected_updates=choices,
                same_actual_scene_stream_hashes=streams,selection_used_Navtest=False))
        results={};pending=False
        for key in reg['variants']:
            results[key]={}
            for update in sorted({choices[key]} if key in held else {50000,choices[key]}):
                summary=self.ensure_evaluation(f'RL_{key}_update{update:06d}_NAVTEST',parent,'NAVTEST',export=dev[key][update]['export'])
                if summary is None:pending=True;continue
                pair=self.root/'analysis'/f'RL_{key}_update{update:06d}_NAVTEST_paired.json'
                if not pair.exists():self.run_analysis('scripts.analysis.qwen_sourceflow.analyze_paired',[
                    '--evaluation',str(self.root/'evaluations'/f'RL_{key}_update{update:06d}_NAVTEST'),
                    '--reference','/mnt/project/qwen-sourceflow-native-20261018/evaluations/selected_IL_NAVTEST','--output',str(pair)])
                results[key][update]=dict(summary=str(summary),paired=str(pair))
        if pending:return dict(status='LOCKED_FULL_NAVTEST',selected_updates=choices)
        final=dict(status='COMPLETE_WITH_HELD_VARIANTS' if held else 'COMPLETE',fixed_endpoints=complete,DEV_selected=choices,results=results,
            registration_sha256=file_sha(self.root/'experiment_registration.json'),seed=42,evaluation_seed=2101)
        atomic_json(self.control/'COMPLETE.json',final);return final


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--training-source',required=True)
    p.add_argument('--analysis-source',required=True);p.add_argument('--evaluation-source',required=True)
    p.add_argument('--once',action='store_true');a=p.parse_args();require_dispatch()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only campaign owner required')
    control=ROOT/'campaign';control.mkdir(exist_ok=True)
    lk=(control/'owner.lock').open('a');fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
    train=immutable_source(a.training_source);analysis=immutable_source(a.analysis_source);evaluation=immutable_source(a.evaluation_source)
    selection='/mnt/project/qwen-sourceflow-native-20261018/frozen_IL_parent53k_user20261023.json'
    identity=dict(training=train,analysis=analysis,evaluation=evaluation,data=dict(
        DEV='/mnt/project/qwen-sourceflow-native-20261018/data/DEV_v2',NAVTEST='/mnt/project/qwen-sourceflow-native-20261018/data/NAVTEST'),
        registration_sha256=file_sha(ROOT/'experiment_registration.json'),parent_selection=selection)
    if (control/'identity.json').exists() and read(control/'identity.json')!=identity:raise ValueError('Existing owner identity changed')
    atomic_json(control/'identity.json',identity);atomic_json(control/'owner.json',dict(pid=os.getpid(),start_unix=time.time()))
    queue=SafeCampaign(ROOT,control,train,analysis,evaluation,parent_selection=selection)
    try:
        while True:
            status=queue.tick();atomic_json(control/'status.json',dict(at=time.time(),**status))
            if a.once or status['status'] in ('COMPLETE','COMPLETE_WITH_HELD_VARIANTS') or (control/'STOP_AFTER_TICK').exists():break
            time.sleep(300)
    except BaseException:
        atomic_json(control/'failure.json',dict(at=time.time(),error=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':main()
