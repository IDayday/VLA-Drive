"""Narrow performance export guard + unchanged canonical single-ODE evaluator.

`owner` reuses the qualified asynchronous EvaluationOwner. `export` validates
new B/G/M, Adam/exposure/stream state before invoking the original deployment
exporter. No policy scoring is implemented a second time here.
"""
import argparse,gc,json,os,socket,time
from pathlib import Path
from .launch import ROOT,TASK,PROTECTED,ORDER,bootstrap,read,sha,publish,validate_registry,inspect_native


def strict_export(parent,output,checkpoint):
    bootstrap()
    import torch
    from starVLA.rl.qwen_world_performance import recovery
    from starVLA.rl.qwen_world_performance.sampling_priority import stream_state_validation
    from starVLA.rl.qwen_world_postcot.deployment import export_checkpoint,inspect_export
    registry=validate_registry(read(ROOT/'RUNS.json'))
    run=next((r for r in registry['runs']if Path(checkpoint).resolve().is_relative_to(Path(r['root'])/'checkpoints')),None)
    if run is None or Path(parent).resolve()!=Path(run['parent_checkpoint']).resolve():raise ValueError('Foreign export parent/native checkpoint')
    cp=inspect_native(run,checkpoint);root=Path(cp['path']);identity=read(root/'identity.json');files=read(root/'files.json')
    if cp['next_inner']!=0 or cp['phase']not in('READY','COMPLETE'):raise ValueError('Periodic export requires completed source window')
    # A complete payload hash is essential once for an exported model. Metadata
    # polling/admission uses inspect_native and never repeatedly loads optimizer.
    for name,digest in files.items():
        if sha(root/name)!=digest:raise ValueError('Changed export payload: '+name)
    common=torch.load(root/'optimizer.pt',map_location='cpu',weights_only=False);summaries=[];stream_digests=[]
    for rank in range(8):
        own=torch.load(root/f'rank{rank}.pt',map_location='cpu',weights_only=False)
        if own['identity_sha256']!=identity['sha256']:raise ValueError('Export rank identity changed')
        summaries.append(recovery.validate_boundary(own['state'],own['rollout'],identity,common['optimizer'],common['scheduler'],expected_rank=rank))
        stream_digests.append(stream_state_validation(own['state'],run_identity=identity['sha256'],stream_identity=identity['data_stream_identity'])['stream_state_sha256'])
    recovery.validate_ranks(summaries)
    if len(set(stream_digests))!=1:raise ValueError('Export ranks disagree on exact scene/feedback state')
    del common,own;gc.collect()
    output=Path(output);receipt=output.with_name(output.name+'.native_validation.json')
    publish(receipt,dict(schema='world_performance_export_native_validation_v1',status='PASS',checkpoint=cp,
        run_identity=identity['sha256'],scene_stream_sha256=stream_digests[0],source=str(Path.cwd()),source_file_sha256=sha(__file__),at_unix=time.time()))
    if not(output/'COMPLETE').exists():export_checkpoint(parent,output,checkpoint)
    meta=inspect_export(output)
    if meta.get('RL_checkpoint_files_sha256')!=cp['files_sha256']or meta.get('RL_updates')!=cp['updates']:raise ValueError('Export bound to a different checkpoint')
    return meta


def owner_class():
    bootstrap()
    from tools.qwen_world_safety_exploration import eval_owner as E
    from starVLA.rl.qwen_world_safety_exploration.resources import MODEL_PYTHON
    E.KNOWN_ROOTS=tuple(dict.fromkeys(E.KNOWN_ROOTS+(str(ROOT)+'/',)))
    E.completion_priority=lambda entry:(0 if entry.get('arm')in ORDER[:2]else ORDER.index(entry['arm']),
        entry.get('registered_exposure_epoch',0),-sum(n.get('status')=='COMPLETE'for n in entry.get('ranks',{}).values()),entry.get('label',''))
    class PerformanceEvaluationOwner(E.EvaluationOwner):
        @staticmethod
        def binding(run):
            # Budget/gates may append under explicit Root authorization; all
            # scientific/existing checkpoint/spec bindings remain immutable.
            return {k:run[k]for k in('id','architecture_id','root','source','source_commit','config','config_sha256','parent_checkpoint')}
        def registry_now(self):
            registry=validate_registry(read(self.registry_path))
            for location in registry['evaluation']['data_paths'].values():
                m=read(Path(location)/'manifest.json',{})
                if m.get('cohort')!='NAVTEST'or len(m.get('tokens',[]))!=12146:raise ValueError('Full canonical NAVTEST data changed')
            return registry
        def samples(self):return[(h,s)for h,s in super().samples()if h not in PROTECTED]
        def discover(self):
            super().discover()
            runs={r['id']:r for r in self.registry['runs']}
            for entry in self.state['evaluations'].values():
                point=next(p for p in runs[entry['arm']]['exposure_mapping'].values()if p['updates']==entry['updates'])
                entry.update(registered_exposure_epoch=point['epoch'],actual_valid_scene_exposures=point['valid_scene_exposures'],NavtestSelected=True)
        def launch_job(self,run,node,command,family,route=None):
            if '-c'in command and 'export_policy'in command[command.index('-c')+1]:
                command=[MODEL_PYTHON,'-u','-m','tools.qwen_world_performance.evaluate','export',
                    '--parent-checkpoint',command[-3],'--output',command[-2],'--rl-checkpoint',command[-1]]
            return super().launch_job(run,node,command,family,route)
        def dispatch_ranks(self,samples):
            # Never take an empty8GPU primary lane while a registered learner
            # waits. Other project owners are not modified or preempted.
            native=read(ROOT/'owners/PERFORMANCE_NATIVE/STATE.json',{}).get('runs',{})
            waiting=any(n.get('status')in('WAITING_RESOURCE','WAITING_AUTHORIZED_RESOURCE')and time.time()-n.get('heartbeat_unix',0)<180 for n in native.values())
            if waiting:
                samples=[(h,s)for h,s in samples if not all(E.lease_free(E.LEASES['primary']/(g['uuid']+'.lock'))for g in s['gpus'])]
            claims=read(ROOT/'resources/NATIVE_HOST_CLAIMS.json',{})
            held={h for h,c in claims.items()if not(Path(c['job'])/'RESULT.json').exists()}
            super().dispatch_ranks([(h,s)for h,s in samples if h not in held])
        def advance(self,samples):
            super().advance(samples)
            for record in self.state['runs'].values():
                for gate in record['gates'].values():
                    export=gate.get('export',{})
                    if export.get('status')=='COMPLETE':
                        receipt=Path(export['path']).with_name(Path(export['path']).name+'.native_validation.json');r=read(receipt,{})
                        if r.get('status')!='PASS'or r.get('checkpoint',{}).get('files_sha256')!=gate['checkpoint']['files_sha256']:
                            raise ValueError('A performance export lacks new native validation; never silently adopt an old export')
    return PerformanceEvaluationOwner


def main():
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='export':
        ap=argparse.ArgumentParser();ap.add_argument('action');ap.add_argument('--parent-checkpoint',required=True);ap.add_argument('--rl-checkpoint',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();strict_export(a.parent_checkpoint,a.output,a.rl_checkpoint);return
    if len(sys.argv)>1 and sys.argv[1]=='owner':
        ap=argparse.ArgumentParser();ap.add_argument('action');ap.add_argument('--registry',default=str(ROOT/'RUNS.json'));ap.add_argument('--max-shards',type=int,default=32);ap.add_argument('--max-workers',type=int,default=64);ap.add_argument('--interval',type=float,default=30);a=ap.parse_args()
        if a.max_shards!=32 or not 1<=a.max_workers<=64 or a.interval<10:ap.error('Canonical32 shards, <=64workers, interval>=10s')
        owner_class()(a.registry,max_shards=32,max_workers=a.max_workers,minimum=32768).run(interval=a.interval);return
    # Exact old CLI and scoring implementation, including resource-return caches.
    bootstrap()
    from tools.qwen_world_safety_exploration.evaluate import main as canonical_main
    canonical_main()
if __name__=='__main__':main()
