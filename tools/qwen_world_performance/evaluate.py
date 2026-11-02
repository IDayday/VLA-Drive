"""Narrow performance export guard + unchanged canonical single-ODE evaluator.

`owner` reuses the qualified asynchronous EvaluationOwner. `export` validates
new B/G/M, Adam/exposure/stream state before invoking the original deployment
exporter. No policy scoring is implemented a second time here.
"""
import argparse,gc,json,os,socket,time
from pathlib import Path
from .launch import ROOT,TASK,PROTECTED,ORDER,FAST,QUALITY,STRUCTURED,INDEPENDENT_FAMILIES,bootstrap,read,sha,publish,validate_registry,inspect_native,register_native_priority_study


def native_claim_startup_hold(host,claim,sample,run):
    """A dispatch reservation is not a lifetime ban on qualified eval sharing.

    This only releases the advisory hold. The original usable_gpu/worker checks
    still decide every actual device, owner, headroom and UUID lease admission.
    """
    try:
        job=Path(claim['job']).resolve()
        if run is None or claim['arm']!=run['id']or not job.is_relative_to(ROOT/'jobs'/run['id']):
            return True,'unresolved_or_foreign_claim'
        if(job/'RESULT.json').exists():return False,'native_attempt_returned'
        running=read(job/'RUNNING.json',{});dispatch=read(job/'DISPATCH.json',{})
        command=running.get('command',[])
        def arg(flag):return command[command.index(flag)+1]if command.count(flag)==1 and command.index(flag)+1<len(command)else None
        if(running.get('status')!='RUNNING'or running.get('host')!=host
           or dispatch.get('host')!=host or dispatch.get('command')!=command
           or running.get('cwd')!=run.get('execution_source',run['source'])
           or running.get('worker_pid')!=dispatch.get('worker_pid')
           or running.get('worker_start_ticks')!=dispatch.get('worker_start_ticks')
           or not running.get('child_pid')or not running.get('child_start_ticks')
           or run['trainer_module']not in command or arg('--output')is None
           or Path(arg('--output')).resolve()!=Path(run['root']).resolve()or arg('--config')!=run['config']):
            return True,'dispatch_or_native_loading_window'
        uuids={g['uuid']for g in sample['gpus']}
        if(len(uuids)!=8 or len(running.get('gpu_uuids',[]))!=8
           or set(running['gpu_uuids'])!=uuids):return True,'physical_mapping_not_confirmed'
        ready={o['uuid']for o in sample['owners']if o.get('visible')is True
               and isinstance(o.get('output'),str)and Path(o['output']).resolve()==Path(run['root']).resolve()and o.get('entry')==run['trainer_module']
               and isinstance(o.get('pid'),int)and isinstance(o.get('start_ticks'),int)
               and isinstance(o.get('used_MiB'),(int,float))and o['used_MiB']>=0}
        if ready!=uuids:return True,'all_eight_native_gpu_owners_not_visible_yet'
        return False,'running_native_ready_use_original_device_admission'
    except(OSError,ValueError,KeyError,TypeError,IndexError):
        return True,'ambiguous_claim_kept_for_review'


def validate_native_export_receipt(run,checkpoint,receipt):
    if run['id']==FAST:
        from .continuation import same_path
        if(receipt.get('schema')!='world_packed_export_native_validation_v1'or receipt.get('status')!='PASS'
           or not same_path(receipt.get('checkpoint'),checkpoint['path'])
           or receipt.get('files_sha256')!=checkpoint['files_sha256']
           or receipt.get('counts',{}).get('updates')!=checkpoint['updates']or receipt.get('micro_scenes')!=4):
            raise ValueError('Packed export lacks its own exact native validation receipt')
    elif run['id']==QUALITY:
        from .continuation import same_path
        from starVLA.rl.qwen_world_research_data.manifest import fingerprint
        import yaml
        protocol_sha=receipt.get('quality_protocol_sha256')
        protocol=yaml.safe_load(Path(run['config']).read_text())['research']['verified_quality']
        if(receipt.get('schema')!='world_verified_quality_export_native_validation_v1'or receipt.get('status')!='PASS'
           or not same_path(receipt.get('checkpoint'),checkpoint['path'])
           or receipt.get('files_sha256')!=checkpoint['files_sha256']
           or receipt.get('counts',{}).get('updates')!=checkpoint['updates']
           or not isinstance(protocol,dict)or protocol_sha!=fingerprint(protocol)):
            raise ValueError('S6 export lacks its own exact quality native validation receipt')
    elif run['id']==STRUCTURED:
        from .continuation import same_path
        from starVLA.rl.qwen_world_research_data.manifest import fingerprint
        import yaml
        cfg=yaml.safe_load(Path(run['config']).read_text())
        if(receipt.get('schema')!='world_structured_progress_export_native_validation_v1'or receipt.get('status')!='PASS'
           or not same_path(receipt.get('checkpoint'),checkpoint['path'])
           or receipt.get('files_sha256')!=checkpoint['files_sha256']or receipt.get('counts',{}).get('updates')!=checkpoint['updates']
           or receipt.get('counts',{}).get('next_inner')!=0 or receipt.get('world')!=8 or receipt.get('qualification')is not False
           or receipt.get('parent_model_sha256')!=cfg['research']['performance']['parent_model_sha256']
           or receipt.get('structured_progress_protocol_sha256')!=fingerprint(cfg['research']['structured_progress'])):
            raise ValueError('S7 export lacks its exact structured-progress native validation receipt')
    elif(receipt.get('status')!='PASS'or receipt.get('checkpoint',{}).get('files_sha256')!=checkpoint['files_sha256']):
        raise ValueError('A performance export lacks native validation')
    return receipt



def native_adoption_command(run,parent,checkpoint,python):
    """Exact family-native API, including S7's required parent argument."""
    module=INDEPENDENT_FAMILIES[run['id']]['native_review']
    if run['id']==STRUCTURED:
        return [python,'-c',f'import sys;from {module} import load_validated;load_validated(sys.argv[1],sys.argv[2])',str(parent),str(checkpoint)]
    return [python,'-c',f'import sys;from {module} import load_validated;load_validated(sys.argv[1])',str(checkpoint)]


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
    if run['id']in INDEPENDENT_FAMILIES:
        # Execute this exact family's complete native validator/exporter in
        # its frozen source, avoiding mixed imports into the ops interpreter.
        import subprocess
        from starVLA.rl.qwen_world_safety_exploration.resources import MODEL_PYTHON
        source=str(Path(run['source']).resolve());env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=source)
        module=INDEPENDENT_FAMILIES[run['id']]['native_review']
        if not(Path(output)/'COMPLETE').exists():
            subprocess.check_call([MODEL_PYTHON,'-m',module,'export',
                '--parent-checkpoint',str(parent),'--rl-checkpoint',cp['path'],'--output',str(output)],cwd=source,env=env)
        else:
            # Adoption still validates the actual packed pending/cursor schema.
            subprocess.check_call(native_adoption_command(run,parent,cp['path'],MODEL_PYTHON),cwd=source,env=env)
        meta=inspect_export(output)
        if meta.get('RL_checkpoint_files_sha256')!=cp['files_sha256']or meta.get('RL_updates')!=cp['updates']:raise ValueError('Independent-family export bound to another checkpoint')
        receipt=Path(output).with_name(Path(output).name+'.native_validation.json')
        proof=validate_native_export_receipt(run,cp,read(receipt,{}))
        if proof.get('identity_sha256')!=identity['sha256']:raise ValueError('Independent-family native receipt identity changed')
        # Keep this scientist's immutable receipt; do not rewrite it using
        # the baseline family's schema or hide its full recovery proof.
        return meta
    # A complete payload hash is essential once for an exported model. Metadata
    # polling/admission uses inspect_native and never repeatedly loads optimizer.
    for name,digest in files.items():
        if sha(root/name)!=digest:raise ValueError('Changed export payload: '+name)
    if run.get('execution_source'):
        from .execution import validate_checkpoint_in_execution
        checked=validate_checkpoint_in_execution(run,root);stream_digests=[checked['scene_stream_sha256']]
    else:
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


def owner_class(max_known_evaluations_per_gpu=1):
    if max_known_evaluations_per_gpu not in(1,2):raise ValueError('Explicit known-evaluator cap must be one or two')
    bootstrap()
    register_native_priority_study()
    from tools.qwen_world_safety_exploration import eval_owner as E
    from starVLA.rl.qwen_world_safety_exploration.resources import MODEL_PYTHON
    E.KNOWN_ROOTS=tuple(dict.fromkeys(E.KNOWN_ROOTS+(str(ROOT)+'/',)))
    E.completion_priority=lambda entry:(0 if entry.get('arm')in ORDER[:2]else ORDER.index(entry['arm']),
        entry.get('registered_exposure_epoch',0),-sum(n.get('status')=='COMPLETE'for n in entry.get('ranks',{}).values()),entry.get('label',''))
    class PerformanceEvaluationOwner(E.EvaluationOwner):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.max_known_evaluations_per_gpu=max_known_evaluations_per_gpu
            self.state['evaluation_sharing_policy']=dict(max_known_evaluations_per_gpu=max_known_evaluations_per_gpu,
                minimum_free_MiB=self.minimum,unknown_owners_allowed=False,worker_rechecks_before_and_after_locks=True)
        @staticmethod
        def binding(run):
            # Budget/gates may append under explicit Root authorization; all
            # scientific/existing checkpoint/spec bindings remain immutable.
            return {k:run[k]for k in('id','architecture_id','root','source','source_commit','config','config_sha256','parent_checkpoint')}
        def checkpoint_at_gate(self,run,updates):
            from .continuous import enabled,published_checkpoint_at_gate
            if enabled(run):return published_checkpoint_at_gate(run,updates)
            return super().checkpoint_at_gate(run,updates)
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
                # The CPU export guard must understand operational sharing
                # fields added after formal_v1. Prediction/model/scoring files
                # remain unchanged; GPU evaluation keeps the scientific source.
                run=dict(run,source=str(Path(__file__).resolve().parents[2]))
            return super().launch_job(run,node,command,family,route)
        def dispatch_ranks(self,samples):
            # Never take an empty8GPU primary lane while a registered learner
            # waits. Other project owners are not modified or preempted.
            native=read(ROOT/'owners/PERFORMANCE_NATIVE/STATE.json',{}).get('runs',{})
            waiting=any(n.get('status')in('WAITING_RESOURCE','WAITING_AUTHORIZED_RESOURCE')and time.time()-n.get('heartbeat_unix',0)<180 for n in native.values())
            if waiting:
                samples=[(h,s)for h,s in samples if not all(E.lease_free(E.LEASES['primary']/(g['uuid']+'.lock'))for g in s['gpus'])]
            claims=read(ROOT/'resources/NATIVE_HOST_CLAIMS.json',{})
            runs={r['id']:r for r in self.registry['runs']};held=set();readiness={}
            for host,sample in samples:
                claim=claims.get(host)
                if not claim:continue
                hold,reason=native_claim_startup_hold(host,claim,sample,runs.get(claim.get('arm')))
                readiness[host]=dict(startup_hold=hold,reason=reason,at_unix=time.time())
                if hold:held.add(host)
            self.state['native_claim_readiness']=readiness
            super().dispatch_ranks([(h,s)for h,s in samples if h not in held])
        def advance(self,samples):
            super().advance(samples)
            runs={r['id']:r for r in self.registry['runs']}
            for arm,record in self.state['runs'].items():
                for gate in record['gates'].values():
                    export=gate.get('export',{})
                    if export.get('status')=='COMPLETE':
                        receipt=Path(export['path']).with_name(Path(export['path']).name+'.native_validation.json');r=read(receipt,{})
                        validate_native_export_receipt(runs[arm],gate['checkpoint'],r)
            signature=tuple(sorted((k,v.get('summary'))for k,v in self.state['evaluations'].items()if v.get('status')=='COMPLETE'))
            if(signature and signature!=getattr(self,'_checkpoint_selection_signature',None)
               and time.time()>=getattr(self,'_checkpoint_selection_retry_unix',0)):
                try:
                    from .select_checkpoints import refresh_selection
                    selected=refresh_selection(self.registry,self.state)
                    self.state['checkpoint_selection']=dict(status=selected['status'],path=str(ROOT/'selections/STATE.json'))
                    self._checkpoint_selection_signature=signature
                except Exception as error:
                    # Administrative selection must not stop healthy scoring.
                    self.state['checkpoint_selection']=dict(status='REVIEW_REQUIRED',error=repr(error))
                    self._checkpoint_selection_retry_unix=time.time()+300
    return PerformanceEvaluationOwner


def main():
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='export':
        ap=argparse.ArgumentParser();ap.add_argument('action');ap.add_argument('--parent-checkpoint',required=True);ap.add_argument('--rl-checkpoint',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();strict_export(a.parent_checkpoint,a.output,a.rl_checkpoint);return
    if len(sys.argv)>1 and sys.argv[1]=='owner':
        ap=argparse.ArgumentParser();ap.add_argument('action');ap.add_argument('--registry',default=str(ROOT/'RUNS.json'));ap.add_argument('--max-shards',type=int,default=32);ap.add_argument('--max-workers',type=int,default=64);ap.add_argument('--interval',type=float,default=30);ap.add_argument('--max-known-evaluations-per-gpu',type=int,choices=(1,2),default=1);a=ap.parse_args()
        if a.max_shards!=32 or not 1<=a.max_workers<=64 or a.interval<10:ap.error('Canonical32 shards, <=64workers, interval>=10s')
        owner_class(a.max_known_evaluations_per_gpu)(a.registry,max_shards=32,max_workers=a.max_workers,minimum=32768).run(interval=a.interval);return
    # Exact old CLI and scoring implementation, including resource-return caches.
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    try:bootstrap()
    finally:
        # Bootstrap is CPU-only for owner/export, but the canonical fallback
        # runs inside an already UUID-admitted GPU worker. Preserve its mask.
        if visible is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else:os.environ['CUDA_VISIBLE_DEVICES']=visible
    from tools.qwen_world_safety_exploration.evaluate import main as canonical_main
    canonical_main()
if __name__=='__main__':main()
