"""Sequential resumable orchestration of the registered two-model study only."""
from pathlib import Path
import os
import subprocess
import sys
from iqe.io import read_json,atomic_json,lock
from iqe.contracts import require
from .training import KINDS,resource,train_model,lock_refit_plan


def run(config,device='cuda:0',resume=False):
    from .binding import bind
    from .data import split_inner
    from .audit import run as audit
    from .privileged_probe import run as probe
    from .evaluation import evaluate_role,calibrate
    from .bundle import export
    from .benchmark import run as benchmark
    from .online import online_check
    from .process_audit import run as process_audit
    resource(config,'run-two-decision-study',training=True)
    with lock(config.root/'WORKFLOW.lock'):
        def stage(name,fn):
            atomic_json(config.root/'WORKFLOW_STATUS.json',dict(status='RUNNING',stage=name,pid=os.getpid(),old_experiments_started=False))
            return fn()
        stage('bind',lambda:bind(config));stage('split',lambda:split_inner(config));stage('existing_audit',lambda:audit(config))
        for phase in ('diagnostic','overfit'):
            for kind in KINDS:
                folder=config.root/phase/kind
                result=read_json(folder/'result.json') if (folder/'result.json').exists() else None
                if result and result['status']=='COMPLETE' and (folder/'FIT_REPORT.json').exists():continue
                recovery='auto' if resume and list(folder.glob('step_*.pt.COMPLETE.json')) else None
                stage(phase+'_'+kind,lambda k=kind,p=phase,r=recovery:train_model(config,k,p,device,r))
        stage('refit_plan',lambda:lock_refit_plan(config))
        current=config.root/'current_probe/training'
        recovery='auto' if resume and list(current.glob('step_*.pt.COMPLETE.json')) else None
        stage('current_probe',lambda:probe(config,device,recovery))
        for kind in KINDS:
            folder=config.root/'refit'/kind
            result=read_json(folder/'result.json') if (folder/'result.json').exists() else None
            if result and result['status']=='COMPLETE' and (folder/'FIT_REPORT.json').exists():continue
            recovery='auto' if resume and list(folder.glob('step_*.pt.COMPLETE.json')) else None
            stage('refit_'+kind,lambda k=kind,r=recovery:train_model(config,k,'refit',device,r))
        stage('external',lambda:evaluate_role(config,'stage_val',device));stage('calibration',lambda:calibrate(config,device))
        stage('dev',lambda:evaluate_role(config,'dev_report',device))
        for kind in KINDS:stage('export_'+kind,lambda k=kind:export(config,k))
        stage('online_equivalence',lambda:online_check(config,device))
        # Each child has its own model construction and imports. No hidden in-process reuse.
        from dataclasses import asdict
        import yaml
        path=config.root/'runtime_workflow_config.yaml'
        from iqe.io import atomic_bytes
        atomic_bytes(path,yaml.safe_dump(asdict(config)).encode(),immutable=True)
        for kind in KINDS:
            command=[sys.executable,'-m','iqe.decision_selection.online','--config',str(path),'--device',device,'--fresh-process',kind]
            code=subprocess.run(command,cwd=Path(__file__).parents[2]).returncode
            require(code==0,'fresh process bundle check failed')
        stage('latency',lambda:benchmark(config,device));stage('process_label_audit',lambda:process_audit(config))
        report=dict(status='COMPLETE',models=list(KINDS),expert_updates=0,navtest_started=False,ACTIVE_updated=False)
        atomic_json(config.root/'WORKFLOW_STATUS.json',report);return report
