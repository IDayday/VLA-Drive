"""User-authorized B-only retirement using the previously audited cleanup engine."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time

ROOT=Path('/mnt/project/qwen-world-four-frameworks-20261029')
RUN=ROOT/'runs/ARCH_B/IL_seed42'
OUT=Path(__file__).resolve().parent
ORIGINAL=Path('/mnt/project/experiment-results-summary/20261102_selected_weight_cleanup/cleanup_lib.py')


def read(path, default=None):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return default


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)


def inside(path, parent):return path==parent or parent in path.parents


def collect():
    from tools.il_report.publish_bd_periodic import collect as collect_scores
    marker=read(RUN/'USER_TERMINATION.json');assert marker['status']=='TERMINATED_BY_USER'and marker['run']==str(RUN)
    assert read(RUN/'progress.json')['status']=='STOPPED'
    latest=Path(read(RUN/'LATEST.json')['checkpoint'])
    scores=[r for r in collect_scores()['completed']if r['architecture']=='B']
    best=max(scores,key=lambda r:(r['PDMS'],-r['updates']))
    best_native=RUN/'checkpoints'/f'update{best["updates"]:06d}'
    best_export=Path(best['evidence']).parent/'export'
    protected={latest,best_native,best_export};native=set(RUN.joinpath('checkpoints').glob('update*'));exports=set()
    cadence=read(Path('/mnt/project/qwen-world-il-research-20261030/navtest_cadence_20261031/STATE.json'),{})
    active={j['root']for j in cadence.get('jobs',[])if j.get('method')=='ARCH_B'and j.get('status')in('PENDING','RUNNING','DISPATCHING','WAITING_RESOURCE')}
    holds=[]
    for root in ROOT.iterdir():
        if not root.is_dir()or'navtest'not in root.name.lower():continue
        for label,pin in read(root/'PINNED.json',{}).get('models',{}).items():
            if pin.get('architecture_id')!='B'and not label.startswith('B_IL_'):continue
            directory=root/label;export=directory/'export'
            if export.is_dir()and not export.is_symlink():exports.add(export)
            for n in (root/'checkpoints'/label,):
                if n.is_dir()and not n.is_symlink():native.add(n)
            if str(root)in active and not(root/'COMPLETE.json').exists():
                protected.add(root);holds.append(dict(root=str(root),updates=pin['updates']))
                for key in('checkpoint','original_checkpoint'):
                    p=Path(pin.get(key,'/nonexistent'))
                    if inside(p,ROOT):protected.add(p)
    lineage=ROOT/'performance_20261031/migration/HANDOFF_OWNER/B/PARENT'
    if lineage.is_dir():native.add(lineage)
    latency=Path('/mnt/project/qwen-world-latency-20261103/exports/B')
    if latency.is_dir()and read(latency/'export.json',{}).get('architecture_id')=='B':exports.add(latency)
    return dict(latest=str(latest),best=best,best_native=str(best_native),best_export=str(best_export),
                protected=sorted(map(str,protected)),active_holds=holds,native=sorted(map(str,native)),
                exports=sorted(map(str,exports)),lineage_metadata_only=str(lineage))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--apply',action='store_true');p.add_argument('--phase',required=True)
    args=p.parse_args();phase=OUT/args.phase;phase.mkdir(exist_ok=False)
    snapshot=collect();save(phase/'RETENTION.json',snapshot)
    # Copy, rather than modify, the mature manifest/identity-safe unlink engine.
    libpath=OUT/'cleanup_lib.py'
    if not libpath.exists():shutil.copy2(ORIGINAL,libpath)
    assert sha(libpath)==sha(ORIGINAL)
    spec=importlib.util.spec_from_file_location('B_cleanup_engine',libpath);lib=importlib.util.module_from_spec(spec);spec.loader.exec_module(lib)
    lib.OUT=phase
    protected=[Path(p)for p in snapshot['protected']]
    candidates=[];seals=[];metadata=[];retired=[]
    for kind,paths in [('native',snapshot['native']),('export',snapshot['exports'])]:
        for name in paths:
            path=Path(name)
            if path.is_symlink()or any(inside(path,p)for p in protected):continue
            identity=read(path/('identity.json'if kind=='native'else'export.json'),{})
            assert identity.get('architecture_id')=='B',str(path)
            files=[Path(p)for p in lib.walk(path)]
            payload=[p for p in files if p.suffix.lower()in lib.BINARY]
            if not payload:continue
            evidence=[p for p in files if p.suffix.lower()in('.json','.yaml','.yml','.txt','.py')or p.name=='COMPLETE']
            metadata+=evidence
            seal=path/'COMPLETE'
            lineage=kind=='native'and str(path)==snapshot['lineage_metadata_only']
            if seal.is_file()and not lineage:seals.append((str(seal),str(path),'B_completion_seals','Retired intermediate payload; original seal archived'))
            candidates.extend((str(p),str(path),'B_'+kind+'_intermediate','User requested retaining only latest/best B weights')for p in payload)
            retired.append(dict(path=str(path),kind=kind,lineage_metadata_only=lineage,metadata=[str(p)for p in evidence],payload_files=len(payload)))
    summary=dict(at=time.time(),latest=snapshot['latest'],best_updates=snapshot['best']['updates'],best_PDMS=snapshot['best']['PDMS'],
        protected=snapshot['protected'],active_holds=snapshot['active_holds'],retired=retired,
        payload_files=len(candidates),payload_logical_bytes=sum(Path(p).stat().st_size for p,_,_,_ in candidates),
        engine_sha256=sha(libpath),scope='B only; results/metrics/source/images/caches and all D assets excluded')
    save(phase/'PLAN.json',summary);print(json.dumps({k:v for k,v in summary.items()if k not in('retired','protected')},indent=2),flush=True)
    if not args.apply:return
    assert read(OUT/'PROTECTED_CHECKPOINT_VERIFICATION.json')
    # Selected checkpoints must still be present and fully sealed at deletion.
    for path in (Path(snapshot['latest']),Path(snapshot['best_native'])):
        seal=read(path/'COMPLETE');inventory=read(path/'files.json');assert seal['files_sha256']==sha(path/'files.json')
        assert all((path/n).is_file()for n in inventory)
    def guard(row):
        return (read(RUN/'USER_TERMINATION.json',{}).get('status')=='TERMINATED_BY_USER'
                and read(RUN/'LATEST.json')['checkpoint']==snapshot['latest']
                and not any(inside(Path(row['path']),p)for p in protected))
    if seals:
        stage=lib.prepare('00_completion_seals',seals,metadata=metadata,protected=map(str,protected),allow_hardlinks=True,
                          allowed_extensions=lib.BINARY|{''},scan_workers=2)
        lib.execute(stage,workers=2,validator=guard)
    result={}
    if candidates:
        stage=lib.prepare('01_checkpoint_payloads',candidates,metadata=metadata,protected=map(str,protected),
                          allow_hardlinks=True,scan_workers=2)
        result=lib.execute(stage,workers=4,validator=guard)
    for item in retired:
        save(Path(item['path'])/'PAYLOAD_RETIREMENT.json',dict(at=time.time(),status='USER_AUTHORIZED_B_INTERMEDIATE_RETIRED',
            audit=str(phase),metadata_retained=True,results_and_scored_trajectories_retained=True,
            directly_resumable=False,lineage_metadata_only=item['lineage_metadata_only']))
    save(phase/'COMPLETE.json',dict(status='COMPLETE',at=time.time(),execution=result,retention=snapshot,
         original_cleanup_engine_unmodified=True,no_D_or_other_experiment_payloads_selected=True))


if __name__=='__main__':main()
