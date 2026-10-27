"""Completion callback: retire closed-evaluation aliases, account, publish."""
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import time
import traceback

ROOT=Path(__file__).resolve().parent
REPO=Path('/mnt/project/VLA-Drive-retire-training-states-20261027')
PYTHON='/mnt/project/flow-safe-improvement-20261013/venv/bin/python'


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    temp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with temp.open('x') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2);handle.write('\n');handle.flush();os.fsync(handle.fileno())
    os.replace(temp,path)


def wait_owner():
    if not(ROOT/'PARALLEL_EXIT.json').exists():
        owner=read(ROOT/'PARALLEL_COORDINATOR.json');path=Path('/proc')/str(owner['pid'])
        fd=os.pidfd_open(owner['pid']);stat=(path/'stat').read_text();fields=stat[stat.rfind(')')+2:].split()
        if int(fields[19])!=owner['start_ticks']:raise ValueError('Cleanup coordinator PID changed')
        poll=select.poll();poll.register(fd,select.POLLIN)
        while not poll.poll(60000):pass
        os.close(fd)
    result=read(ROOT/'PARALLEL_EXIT.json')
    if result['status']!='COMPLETE' or any(result['exit_codes'][str(i)]!=0 for i in range(4)):
        raise ValueError('Some cleanup workers failed; preserve all transaction evidence')


def post_evaluation():
    request=read(ROOT/'POST_NAVTEST_RECOVERY_TARGETS.json')
    complete=read(request['source_complete'])
    if complete['status']!='COMPLETE' or not complete['all_owned_workers_exited']:
        raise ValueError('Evaluation raw dependencies are still in use')
    post=ROOT/'post_navtest';post.mkdir(exist_ok=False)
    write(post/'ACTIVE_PROTECTED.json',dict(inodes=[],scope='Only closed native RL and native IL parents; World never enters this scope'))
    rows=[]
    for text in request['roots']:
        path=Path(text);identity=read(path/'identity.json')
        if identity['schema'] not in ('native_qwen_sourceflow_RL_v1','native_qwen_sourceflow_IL_v1'):
            raise ValueError('Unexpected parent or model family')
        files=[]
        for name in ['optimizer.pt',*[f'rank{i}.pt' for i in range(8)]]:
            p=path/name
            if not p.is_file():continue
            s=p.stat();files.append(dict(name=name,path=str(p),size=s.st_size,blocks=s.st_blocks*512,
                device=s.st_dev,inode=s.st_ino,nlink=s.st_nlink))
        rows.append(dict(path=text,schema=identity['schema'],files=files,model_exists=(path/'model.pt').is_file()))
    inventory=post/'INVENTORY.json';write(inventory,dict(checkpoints=rows))
    command=[PYTHON,str(REPO/'tools/maintenance/retire_training_states/run.py'),
        '--audit-dir',str(post),'--inventory',str(inventory),'--apply','--attempt','post_navtest']
    with(post/'cleanup.log').open('x') as log:
        subprocess.run(command,cwd=REPO,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1',PYTHONPATH='/mnt/project/VLA-Drive-qwenflow-rl77-selected-navtest-20261027'),
            stdout=log,stderr=subprocess.STDOUT,check=True)
    if read(post/'PHASE2_COMPLETE.json')['status']!='COMPLETE':raise ValueError('Post-evaluation cleanup failed')
    return rows


def account(post_rows):
    inventory=read(ROOT/'CHECKPOINT_INVENTORY.json')['checkpoints'];all_rows=inventory+post_rows
    groups={};parents={}
    def add(file):
        key=(file['device'],file['inode']);row=groups.setdefault(key,dict(paths=set(),blocks=0,nlink=0,size=0))
        row['paths'].add(file['path']);row['blocks']=max(row['blocks'],file.get('blocks',file.get('allocated',0)))
        row['nlink']=max(row['nlink'],file['nlink']);row['size']=max(row['size'],file['size'])
    for row in all_rows:
        for file in row['files']:add(file)
    for file in read(ROOT/'PHASE1_COMPLETE.json')['removed_files']:add(file)
    for group in groups.values():
        for name in group['paths']:
            parent=Path(name).parent
            if parent not in parents:parents[parent]={p.name for p in parent.iterdir()}
    removed_paths=0;removed_link_bytes=0;freed=0;remaining=[];unknown_aliases=0
    for row in groups.values():
        gone=[p for p in row['paths'] if Path(p).name not in parents[Path(p).parent]]
        removed_paths+=len(gone);removed_link_bytes+=len(gone)*row['size']
        if len(gone)==len(row['paths']):
            if row['nlink']<=len(row['paths']):freed+=row['blocks']
            else:unknown_aliases+=1
        else:remaining.extend(sorted(set(row['paths'])-set(gone)))
    archives={};staging=[]
    for directory in [ROOT/'inference_weights',ROOT/'post_navtest/inference_weights']:
        for p in directory.iterdir():
            if not p.is_file():continue
            s=p.stat();archives[(s.st_dev,s.st_ino)]=s.st_blocks*512
            if p.suffix!='.pt':staging.append(str(p))
    # Model weights originally present must still be available. LoRA-only
    # policies must have either the original container or an exact capsule.
    kept_models=kept_adapters=0
    for row in all_rows:
        path=Path(row['path']);names=parents.get(path)
        if names is None:names={p.name for p in path.iterdir()}
        if row.get('model_exists'):
            if 'model.pt' not in names:raise ValueError('An inference model is missing: '+str(path))
            kept_models+=1
        if any(f['name']=='optimizer.pt' for f in row['files']) and row['schema'] in (
                'native_qwen_sourceflow_RL_v1','fm_transfer_source_RL_v1','ddp_simwam_recipe_v1'):
            if 'optimizer.pt' not in names and 'inference_adapters.pt' not in names:
                raise ValueError('Lost inference adapter closure: '+str(path))
            if 'inference_adapters.pt' in names:kept_adapters+=1
    archive_bytes=sum(archives.values());net=freed-archive_bytes
    output=dict(status='COMPLETE',at=time.time(),user_scope='Stopped and completed experiments; unfinished World training recovery state retained',
        checkpoint_rows_checked=len(all_rows),unique_checkpoint_dirs_checked=len({r['path'] for r in all_rows}),
        removed_paths=removed_paths,removed_logical_link_bytes=removed_link_bytes,
        original_allocated_bytes_with_all_registered_links_removed=freed,preserved_new_archive_allocated_bytes=archive_bytes,
        net_reclaimed_allocated_bytes_lower_bound=net,net_reclaimed_GiB_lower_bound=net/1024**3,
        net_reclaimed_TiB_lower_bound=net/1024**4,unknown_remaining_external_alias_groups=unknown_aliases,
        retained_state_paths=len(remaining),retained_state_examples=remaining[:10],private_staging_artifacts=staging,
        inference_model_rows_retained=kept_models,inference_adapter_rows_retained=kept_adapters,
        exact_roundtrip_before_every_adapter_unlink=True,full_native_deployment_parity=read(ROOT/'NATIVE_SOURCE5K_DEPLOYMENT_PARITY.json'),
        file_accounting='Deduplicate original inode/hardlinks, subtract actual new archive allocated blocks; not inferred from shared filesystem df changes.',
        no_training_or_evaluation_process_signaled=True,no_model_or_score_or_data_or_feature_cache_deletion=True,
        historical_manifests_preserved=True,resume_of_retired_runs_available=False,
        retained_weight_bearing_exception='Legacy C/N initial ZeRO optim_states can contain the only FP32 model masters; these were not treated as disposable Adam state.')
    write(ROOT/'CLEANUP_COMPLETE.json',output)
    return output


def publish(result):
    report=REPO/'reports/retired_training_states_20261027'
    write(report/'summary.json',result)
    identity=read(report/'operator_identity.json');identity['phase']='CLEANUP_COMPLETE'
    write(report/'operator_identity.json',identity)
    subprocess.run(['git','add','--',str(report.relative_to(REPO))],cwd=REPO,check=True)
    subprocess.run(['git','commit','-m','Record completed state retirement and net storage accounting'],cwd=REPO,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+branch],cwd=REPO,check=True)
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=REPO,text=True).split()[0]
    if remote!=commit:raise ValueError('Maintenance remote SHA mismatch')
    write(ROOT/'PUBLICATION.json',dict(status='PUBLISHED',commit=commit,remote_SHA=remote,branch=branch,at=time.time()))


def main():
    lock=(ROOT/'finalizer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if(ROOT/'FINALIZER.json').exists():raise FileExistsError('Do not duplicate finalization')
    write(ROOT/'FINALIZER.json',dict(pid=os.getpid(),at=time.time()))
    try:
        wait_owner();rows=post_evaluation();result=account(rows);publish(result)
        write(ROOT/'FINAL_COMPLETE.json',dict(status='COMPLETE',net_reclaimed_TiB_lower_bound=result['net_reclaimed_TiB_lower_bound'],at=time.time()))
    except BaseException:
        write(ROOT/'FINALIZER_FAILURE.json',dict(at=time.time(),error=traceback.format_exc(),inference_archives_retained=True));raise


if __name__=='__main__':main()
