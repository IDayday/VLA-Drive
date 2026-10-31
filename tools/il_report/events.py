"""Reuse RL inotify mechanics; extend to several IL source/evaluation roots."""
import os
from pathlib import Path
import select
from .rl_events_reference import OwnerStateEvents, read_object, fingerprint, state_signatures

NAMES = {'STATE.json','SUMMARY.json','COMPLETE.json','SELECTION.json','RESULT.json','report.html','USER_TERMINATION.json',
         'LATEST.json','FAILURE.json','server_dispatch_policy.json','refresh_request.json'}
SKIP = {'scenes','checkpoints','weights','images','data','teachers','candidates','candidate_refresh',
        'export','exports','runtime','invocations','assets','upstream','targets','__pycache__'}


class ILStateEvents(OwnerStateEvents):
    def __init__(self, root, persist, directories=()):
        super().__init__(root, persist)
        self.add_sources(directories, initial=True)
        self.watch(Path('/mnt/project'))

    def add_sources(self, directories, *, initial=False):
        before=set(self.directories.values())
        for directory in directories:
            self.watch(Path(directory))
        if initial:
            for directory in set(self.directories.values())-before:
                for name in NAMES-{'refresh_request.json'}:
                    path=directory/name
                    if path.is_file():
                        self.seen.update(self.signatures(path))
            self.save()

    def signatures(self, path):
        if path.name=='USER_TERMINATION.json':
            data=read_object(path)
            return {fingerprint(['user_termination',str(path),data.get('at')]):dict(kind='user_experiment_terminated',source=str(path))}if data.get('status')=='TERMINATED_BY_USER'else{}
        if path.name=='report.html':
            if path!=Path('/mnt/project/qwen-world-rl-optimization-20261030/report.html'):return {}
            try:at=path.stat().st_mtime_ns
            except FileNotFoundError:return {}
            return {fingerprint(['RL_report',at]):dict(kind='RL_phase_report_resource_review',source=str(path))}
        if path.name=='server_dispatch_policy.json':
            data=read_object(path)
            value={key:data.get(key)for key in ('allowed_hosts','global_dispatch_prohibitions','pending_resource_additions')}
            return {fingerprint(['resources',value]):dict(kind='resource_authorization_changed',source=str(path))}
        if path.name=='refresh_request.json':
            return super().signatures(path)
        data=read_object(path)
        if path.name=='LATEST.json':
            update=data.get('updates')
            if isinstance(update,int) and update>0 and update%5000==0:
                return {fingerprint(['checkpoint',str(path),update]):dict(kind='training_stage_complete',source=str(path),updates=update)}
            return {}
        if path.name=='SUMMARY.json':
            if data.get('status')=='COMPLETE' and data.get('sha256'):
                return {fingerprint(['sealed_summary',str(path),data['sha256']]):dict(kind='evaluation_complete',source=str(path))}
            return {}
        if path.name=='SELECTION.json'and data.get('status')in ('LOCKED','COMPLETE'):
            return {fingerprint(['selection',str(path),data.get('sha256')]):dict(kind='stage_selection_complete',source=str(path))}
        if path.name=='RESULT.json'and(data.get('status')in('FAILED','FORCED_EXIT_REVIEW_REQUIRED')or data.get('returncode',0)not in(0,75)):
            return {fingerprint(['failure',str(path),data.get('status'),data.get('end_unix')]):dict(kind='important_exception',source=str(path))}
        if path.name in('STATE.json','FAILURE.json','COMPLETE.json'):
            signatures=state_signatures(path)
            if path.name=='FAILURE.json'and data:
                signatures[fingerprint(['failure_file',str(path),data.get('at'),data.get('error')])]=dict(kind='important_exception',source=str(path))
            return signatures
        return {}

    def wait(self, timeout):
        ready,_,_=select.select([self.fd,self.wake_read],[],[],max(0,timeout))
        if self.wake_read in ready:
            try:os.read(self.wake_read,4096)
            except BlockingIOError:pass
        if self.fd not in ready:return []
        paths=set();overflow=False
        while True:
            try:raw=os.read(self.fd,262144)
            except BlockingIOError:break
            position=0
            while position<len(raw):
                wd,mask,cookie,length=self._HEADER.unpack_from(raw,position)
                position+=self._HEADER.size
                name=os.fsdecode(raw[position:position+length].split(b'\0',1)[0]);position+=length
                if mask&self._OVERFLOW:overflow=True;continue
                directory=self.directories.get(wd)
                if directory is None:continue
                path=directory/name
                if mask&self._ISDIR:
                    # Watch newly-created evaluation/control directories only;
                    # scene records and big payload trees never enter inotify.
                    if name not in SKIP and (directory!=Path('/mnt/project')) and (
                            'NAVTEST'in name.upper()or name=='evaluation'or name.startswith(('ARCH_','DEV_','update',
                            'A_IL_','B_IL_','C_IL_','D_IL_','OPTIMIZED_IL_','TIME_RESIDUAL_IL_','HUBER_IL_','WORLD_IL_'))):
                        self.watch(path)
                        if(path/'evaluation').is_dir():self.watch(path/'evaluation')
                        paths.update(path/n for n in NAMES if(path/n).is_file())
                    continue
                if name in NAMES and mask&(self._CLOSE_WRITE|self._MOVED_TO):
                    paths.add(path)
        if overflow:
            paths.update(directory/name for directory in self.directories.values()for name in NAMES if(directory/name).is_file())
        new={}
        for path in paths:
            new.update({key:value for key,value in self.signatures(path).items()if key not in self.seen})
        if new:
            self.seen.update(new);self.save()
        return list(new.values())
