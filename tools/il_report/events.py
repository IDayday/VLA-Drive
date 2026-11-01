"""Reuse RL inotify mechanics; extend to several IL source/evaluation roots."""
import os
import errno
import time
from pathlib import Path
import select
from .rl_events_reference import OwnerStateEvents, read_object, fingerprint, state_signatures

NAMES = {'STATE.json','SUMMARY.json','COMPLETE.json','SELECTION.json','RESULT.json','report.html','USER_TERMINATION.json',
         'LATEST.json','FAILURE.json','server_dispatch_policy.json','refresh_request.json'}
SKIP = {'scenes','checkpoints','weights','images','data','teachers','candidates','candidate_refresh',
        'export','exports','runtime','invocations','assets','upstream','targets','__pycache__'}


class ILStateEvents(OwnerStateEvents):
    MAX_WATCHES=512
    METADATA_FALLBACK_SECONDS=60
    CAPACITY_ERRORS={errno.ENOSPC,errno.EMFILE,errno.ENFILE,errno.ENOMEM}

    def __init__(self, root, persist, directories=()):
        self.watch_errors={};self.degraded=False;self.next_metadata_check=0
        try:super().__init__(root, persist)
        except OSError as error:
            if error.errno not in self.CAPACITY_ERRORS:raise
            self.root=Path(root);self.reporting=self.root/'reporting';self.persist=persist
            self.fd=None;self.wake_read,self.wake_write=os.pipe2(os.O_NONBLOCK|os.O_CLOEXEC)
            self.directories={};self.seen=set(read_object(self.reporting/'EVENT_STATE.json').get('seen_signatures',[]))
            self.initial_signatures=0;self.degraded=True;self.watch_errors['inotify_init1']=str(error)
        self.add_sources(directories, initial=True)
        self.watch(Path('/mnt/project'))

    def watch(self,path):
        path=Path(path)
        if not path.is_dir()or path in self.directories.values():return
        if self.fd is None or len(self.directories)>=self.MAX_WATCHES:
            self.degraded=True;self.watch_errors[str(path)]='Bounded watch capacity; authoritative metadata fallback active';return
        try:super().watch(path)
        except OSError as error:
            if error.errno not in self.CAPACITY_ERRORS:raise
            self.degraded=True;self.watch_errors[str(path)]=str(error)

    def monitor_status(self):
        return dict(backend='linux_inotify_with_metadata_fallback'if self.degraded else'linux_inotify',
            watched_directories=len(self.directories),watch_limit=self.MAX_WATCHES,
            degraded=self.degraded,failed_watch_count=len(self.watch_errors),
            metadata_fallback_seconds=self.METADATA_FALLBACK_SECONDS if self.degraded else None,
            training_journal_polling=False)

    def save(self):
        self.persist(self.reporting/'EVENT_STATE.json',dict(schema='world_IL_report_event_state_v2',
            seen_signatures=sorted(self.seen),initial_state_signatures=self.initial_signatures,
            **self.monitor_status()))

    def metadata_events(self):
        # Only authoritative queue/control requests; never scan scene records,
        # training journals or every historical evaluation directory each minute.
        paths=[self.reporting/'refresh_request.json',self.root/'navtest_cadence_20261031/STATE.json']
        new={}
        for path in paths:
            if path.is_file():new.update({k:v for k,v in self.signatures(path).items()if k not in self.seen})
        if new:self.seen.update(new);self.save()
        return list(new.values())

    def close(self):
        for fd in(self.fd,self.wake_read,self.wake_write):
            if fd is not None:os.close(fd)

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
        if self.degraded and time.monotonic()>=self.next_metadata_check:
            self.next_metadata_check=time.monotonic()+self.METADATA_FALLBACK_SECONDS
            new=self.metadata_events()
            if new:return new
        if self.degraded:timeout=min(timeout,max(0,self.next_metadata_check-time.monotonic()))
        ready,_,_=select.select(([self.fd]if self.fd is not None else[])+[self.wake_read],[],[],max(0,timeout))
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
