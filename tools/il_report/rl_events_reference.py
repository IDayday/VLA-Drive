"""Read owner-state notifications without polling training journals.

Only metadata files are watched. An event permits a normal read-only report
render; it is never accepted as proof of a complete scientific result.
"""
from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import select
import struct
import time


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        ensure_ascii=False).encode()).hexdigest()


def read_object(path):
    try:
        value=json.loads(path.read_text())
        return value if isinstance(value,dict) else {}
    except (OSError,ValueError):
        return {}


def state_signatures(path):
    """Keep semantic completion identities stable across later progress writes."""
    data=read_object(path)
    signatures={}
    def add(kind,identity,detail):
        signatures[fingerprint([kind,identity])]={'kind':kind,'identity':identity,
            'source':str(path),**detail}
    def visit(node,arm=None):
        if isinstance(node,list):
            for child in node:visit(child,arm)
            return
        if not isinstance(node,dict):return
        arm=node.get('arm',node.get('id',arm))
        event=str(node.get('event',node.get('kind',''))).lower()
        label=node.get('label') or node.get('spec',{}).get('label')
        if event=='evaluation_complete':
            add(event,label or [arm,node.get('cohort'),node.get('updates')],dict(label=label))
        elif event=='training_stage_complete':
            add(event,[arm,node.get('updates')],dict(arm=arm,updates=node.get('updates')))
        elif event and any(term in event for term in ('failed','failure','exception','fatal','withdraw')):
            add('important_exception',[event,arm,label,node.get('job'),node.get('at')],dict(event=event,arm=arm))
        elif event in ('stage_selection_complete','performance_selection_complete','campaign_complete'):
            add(event,[arm,node.get('updates'),node.get('gate'),node.get('at')],dict(event=event))
        summary=node.get('summary',node)
        if isinstance(summary,dict) and summary.get('status')=='COMPLETE' and (
                summary.get('evidence') or summary.get('evaluation')):
            add('evaluation_complete',label or summary.get('evaluation') or summary.get('evidence'),dict(label=label))
        status=str(node.get('status','')).upper()
        if status in ('FAILED','FATAL','ABORTED','REPORT_RENDER_FAILED'):
            add('important_exception',[arm,label,status,node.get('job'),node.get('error')],dict(arm=arm,status=status))
        # Per-arm STATE may arrive before campaign's matching event.
        if arm and node.get('checkpoint') and node.get('updates')==node.get('stage_target_updates') and status in (
                'EVALUATING','STAGE_COMPLETE','WAITING_EVALUATION','COMPLETE'):
            add('training_stage_complete',[arm,node['updates']],dict(arm=arm,updates=node['updates']))
        for key,value in node.items():
            if isinstance(value,(dict,list)):
                if key=='runs' and isinstance(value,dict):
                    for name,child in value.items():visit(child,name)
                else:visit(value,arm)
    visit(data,path.parent.name)
    return signatures


class OwnerStateEvents:
    """Linux inotify, no periodic state or training-journal reads."""
    _HEADER=struct.Struct('iIII')
    _CLOSE_WRITE=0x8
    _MOVED_TO=0x80
    _CREATE=0x100
    _ISDIR=0x40000000
    _OVERFLOW=0x4000

    def __init__(self,root,persist):
        self.root=Path(root)
        self.reporting=self.root/'reporting'
        self.persist=persist
        self.libc=ctypes.CDLL(None,use_errno=True)
        self.libc.inotify_init1.argtypes=[ctypes.c_int]
        self.libc.inotify_init1.restype=ctypes.c_int
        self.libc.inotify_add_watch.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_uint32]
        self.libc.inotify_add_watch.restype=ctypes.c_int
        self.fd=self.libc.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC)
        if self.fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
        self.wake_read,self.wake_write=os.pipe2(os.O_NONBLOCK|os.O_CLOEXEC)
        self.directories={}
        self.seen=set(read_object(self.reporting/'EVENT_STATE.json').get('seen_signatures',[]))
        self.watch(self.root)
        self.watch(self.reporting)
        self.watch(self.root/'owners')
        for path in (self.root/'owners').glob('*'):
            if path.is_dir():self.watch(path)
        initial={}
        for path in (self.root/'owners').glob('*/STATE.json'):
            initial.update(state_signatures(path))
        self.initial_signatures=len(initial)
        self.seen.update(initial)
        for path in (self.root.glob('STAGE_POLICY_AMENDMENT*.json')):
            self.seen.update(self.signatures(path))
        self.seen.update(self.signatures(self.reporting/'refresh_request.json'))
        self.save()

    def watch(self,path):
        if not path.is_dir() or path in self.directories.values():return
        wd=self.libc.inotify_add_watch(self.fd,os.fsencode(path),self._CLOSE_WRITE|self._MOVED_TO|self._CREATE)
        if wd<0:raise OSError(ctypes.get_errno(),'inotify_add_watch',str(path))
        self.directories[wd]=path

    def signatures(self,path):
        if path.name=='STATE.json':return state_signatures(path)
        if path.name.startswith('STAGE_POLICY_AMENDMENT') and path.suffix=='.json':
            data=read_object(path)
            return {fingerprint(['stage_policy',data]):dict(kind='stage_policy_amended',source=str(path))} if data else {}
        if path==self.reporting/'refresh_request.json':
            data=read_object(path)
            # Administrative verification is explicitly separate from science.
            if data.get('id') and data.get('scientific_result') is False:
                return {fingerprint(['report_request',data['id']]):dict(kind='report_maintenance_request',
                    source=str(path),id=data['id'],scientific_result=False,reason=data.get('reason'))}
        return {}

    def save(self):
        self.persist(self.reporting/'EVENT_STATE.json',dict(schema='world_rl_report_event_state_v1',
            seen_signatures=sorted(self.seen),initial_state_signatures=self.initial_signatures,
            backend='linux_inotify',journal_polling=False))

    def wake(self):
        try:os.write(self.wake_write,b'x')
        except OSError:pass

    def wait(self,timeout):
        ready,_,_=select.select([self.fd,self.wake_read],[],[],max(0,timeout))
        if self.wake_read in ready:
            try:os.read(self.wake_read,4096)
            except BlockingIOError:pass
        if self.fd not in ready:return []
        paths=set()
        overflow=False
        while True:
            try:raw=os.read(self.fd,262144)
            except BlockingIOError:break
            position=0
            while position<len(raw):
                wd,mask,cookie,length=self._HEADER.unpack_from(raw,position)
                position+=self._HEADER.size
                name=os.fsdecode(raw[position:position+length].split(b'\0',1)[0])
                position+=length
                if mask&self._OVERFLOW:overflow=True;continue
                directory=self.directories.get(wd)
                if directory is None:continue
                path=directory/name
                if mask&self._ISDIR:
                    if directory==self.root/'owners':
                        self.watch(path)
                        paths.add(path/'STATE.json')
                    continue
                if mask&(self._CLOSE_WRITE|self._MOVED_TO):
                    if (name=='STATE.json' and directory.parent==self.root/'owners') or (
                            directory==self.root and name.startswith('STAGE_POLICY_AMENDMENT') and path.suffix=='.json') or (
                            path==self.reporting/'refresh_request.json'):
                        paths.add(path)
        if overflow:paths.update((self.root/'owners').glob('*/STATE.json'))
        new={}
        for path in paths:
            new.update({key:value for key,value in self.signatures(path).items() if key not in self.seen})
        if not new:return []
        self.seen.update(new)
        self.save()
        return list(new.values())

    def close(self):
        for fd in (self.fd,self.wake_read,self.wake_write):os.close(fd)
