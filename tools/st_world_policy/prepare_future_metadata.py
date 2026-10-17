"""Registered, bounded future image metadata; no teacher, tensors or training.

plan is metadata/stat only. build requires a separately issued registration.
All five target times are retained; time-pair/patch sampling remains downstream.
"""
import argparse
from contextlib import contextmanager
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import os
from pathlib import Path
import resource
import stat
import sys
import threading
import time
import uuid

import numpy as np
from PIL import Image

from starVLA.dataloader.st_world_history import VIEWS, resolve_camera_path
from starVLA.dataloader.st_world_schema import identity_hash, require_hash
from starVLA.dataloader.st_world_targets import validate_target_manifest
from starVLA.dataloader.st_world_metadata_index import _atomic_rows, canonical, version, file_sha256, _storage_check
from starVLA.dataloader.st_world_future_provenance import (
    LogCoverage, LEDGER_SCHEMA, REF_SCHEMA, validate_ledger)
from tools.st_world_policy.prepare_current_teacher import source_population
from tools.st_world_policy.prepare_history import TrustedNumpyUnpickler

MiB, GiB = 1 << 20, 1 << 30
TIMES = (0., .5, 1., 2., 4.)
TOLERANCE = .05
PLAN_SCHEMA = 'st_future_metadata_plan_v1'
REG_SCHEMA = 'st_future_metadata_registration_v1'
SHARD_SCHEMA = 'st_future_metadata_log_shard_v1'
METADATA_STORAGE_CALLBACK_PROTOCOL = 'st_metadata_storage_check_v1'
METADATA_STORAGE_CALLBACK_DEPENDENCIES = (
    'starVLA.dataloader.st_world_metadata_index',
    'starVLA.dataloader.st_world_future_provenance',
    'tools.st_world_policy.prepare_current_teacher',
)


def sha_bytes(data): return hashlib.sha256(data).hexdigest()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

def signed(value):
    _storage_check(); result=dict(value, identity=identity_hash(value)); _storage_check()
    return result


def metadata_path(path, *, missing=False, directory=False):
    """Never resolve an alias into the admitted metadata tree."""
    path=Path(path)
    if '..' in path.parts: raise ValueError('Noncanonical metadata path')
    path=path.absolute()
    for parent in reversed(path.parents):
        info=parent.lstat()
        if not stat.S_ISDIR(info.st_mode): raise ValueError('Metadata ancestor is not a plain directory: '+str(parent))
    try: info=path.lstat()
    except FileNotFoundError:
        if missing: return path
        raise
    if directory:
        if not stat.S_ISDIR(info.st_mode): raise ValueError('Metadata root is not a plain directory')
    elif not stat.S_ISREG(info.st_mode) or info.st_nlink!=1:
        raise ValueError('Metadata must be an unlinked-alias-free regular file: '+str(path))
    return path


def metadata_inventory(root):
    """No symlink traversal; include orphan partials in the disk allowance."""
    root=metadata_path(root,directory=True); total=0; pending=[root]
    while pending:
        _storage_check()
        directory=pending.pop()
        with metadata_parent(directory/'__inventory__') as (_,fd):
            with os.scandir(fd) as entries:
                for entry in entries:
                    _storage_check()
                    info=entry.stat(follow_symlinks=False); path=directory/entry.name
                    if stat.S_ISDIR(info.st_mode): pending.append(path)
                    elif stat.S_ISREG(info.st_mode) and info.st_nlink==1: total+=info.st_size
                    else: raise ValueError('Unsafe metadata entry (link/nonregular): '+str(path))
    return total


@contextmanager
def metadata_parent(path, *, create=False):
    """Walk using no-follow directory descriptors, including missing subdirs."""
    path=Path(path)
    if '..' in path.parts: raise ValueError('Noncanonical metadata path')
    path=path.absolute(); fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for part in path.parent.parts[1:]:
            if create:
                try: os.stat(part,dir_fd=fd,follow_symlinks=False)
                except FileNotFoundError:
                    _storage_check(before_commit=True,path=path)
                    try: os.mkdir(part,dir_fd=fd)
                    except FileExistsError: pass
            try: following=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            except OSError as exc: raise ValueError('Unsafe metadata parent: '+str(path)) from exc
            os.close(fd); fd=following
        yield path,fd
    finally: os.close(fd)


@contextmanager
def metadata_reader(path):
    path=metadata_path(path)
    with metadata_parent(path) as (path,parent):
        try: fd=os.open(path.name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=parent)
        except OSError as exc: raise ValueError('Unsafe metadata read') from exc
        with os.fdopen(fd,'rb') as stream:
            info=os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1: raise ValueError('Unsafe metadata read')
            yield stream
            metadata_path(path)
            after=path.stat()
            if (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)!=(
                    after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
                raise ValueError('Metadata read changed')


@contextmanager
def metadata_lock(path,budget):
    budget.scan(); budget.path(path,missing=True)
    _storage_check(before_commit=True,path=path)
    with metadata_parent(path) as (path,parent):
        try: fd=os.open(path.name,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600,dir_fd=parent)
        except OSError as exc: raise ValueError('Unsafe metadata lock') from exc
        try:
            info=os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1: raise ValueError('Unsafe metadata lock')
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            budget.path(path); yield
        finally: os.close(fd)


def small_json(path, limit=MiB, expected_sha=None):
    path=metadata_path(path); before=version(path)
    if not 0 < before[2] <= limit: raise ValueError('Metadata file exceeds bounded read: '+str(path))
    with metadata_reader(path) as f:
        chunks=[]; remaining=limit+1
        while remaining:
            _storage_check(); part=f.read(min(MiB,remaining)); chunks.append(part); remaining-=len(part)
            if not part: break
        data=b''.join(chunks)
    metadata_path(path)
    if version(path)!=before or len(data)!=before[2]: raise ValueError('Metadata changed during read')
    if expected_sha and sha_bytes(data)!=expected_sha: raise ValueError('Pinned metadata checksum mismatch')
    _storage_check(); value=json.loads(data); _storage_check()
    return value


def validate_self(value, schema):
    _storage_check()
    if value.get('schema')!=schema or value.get('identity')!=identity_hash({k:v for k,v in value.items() if k!='identity'}):
        raise ValueError('Metadata schema/semantic identity mismatch')
    _storage_check()
    return value


class BudgetExceeded(RuntimeError): pass


class PreparationBudget:
    """Aggregate read pacing and actual metadata allocation, including orphans."""
    def __init__(self, root, limits):
        self.root=metadata_path(root,directory=True); self.root_version=(self.root.stat().st_dev,self.root.stat().st_ino)
        self.limits=limits; self.lock=threading.RLock()
        self.started=time.monotonic(); self.deadline=self.started+limits['max_seconds']
        self.next_read=self.started; self.bytes=0; self.throttle_seconds=0.; self.read_categories={}; self.written=0
        self.disk_bytes=metadata_inventory(self.root)
        self.check()

    def check(self):
        _storage_check()
        root=metadata_path(self.root,directory=True).stat()
        if (root.st_dev,root.st_ino)!=self.root_version: raise ValueError('Metadata root changed')
        if time.monotonic()>=self.deadline: raise BudgetExceeded('TIME_BUDGET')
        if rss()>self.limits['max_rss_bytes']: raise BudgetExceeded('RSS_BUDGET')
        if self.disk_bytes>self.limits['max_metadata_bytes']: raise BudgetExceeded('METADATA_DISK_BUDGET')

    def path(self,path,*,missing=False):
        self.check(); path=Path(path).absolute()
        if not path.is_relative_to(self.root): raise ValueError('Metadata path outside admitted root')
        return metadata_path(path,missing=missing)

    def scan(self):
        self.check()
        if metadata_inventory(self.root)>self.limits['max_metadata_bytes']:
            raise BudgetExceeded('METADATA_DISK_BUDGET')

    def read_allowance(self,n,category="source"):
        self._io_allowance(n,category,True)

    def write_allowance(self,n):
        self._io_allowance(n,"metadata_write",False)

    def _io_allowance(self,n,category,is_read):
        self.check()
        with self.lock:
            if is_read and self.bytes+n>self.limits['max_source_read_bytes']: raise BudgetExceeded('SOURCE_READ_BYTES_BUDGET')
            now=time.monotonic(); start=max(now,self.next_read)
            self.next_read=start+n/(self.limits['read_mib_per_second']*MiB)
            # Reserve before IO: one aggregate limiter shared by all log workers.
            delay=self.next_read-now
            if self.next_read>=self.deadline: raise BudgetExceeded('TIME_BUDGET')
            if is_read:
                self.bytes+=n
                self.read_categories[category]=self.read_categories.get(category,0)+n
            else: self.written+=n
            self.throttle_seconds+=delay
        # Long aggregate throttles must not leave the shared guard unchecked.
        while delay>0:
            part=min(delay,1.); time.sleep(part); delay-=part; self.check()
        self.check()

    def reserve(self,n):
        with self.lock:
            self.check()
            if self.disk_bytes+n>self.limits['max_metadata_bytes']: raise BudgetExceeded('METADATA_DISK_BUDGET')
            self.disk_bytes+=n

    def snapshot(self):
        return {'source_bytes_read':self.bytes,'metadata_bytes_reserved':self.disk_bytes,
                'elapsed_s':time.monotonic()-self.started,'peak_rss_bytes':rss(),
                'aggregate_throttle_wait_s':self.throttle_seconds,'source_read_categories':dict(self.read_categories),
                'metadata_bytes_written':self.written}


class BudgetedReader:
    def __init__(self,path,budget):
        self.metadata_context=None
        if Path(path).absolute().is_relative_to(budget.root):
            budget.path(path); self.metadata_context=metadata_reader(path)
            self.stream=self.metadata_context.__enter__()
        else: self.stream=Path(path).open('rb')
        self.budget=budget
        suffix=Path(path).suffix.lower()
        self.category={'.pkl':'raw_log','.pt':'opaque_ego','.json':'metadata'}.get(suffix,'image_or_other')
    def read(self,n=-1):
        if n<0: n=os.fstat(self.stream.fileno()).st_size-self.stream.tell()
        # Do not allocate a whole raw log or image in one helper read.
        if n>MiB:
            chunks=[]
            while n:
                chunk=self.read(min(n,MiB)); chunks.append(chunk); n-=len(chunk)
                if not chunk: break
            return b''.join(chunks)
        remaining=max(0,os.fstat(self.stream.fileno()).st_size-self.stream.tell())
        self.budget.read_allowance(min(n,remaining),self.category); return self.stream.read(n)
    def readline(self,n=-1):
        # Pickle protocol text lines are tiny; binary blocks use read().
        n=65536 if n<0 else min(n,65536)
        value=bytearray()
        while len(value)<n:
            c=self.read(1); value+=c
            if not c or c==b'\n': return bytes(value)
        raise ValueError('Oversized pickle/header line')
    def readinto(self,target):
        data=self.read(len(target)); target[:len(data)]=data; return len(data)
    def seek(self,*args): return self.stream.seek(*args)
    def tell(self): return self.stream.tell()
    def close(self):
        if self.metadata_context is not None:
            self.metadata_context.__exit__(None,None,None); self.metadata_context=None
        else: self.stream.close()
    def __enter__(self): return self
    def __exit__(self,*_): self.close()


def budget_sha(path,budget):
    managed=Path(path).absolute().is_relative_to(budget.root)
    if managed: budget.path(path)
    before=version(path); digest=hashlib.sha256()
    with BudgetedReader(path,budget) as f:
        while block:=f.read(MiB): digest.update(block)
    if managed: budget.path(path)
    if version(path)!=before: raise ValueError('Source changed while hashing')
    return digest.hexdigest()


def fsync_dir(path):
    fd=os.open(path,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def write_chunks(path,chunks,budget):
    """Keep failed temporary files; never overwrite a different immutable artifact."""
    path=Path(path).absolute()
    if not path.is_relative_to(budget.root) or '..' in path.parts: raise ValueError('Metadata output outside admitted root')
    budget.scan()
    with metadata_parent(path,create=True) as (path,parent):
        budget.path(path,missing=True)
        if path.exists():
            expected=hashlib.sha256()
            for piece in chunks:
                _storage_check(); expected.update(piece)
            if budget_sha(path,budget)!=expected.hexdigest(): raise ValueError('Immutable artifact conflict: '+str(path))
            return
        temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
        _storage_check(before_commit=True,path=temporary)
        fd=os.open(temporary.name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=parent)
        with os.fdopen(fd,'wb') as f:
            for piece in chunks:
                for offset in range(0,len(piece),MiB):
                    chunk=piece[offset:offset+MiB]
                    budget.reserve(len(chunk)); budget.write_allowance(len(chunk)); f.write(chunk)
            f.flush(); os.fsync(f.fileno())
        budget.scan(); budget.path(temporary); budget.path(path,missing=True)
        if path.exists(): raise ValueError('Concurrent immutable metadata publication')
        _storage_check(before_commit=True,path=path)
        os.rename(temporary.name,path.name,src_dir_fd=parent,dst_dir_fd=parent); os.fsync(parent)
        budget.path(path); _storage_check(force=True)


def write_json(path,value,budget):
    _storage_check()
    data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    _storage_check()
    write_chunks(path,[data],budget)


def row_document_identity(header,rows):
    digest=hashlib.sha256(b'{'); first=True
    for key in sorted(set(header)|{'scenes'}):
        if key=='identity': continue
        digest.update((('' if first else ',')+canonical(key)+':').encode()); first=False
        if key!='scenes': digest.update(canonical(header[key]).encode()); continue
        digest.update(b'[')
        for i,row in enumerate(rows()):
            _storage_check()
            digest.update(((',' if i else '')+canonical(row)).encode())
        digest.update(b']')
    digest.update(b'}'); return digest.hexdigest()


def write_row_document(path,header,rows,budget):
    identity=row_document_identity(header,rows)
    header=dict(header,identity=identity)
    def chunks():
        yield b'{\n'
        keys=sorted(set(header)|{'scenes'})
        for i,key in enumerate(keys):
            comma=',' if i<len(keys)-1 else ''
            if key!='scenes':
                lines=json.dumps(header[key],sort_keys=True,indent=2,allow_nan=False).splitlines()
                text='  '+json.dumps(key)+': '+lines[0]
                if len(lines)>1: text+='\n'+'\n'.join('  '+v for v in lines[1:])
                yield (text+comma+'\n').encode(); continue
            iterator=iter(rows()); current=next(iterator,None)
            if current is None: yield ('  "scenes": []'+comma+'\n').encode(); continue
            yield b'  "scenes": [\n'
            while current is not None:
                _storage_check()
                following=next(iterator,None)
                text='\n'.join('    '+v for v in json.dumps(current,sort_keys=True,indent=2,allow_nan=False).splitlines())
                yield (text+(',' if following is not None else '')+'\n').encode(); current=following
            yield ('  ]'+comma+'\n').encode()
        yield b'}\n'
    write_chunks(path,chunks(),budget)
    return identity


class ImageMemo:
    def __init__(self,budget,max_bytes=64*MiB):
        self.budget=budget; self.maximum=max_bytes; self.bytes=0; self.values={}; self.hits=0
    def __call__(self,path):
        path=Path(path).resolve(); stamp=version(path); key=str(path)
        if key in self.values:
            ref,old=self.values[key]
            if stamp!=old: raise ValueError('Previously observed image changed')
            self.hits+=1; return dict(ref)
        digest=budget_sha(path,self.budget)
        with BudgetedReader(path,self.budget) as f:
            with Image.open(f) as image: wh=list(image.size)
        if version(path)!=stamp: raise ValueError('Image changed during metadata construction')
        ref={'path':key,'sha256':digest,'bytes':stamp[2],'mtime_ns':stamp[3],'original_wh':wh}
        # Conservative object allowance, not only serialized characters.
        size=2048+4*len(canonical(ref).encode())
        if self.bytes+size>self.maximum: raise BudgetExceeded('PER_LOG_SHA_MEMO_BUDGET')
        self.bytes+=size; self.values[key]=(ref,stamp); return dict(ref)


class TargetLogLookup:
    """Same frame selection as prepare_history.build_target_scene, once per log."""
    def __init__(self,frames):
        self.frames=frames; self.locations={}
        for i,f in enumerate(frames):
            _storage_check(); self.locations[f['token']]=i
        if len(self.locations)!=len(frames): raise ValueError('Duplicate raw token')
        self.times=np.array([int(f['timestamp']) for f in frames],dtype=np.int64)
        if not (np.diff(self.times)>0).all(): raise ValueError('Nonmonotonic raw log')
    def scene(self,token,log,split,sensor_roots,reference):
        t0=int(self.times[self.locations[token]]); rows=[]
        for t in TIMES:
            _storage_check()
            wanted=t0+round(t*1e6); distance=np.abs(self.times-wanted)
            eligible=np.flatnonzero((distance<=round(TOLERANCE*1e6)) & ((self.times==t0) if t==0 else (self.times>t0)))
            frame=None
            if len(eligible):
                nearest=eligible[distance[eligible]==distance[eligible].min()]
                if len(nearest)!=1: raise ValueError('Ambiguous teacher timestamp')
                frame=self.frames[int(nearest[0])]
            refs=[]
            for name in VIEWS:
                cam=frame.get('cams',{}).get(name) if frame else None
                path=resolve_camera_path(cam['data_path'],sensor_roots) if cam else None
                refs.append(reference(path) if path is not None else None)
            if t==0 and any(ref is None for ref in refs): raise ValueError('Current image missing from registered population')
            rows.append({'requested_s':float(t),'timestamp_us':int(frame['timestamp']) if frame else None,
                         'frame_token':frame['token'] if frame else None,'views':refs})
        return {'token':token,'log':log,'split':split,'timestamp_us':t0,'frames':rows}


def source_with_pins(student,partition,split,current_population,budget=None):
    # Existing small source-control metadata only, not expanded future rows.
    for path in (Path(student)/'index.json',Path(partition)):
        if path.stat().st_size>128*MiB: raise ValueError('Source control metadata exceeds bounded preparation input')
    current=Path(current_population).resolve()
    if budget is not None:
        # Exact logical byte reads of the reused source_population + pin verifier:
        # reserve before IO; no monkeypatch/global read hook and no image access.
        reads=[(Path(student)/'identity.json',1),(Path(student)/'index.json',2),
            (Path(partition),2),(Path(student)/'ego_identity.json',2),
            (current/'REGISTRATION.json',2),(current/'COMPLETE.json',2),
            (current/'SELECTED_SCENES.json',3)]
        budget.read_allowance(sum(path.stat().st_size*count for path,count in reads),'control_metadata')
    _storage_check(force=True)
    source,index,binding=source_population(student,partition,split,resource_check=_storage_check)
    _storage_check(force=True)
    registration=validate_self(small_json(current/'REGISTRATION.json'),'st_current_population_registration_v1')
    done=small_json(current/'COMPLETE.json')
    selected=current/'SELECTED_SCENES.json'
    if (registration['source']!=binding or done.get('status')!='COMPLETE' or
            done.get('scope')!='current_only_metadata' or done['registration_identity']!=registration['identity'] or
            done['scenes']!=len(index) or done['logs']!=binding['logs'] or
            file_sha256(selected)!=done['selected_sha256']):
        raise ValueError('Original complete current population/pins mismatch')
    def consume(key,pos,row,encoded):
        if (pos>=len(index) or set(row)!=(set(index[pos])|{'current_record_sha256','ego_record_sha256'}) or
                any(row.get(k)!=v for k,v in index[pos].items())):
            raise ValueError('Selected source order or row differs from original population')
        for field in ('current_record_sha256','ego_record_sha256'):
            index[pos][field]=require_hash(row.get(field),field)
    selected_version=version(selected)
    header,counts,_=_atomic_rows(selected,{'scenes'},consume)
    if version(selected)!=selected_version or file_sha256(selected)!=done['selected_sha256']:
        raise ValueError('Pinned selected metadata changed while streaming')
    if (counts.get('scenes')!=len(index) or header.get('schema')!='st_fixed_scene_list_v1' or
            header['split']!=split or header['source_index_hash']!=source['index_sha256'] or
            header['source_student_identity']!=source['identity'] or
            header['ego_identity_sha256']!=binding['ego_identity_sha256']):
        raise ValueError('Original selected manifest source mismatch')
    pins={'path':str(selected),'sha256':done['selected_sha256'],
          'current_registration_identity':registration['identity'],
          'current_registration_sha256':file_sha256(current/'REGISTRATION.json'),
          'current_complete_sha256':file_sha256(current/'COMPLETE.json')}
    if budget is not None: budget.check()
    return source,index,binding,pins


def groups_for(rows):
    groups=OrderedDict()
    for row in rows:
        _storage_check()
        log=row['log']
        if Path(log).name!=log or log in ('.','..'): raise ValueError('Unsafe registered log name')
        groups.setdefault(log,[]).append(row)
    return groups


def validate_limits(limits):
    ceilings={'workers':2,'read_mib_per_second':64.,'max_seconds':14400.,
              'max_source_read_bytes':1024*GiB,'max_metadata_bytes':2*GiB,
              'max_rss_bytes':4*GiB,'max_raw_log_bytes':128*MiB}
    if set(limits)!=set(ceilings): raise ValueError('Unknown/missing preparation limits')
    for key,ceiling in ceilings.items():
        value=limits[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or not 0<value<=ceiling:
            raise ValueError('Preparation hard limit outside supported range: '+key)
        if key not in ('read_mib_per_second','max_seconds') and type(value) is not int:
            raise ValueError('Integer preparation limit required: '+key)
    return limits


def plan(args):
    limits=validate_limits({k:getattr(args,k) for k in ('workers','read_mib_per_second','max_seconds',
        'max_source_read_bytes','max_metadata_bytes','max_rss_bytes','max_raw_log_bytes')})
    out=Path(args.output).absolute()
    # Create through no-follow parents, without silently resolving a root alias.
    with metadata_parent(out/'PLAN.json',create=True): pass
    metadata_path(out,directory=True)
    budget=PreparationBudget(out,limits)
    source,index,binding,pins=source_with_pins(args.student_root,args.partition,args.split,args.current_population,budget)
    if args.scope=='qualification':
        if not 1<=args.max_scenes<=256: raise ValueError('Qualification requires <=256 scenes')
        # Fixed before any image/label reads: first two source logs, leading contiguous windows.
        groups=list(groups_for(index).items())[:2]
        each=(args.max_scenes+len(groups)-1)//len(groups)
        chosen={r['token'] for _,rows in groups for r in rows[:each]}
        index=[r for r in index if r['token'] in chosen][:args.max_scenes]
    elif args.scope!='full_population' or args.max_scenes!=binding['scenes']:
        raise ValueError('Full preparation must explicitly name the exact entire source population size')
    raw_root=Path(args.raw_log_root).resolve(); roots=[str(Path(p).resolve()) for p in args.sensor_root]
    if not raw_root.is_dir() or not roots or any(not Path(p).is_dir() for p in roots):
        raise ValueError('Registered raw-log/sensor mounts must exist')
    groups=groups_for(index)
    stats=[{'log':log,'path':str(raw_root/(log+'.pkl')),'bytes':(raw_root/(log+'.pkl')).stat().st_size} for log in groups]
    if any(r['bytes']>limits['max_raw_log_bytes'] for r in stats): raise ValueError('Raw log exceeds bounded per-log input; explicit replan required')
    # Two implementations, each hash + deserialize the raw file once.
    comparison_raw=4*sum(r['bytes'] for r in stats)
    if args.scope=='qualification' and comparison_raw>limits['max_source_read_bytes']//2:
        raise ValueError('Fixed qualification raw-log comparison cost exceeds half the read budget; register a smaller fixed log window')
    proposal={'schema':PLAN_SCHEMA,'status':'PLAN_ONLY','scope':args.scope,'source':binding,
        'original_selection':pins,'selected_rows':index if args.scope=='qualification' else None,
        'selected_scene_count':len(index),'selected_logs':list(groups),
        'ordered_scene_ids_hash':identity_hash([r['token'] for r in index]),
        'selection_identity':identity_hash(index),'target_times_s':list(TIMES),'timestamp_tolerance_s':TOLERANCE,
        'raw_log_root':str(raw_root),'sensor_roots':roots,'raw_log_stats':stats,
        'reference_plus_new_raw_read_estimate_bytes':comparison_raw,'limits':limits,
        'selection_rule':'first_two_source_logs_leading_contiguous_windows_in_original_order' if args.scope=='qualification' else 'entire_original_index_in_original_order',
        'implementation':{str(Path(p).relative_to(Path(__file__).resolve().parents[2])):file_sha256(p) for p in
            (__file__,sys.modules[validate_ledger.__module__].__file__,sys.modules[_atomic_rows.__module__].__file__,
             sys.modules[identity_hash.__module__].__file__,sys.modules[resolve_camera_path.__module__].__file__,
             sys.modules[source_population.__module__].__file__,sys.modules[TrustedNumpyUnpickler.__module__].__file__)},
        'created_utc':datetime.now(timezone.utc).isoformat(),'gpu_authorized':False,
        'teacher_extraction_authorized':False,'training_authorized':False,
        'aggregate_preparation_plus_index_budget_bytes':4*GiB,'aggregate_budget_status':'PROPOSED_NOT_RESERVED'}
    proposal=signed(proposal)
    with metadata_lock(out/'PLAN.lock',budget):
        path=out/'PLAN.json'
        if path.exists():
            old=validate_self(small_json(path,8*MiB),PLAN_SCHEMA)
            a={k:v for k,v in old.items() if k not in ('identity','created_utc')}
            b={k:v for k,v in proposal.items() if k not in ('identity','created_utc')}
            if a!=b: raise ValueError('Existing plan conflicts; new output namespace required')
            return old
        write_json(path,proposal,budget)
    return proposal


def verify_registration(plan_path,registration_path,registration_sha):
    reg=validate_self(small_json(registration_path,expected_sha=require_hash(registration_sha,'registration SHA')),REG_SCHEMA)
    p=validate_self(small_json(plan_path,8*MiB,expected_sha=require_hash(reg.get('plan_sha256'),'plan SHA')),PLAN_SCHEMA)
    if (reg.get('plan_sha256')!=file_sha256(plan_path) or reg.get('plan_identity')!=p['identity'] or
            reg.get('phase')!='data_preparation' or reg.get('approved_scope')!=p['scope'] or
            reg.get('gpu_authorized') is not False or reg.get('teacher_extraction_authorized') is not False or
            reg.get('training_authorized') is not False):
        raise ValueError('Build requires exact separate metadata-only registration')
    validate_limits(p['limits'])
    if p['scope'] not in ('qualification','full_population'):
        raise ValueError('Unknown future preparation scope')
    if p['scope']=='qualification' and (not 1<=p['selected_scene_count']<=256 or len(p['selected_rows'])!=p['selected_scene_count']):
        raise ValueError('Qualification exceeds fixed <=256 scene boundary')
    if p['scope']=='full_population' and (p['selected_rows'] is not None or p['selected_scene_count']!=p['source']['scenes']):
        raise ValueError('Full preparation requires unchanged complete source population')
    if p['target_times_s']!=list(TIMES) or p['timestamp_tolerance_s']!=TOLERANCE:
        raise ValueError('Registered future selection policy changed')
    for relative,digest in p['implementation'].items():
        if file_sha256(Path(__file__).resolve().parents[2]/relative)!=digest:
            raise ValueError('Registered implementation changed')
    return p,reg


def shard_path(output,log): return Path(output)/'log_shards'/(identity_hash({'log':log})+'.json')


def read_shard(path,registration,budget):
    budget.path(path); budget.read_allowance(Path(path).stat().st_size,'generated_metadata')
    value=small_json(path,64*MiB)
    validate_self(value,SHARD_SCHEMA)
    if value['registration_identity']!=registration['identity']: raise ValueError('Shard registration changed')
    return value


def validate_completed(done,p,reg,rows,groups,pins,output,budget):
    """A receipt is not success unless it describes these exact real shards."""
    expected={'schema':'st_future_metadata_complete_v1','status':'COMPLETE','scope':p['scope'],
        'registration_identity':reg['identity'],'plan_identity':p['identity'],'scenes':len(rows),'logs':len(groups),
        'original_selection':pins,'target_times_s':list(TIMES),'raw_log_deserialized':True,
        'ego_tensor_decoded':False,'teacher_extractions':0,'tensor_disk_bytes':0,'training_admission':False}
    if (not isinstance(done,dict) or set(done)!=set(expected)|{'request_identity','files','cost','log_costs'} or
            any(canonical(done[k])!=canonical(value) for k,value in expected.items())):
        raise ValueError('Completed preparation control fields do not match exact plan/population')
    require_hash(done['request_identity'],'completed request identity')
    names={'TEACHER_REQUESTS.json','RAW_LOG_PROVENANCE.json','SHARD_RECEIPTS.json'}
    if not isinstance(done['files'],dict) or set(done['files'])!=names:
        raise ValueError('Completed metadata file set changed')
    budget.scan()
    for name,digest in done['files'].items():
        if budget_sha(output/name,budget)!=require_hash(digest,'completed metadata SHA'):
            raise ValueError('Completed metadata changed')
    costs=done['log_costs']
    if (not isinstance(done['cost'],dict) or not isinstance(costs,list) or len(costs)!=len(groups) or
            [row.get('log') for row in costs]!=list(groups)):
        raise ValueError('Completed log cost coverage changed')
    receipts=small_json(output/'SHARD_RECEIPTS.json',8*MiB)
    if (set(receipts)!={'schema','registration_identity','shards'} or
            receipts['schema']!='st_future_metadata_shards_v1' or receipts['registration_identity']!=reg['identity'] or
            not isinstance(receipts['shards'],list) or len(receipts['shards'])!=len(groups)):
        raise ValueError('Completed shard receipt schema/coverage changed')
    expected_paths={shard_path(output,log) for log in groups}
    # Only orphan partials may coexist with the exact complete shard set.
    actual_paths=set()
    for path in (output/'log_shards').iterdir():
        budget.path(path)
        if path.suffix=='.json': actual_paths.add(path)
        elif path.suffix!='.partial': raise ValueError('Unexpected metadata in completed shard directory')
    if actual_paths!=expected_paths: raise ValueError('Completed shard file coverage changed')
    for (log,selected),record in zip(groups.items(),receipts['shards']):
        path=shard_path(output,log)
        if (set(record)!={'log','path','sha256'} or record['log']!=log or record['path']!=str(path) or
                budget_sha(path,budget)!=require_hash(record['sha256'],'completed shard SHA')):
            raise ValueError('Completed shard receipt identity changed')
        shard=read_shard(path,reg,budget)
        cover=LogCoverage()
        for row in selected: cover.add(row['token'],log)
        if (shard.get('log')!=log or shard.get('selected')!=selected or
                len(shard.get('scenes',[]))!=len(selected) or
                [{k:s.get(k) for k in ('token','log','split')} for s in shard['scenes']]!=
                [{'token':r['token'],'log':log,'split':p['source']['split']} for r in selected] or
                any(shard['raw_log'].get(k)!=v for k,v in cover.records()[0].items()) or
                shard['raw_log'].get('path')!=str(Path(p['raw_log_root'])/(log+'.pkl'))):
            raise ValueError('Completed shard does not cover registered scene/log population')
    # Later replay checks original raw/image versions and reconstructs all final
    # request/ledger bytes; no COMPLETE return is possible before that succeeds.


def build_log(p,reg,log,rows,output,budget):
    started=time.monotonic(); path=shard_path(output,log); root=Path(p['source']['student_root'])
    raw=Path(p['raw_log_root'])/(log+'.pkl'); stamp=version(raw)
    if stamp[2]>p['limits']['max_raw_log_bytes']: raise BudgetExceeded('RAW_LOG_SIZE_BUDGET')
    raw_sha=budget_sha(raw,budget); raw_hash_s=time.monotonic()-started
    if path.exists():
        shard=read_shard(path,reg,budget)
        if shard['selected']!=rows or shard['raw_log']['sha256']!=raw_sha or shard['raw_log']['path']!=str(raw):
            raise ValueError('Previously pinned log/selection changed')
        for row in rows:
            for field,folder,ext in [('current_record_sha256','current','.json'),('ego_record_sha256','ego','.pt')]:
                if budget_sha(root/folder/(row['token']+ext),budget)!=row[field]: raise ValueError('Previously pinned source record changed')
        for image in shard['image_versions']:
            if version(image['path'])!=image['version']: raise ValueError('Previously observed image changed')
        return {'log':log,'path':str(path),'sha256':budget_sha(path,budget),'reused':True}
    unpickle_started=time.monotonic()
    with BudgetedReader(raw,budget) as f: frames=TrustedNumpyUnpickler(f).load()
    unpickle_s=time.monotonic()-unpickle_started
    if version(raw)!=stamp: raise ValueError('Raw log changed during deserialization')
    budget.check(); lookup=TargetLogLookup(frames); memo=ImageMemo(budget); scenes=[]; scene_bytes=0
    for row in rows:
        budget.check(); token=row['token']; current=root/'current'/(token+'.json')
        if current.stat().st_size>MiB: raise ValueError('Oversized current source record')
        with BudgetedReader(current,budget) as f: data=f.read()
        if sha_bytes(data)!=row['current_record_sha256']: raise ValueError('Pinned current source changed')
        record=json.loads(data)
        if (record['identity']!=p['source']['source_identity'] or record['token']!=token or record['log']!=log):
            raise ValueError('Foreign current source record')
        if budget_sha(root/'ego'/(token+'.pt'),budget)!=row['ego_record_sha256']: raise ValueError('Pinned ego payload changed')
        scene=lookup.scene(token,log,p['source']['split'],p['sensor_roots'],memo)
        if scene['timestamp_us']!=record['timestamp'] or [ref['path'] for ref in scene['frames'][0]['views']]!=[str(Path(x).resolve()) for x in record['image_paths']]:
            raise ValueError('Raw current frame disagrees with original exact input')
        scenes.append(scene)
        scene_bytes+=len(canonical(scene).encode())
        if scene_bytes>32*MiB: raise BudgetExceeded('PER_LOG_SCENE_METADATA_BUDGET')
    cover=LogCoverage()
    for row in rows: cover.add(row['token'],log)
    raw_record={**cover.records()[0],'path':str(raw),'sha256':raw_sha,'bytes':stamp[2],'mtime_ns':stamp[3]}
    value=signed({'schema':SHARD_SCHEMA,'registration_identity':reg['identity'],'log':log,'selected':rows,
        'scenes':scenes,'raw_log':raw_record,
        'image_versions':[{'path':key,'version':v[1]} for key,v in memo.values.items()],
        'metrics':{'image_memo_hits':memo.hits,'unique_image_files':len(memo.values),
                   'memo_allowance_bytes':memo.bytes,'elapsed_s':time.monotonic()-started,
                   'raw_hash_s':raw_hash_s,'raw_unpickle_s':unpickle_s,
                   'unique_image_bytes':sum(v[0]['bytes'] for v in memo.values.values())}})
    # Identity must survive resume without invocation-dependent timing differences.
    # metrics belong in separate invocation evidence, not immutable shard content.
    metrics=value.pop('metrics'); value.pop('identity'); value=signed(value)
    if len(json.dumps(value,sort_keys=True,indent=2).encode())>64*MiB:
        raise BudgetExceeded('PER_LOG_SERIALIZED_METADATA_BUDGET')
    write_json(path,value,budget)
    return {'log':log,'path':str(path),'sha256':budget_sha(path,budget),'reused':False,'metrics':metrics}


class SceneStream:
    def __init__(self,rows,output,reg,budget): self.rows,self.output,self.reg,self.budget=rows,output,reg,budget
    def __call__(self):
        cached_log=None; cached=None
        for row in self.rows:
            self.budget.check()
            if row['log']!=cached_log:
                shard=read_shard(shard_path(self.output,row['log']),self.reg,self.budget)
                cached={s['token']:s for s in shard['scenes']}; cached_log=row['log']
            yield cached[row['token']]


def build(args):
    _storage_check(force=True)
    plan_path=metadata_path(args.plan); p,reg=verify_registration(plan_path,args.registration,args.registration_sha256)
    output=plan_path.parent; limits=p['limits']; budget=PreparationBudget(output,limits)
    source,index,binding,pins=source_with_pins(p['source']['student_root'],p['source']['partition_path'],p['source']['split'],Path(p['original_selection']['path']).parent,budget)
    if binding!=p['source'] or pins!=p['original_selection']: raise ValueError('Source population changed after planning')
    rows=index
    if p['scope']=='qualification':
        expected={r['token']:r for r in p['selected_rows']}
        rows=[r for r in index if r['token'] in expected]
        if rows!=p['selected_rows']: raise ValueError('Qualification source/order changed')
    if (len(rows)!=p['selected_scene_count'] or identity_hash(rows)!=p['selection_identity'] or
            identity_hash([r['token'] for r in rows])!=p['ordered_scene_ids_hash']):
        raise ValueError('Registered exact scene population changed')
    groups=groups_for(rows)
    if list(groups)!=p['selected_logs']: raise ValueError('Registered exact log population changed')
    with metadata_lock(output/'BUILD.lock',budget):
        complete=output/'COMPLETE.json'
        prior=None
        if complete.exists():
            done=small_json(complete)
            validate_completed(done,p,reg,rows,groups,pins,output,budget)
            prior=done
        completed=[]
        try:
            with ThreadPoolExecutor(max_workers=limits['workers']) as pool:
                iterator=iter(groups.items()); pending=deque()
                def submit():
                    log,selected=next(iterator)
                    return pool.submit(build_log,p,reg,log,selected,output,budget)
                for _ in range(min(limits['workers'],len(groups))): pending.append(submit())
                while pending:
                    completed.append(pending.popleft().result())
                    try: pending.append(submit())
                    except StopIteration: pass
            shared={'split':p['source']['split'],'source_student_identity':source['identity'],
                'source_index_sha256':source['index_sha256'],'partition_sha256':source['partition_sha256'],
                'selection_identity':p['selection_identity'],'registration_identity':reg['identity']}
            ledger_rows=[read_shard(shard_path(output,log),reg,budget)['raw_log'] for log in groups]
            ledger=signed({'schema':LEDGER_SCHEMA,**shared,'logs':ledger_rows})
            ledger_path=output/'RAW_LOG_PROVENANCE.json'; write_json(ledger_path,ledger,budget)
            header={'schema':'st_teacher_requests_v1',**shared,'view_names':list(VIEWS),
                'timestamp_tolerance_s':TOLERANCE,'times_s':list(TIMES),
                'source':'original_raw_log_future_frames_v1',
                'source_logs_ledger':{'schema':REF_SCHEMA,'ledger_schema':LEDGER_SCHEMA,'path':str(ledger_path),
                    'sha256':budget_sha(ledger_path,budget),'identity':ledger['identity']}}
            coverage=LogCoverage()
            for row in rows: coverage.add(row['token'],row['log'])
            budget.read_allowance(ledger_path.stat().st_size*2,'generated_metadata')
            validate_ledger(header,coverage=coverage.records())
            stream=SceneStream(rows,output,reg,budget)
            for scene in stream():
                one=signed(dict(header,scenes=[scene])); validate_target_manifest(one)
            request=output/'TEACHER_REQUESTS.json'
            request_id=write_row_document(request,header,stream,budget)
            # Re-read streamed metadata with the same bounded parser used by admission.
            actual_coverage=LogCoverage()
            def check_row(key,pos,row,encoded):
                if pos>=len(rows) or row['token']!=rows[pos]['token'] or row['log']!=rows[pos]['log']:
                    raise ValueError('Final future scene order/coverage mismatch')
                actual_coverage.add(row['token'],row['log'])
            budget.read_allowance(request.stat().st_size,'generated_metadata')
            h,counts,digest=_atomic_rows(request,{'scenes'},check_row)
            if counts['scenes']!=len(rows) or digest!=request_id or h['identity']!=request_id:
                raise ValueError('Final future metadata identity/coverage mismatch')
            budget.read_allowance(ledger_path.stat().st_size*2,'generated_metadata')
            validate_ledger(h,coverage=actual_coverage.records())
            receipt={'schema':'st_future_metadata_shards_v1','registration_identity':reg['identity'],
                'shards':[{k:r[k] for k in ('log','path','sha256')} for r in completed]}
            write_json(output/'SHARD_RECEIPTS.json',receipt,budget)
            files={name:budget_sha(output/name,budget) for name in ('TEACHER_REQUESTS.json','RAW_LOG_PROVENANCE.json','SHARD_RECEIPTS.json')}
            if prior is not None:
                if files!=prior['files'] or request_id!=prior['request_identity']:
                    raise ValueError('Completed metadata no longer matches verified sources')
                return dict(prior,reused=True,invocation_cost=budget.snapshot())
            done={'schema':'st_future_metadata_complete_v1','status':'COMPLETE','scope':p['scope'],
                'registration_identity':reg['identity'],'plan_identity':p['identity'],'scenes':len(rows),'logs':len(groups),
                'request_identity':request_id,'files':files,'original_selection':pins,
                'target_times_s':list(TIMES),'raw_log_deserialized':True,'ego_tensor_decoded':False,
                'teacher_extractions':0,'tensor_disk_bytes':0,'training_admission':False,'cost':budget.snapshot(),
                'log_costs':[{'log':r['log'],'reused':r['reused'],'metrics':r.get('metrics')} for r in completed]}
            write_json(complete,done,budget); return done
        except BudgetExceeded as exc:
            # Preserve complete shards and every partial file. No misleading COMPLETE.
            return {'status':'PARTIAL','reason':str(exc),'complete_logs_observed':len(completed),
                    'scope':p['scope'],'cost':budget.snapshot(),'teacher_extractions':0,'tensor_disk_bytes':0}


def parser():
    p=argparse.ArgumentParser(description=__doc__); commands=p.add_subparsers(dest='command',required=True)
    a=commands.add_parser('plan',help='Metadata/stat-only proposal; never scans images or issues approval')
    for flag in ('student-root','partition','current-population','raw-log-root','output'): a.add_argument('--'+flag,required=True)
    a.add_argument('--sensor-root',action='append',required=True)
    a.add_argument('--split',choices=['train','dev'],required=True)
    a.add_argument('--scope',choices=['qualification','full_population'],default='qualification')
    a.add_argument('--max-scenes',type=int,default=32)
    a.add_argument('--workers',type=int,default=2); a.add_argument('--read-mib-per-second',type=float,default=64.)
    a.add_argument('--max-seconds',type=float,default=900.); a.add_argument('--max-source-read-bytes',type=int,default=8*GiB)
    a.add_argument('--max-metadata-bytes',type=int,default=2*GiB); a.add_argument('--max-rss-bytes',type=int,default=4*GiB)
    a.add_argument('--max-raw-log-bytes',type=int,default=128*MiB)
    a=commands.add_parser('build',help='Requires separate pinned metadata-only registration')
    a.add_argument('--plan',required=True); a.add_argument('--registration',required=True); a.add_argument('--registration-sha256',required=True)
    return p


def main():
    args=parser().parse_args()
    try:
        result=plan(args) if args.command=='plan' else build(args)
        print(json.dumps(result,sort_keys=True)); return 3 if result.get('status')=='PARTIAL' else 0
    except Exception as exc:
        print(json.dumps({'status':'FAILED','error':repr(exc)}),file=sys.stderr); return 2


if __name__=='__main__': raise SystemExit(main())
