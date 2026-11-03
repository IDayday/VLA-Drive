"""Sleep until sealed metadata is published; no training-progress polling."""
import ctypes
import os
from pathlib import Path
import select
import struct
import time


def wait_for_events(paths,seconds=1800.,ignore=()):
    libc=ctypes.CDLL(None,use_errno=True)
    fd=libc.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC)
    if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
    watched=set()
    try:
        for value in paths:
            path=Path(value)
            while not path.exists()and path!=path.parent:path=path.parent
            if path.is_file():path=path.parent
            if str(path)in watched:continue
            # Atomic publication/close/create/delete. Open journal writes do
            # not wake this watcher on every Adam update.
            result=libc.inotify_add_watch(fd,os.fsencode(path),0x8|0x80|0x100|0x200)
            if result<0:raise OSError(ctypes.get_errno(),'inotify_add_watch',str(path))
            watched.add(str(path))
        if not watched:raise ValueError('No metadata directory can be watched')
        deadline=time.monotonic()+max(.01,float(seconds))
        while time.monotonic()<deadline:
            if not select.select([fd],[],[],max(0.,deadline-time.monotonic()))[0]:return False
            payload=os.read(fd,65536);offset=0
            while offset+16<=len(payload):
                _,_,_,length=struct.unpack_from('iIII',payload,offset)
                name=payload[offset+16:offset+16+length].split(b'\0',1)[0].decode(errors='replace')
                offset+=16+length
                if not any(name==item or name.startswith(item+'.')for item in ignore):return True
        return False
    finally:os.close(fd)
