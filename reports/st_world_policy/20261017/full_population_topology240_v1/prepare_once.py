#!/usr/bin/env python3
"""Pinned adapter over existing reviewed bounded controls; no new trainer/monitor."""
import hashlib, os, stat
from pathlib import Path
from types import ModuleType
P=Path(__file__).resolve().with_name('adapter.py')
fd=os.open(P,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
with os.fdopen(fd,'rb') as f:
    info=os.fstat(f.fileno())
    if not stat.S_ISREG(info.st_mode) or info.st_size>16384:raise ValueError('Small regular adapter required')
    source=f.read(16385)
if hashlib.sha256(source).hexdigest()!='1368a78a606b32152fccfa0a2102e1b054a4c425e5dd483997e38a7b84e3416f':raise ValueError('Adapter source changed')
adapter=ModuleType('fixed_topology240_adapter');adapter.__file__=str(P)
exec(compile(source,str(P),'exec'),adapter.__dict__)
if __name__=='__main__':raise SystemExit(adapter.load('prepare').main())
