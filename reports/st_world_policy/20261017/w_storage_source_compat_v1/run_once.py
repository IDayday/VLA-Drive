#!/usr/bin/env python3
"""Explicit compatibility wrapper for fixed matched probe stages; never auto-admits."""
with open('/proc/self/io') as f: BOOT_IO={k:int(v) for k,v in (line.split(':') for line in f)}
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types
import compat

HERE=Path(__file__).resolve().parent
CONTROL=HERE.parent/'matched_spatial_probe_controls_v1/control.py'
CONTROL_SHA='c5c8521379b36c979bf8743f12c2a2baf80df5ca5d21941ba2fe9d453540c38c'


def original_control():
    source=compat.raw(CONTROL,CONTROL_SHA).decode()
    old="[s.PYTHON,str(HERE/'control.py'),'observe','--arm',arm,*argv[2:]]"
    new="[s.PYTHON,str(COMPAT_OBSERVER),'observe','--arm',arm,*argv[2:]]"
    if source.count(old)!=1:raise ValueError('Fixed observer routing changed')
    module=types.ModuleType('matched_original_control');module.__file__=str(CONTROL)
    module.COMPAT_OBSERVER=HERE/'run_once.py'
    exec(compile(source.replace(old,new),str(CONTROL),'exec'),module.__dict__)
    return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('run','observe'));p.add_argument('--arm',choices=('SP1','SP2','SP3'),required=True)
    p.add_argument('--admission',required=True);p.add_argument('--admission-sha256',required=True)
    p.add_argument('--attempt');p.add_argument('--read-cap',type=int)
    a=p.parse_args();c=original_control()
    admission,_=c.document(a.admission,a.admission_sha256)
    own={'path':str(HERE/'run_once.py'),'sha256':hashlib.sha256(compat.raw(HERE/'run_once.py')).hexdigest()}
    if admission.get('source_compat_controller')!=own:raise ValueError('Exact compatibility controller pin required')
    pin=admission.get('frozen_W_source_compatibility')
    if not isinstance(pin,dict):raise ValueError('Explicit independently reviewed compatibility receipt required')
    compat.verify(pin)
    _,_,s=c.controls(a.arm,supervisor=True)
    if a.command=='run':
        sys.argv=[str(c.UP/'supervise_once.py'),'--admission',a.admission,'--admission-sha256',a.admission_sha256]
        return s.main()
    if a.attempt is None or a.read_cap is None:raise ValueError('Original observer arguments required')
    preparation,_=c.document(admission['preparation']['path'],admission['preparation']['sha256'])
    s.validate(admission,preparation,admission['stage'],time.time())
    fd=int(os.environ['ST_PROBE_SUPERVISOR_LOCK_FD']);os.fstat(fd)
    if os.getpid()!=os.getpgrp():raise ValueError('Original owned child session required')
    def observer_source(source):
        old='    baseline = proc_io()  # before torch/runtime imports, including their actual syscalls'
        new="    baseline = COMPAT_BOOT_IO\n    provenance = INSTALL_COMPAT()\n    write_once(Path(attempt) / 'SOURCE_COMPATIBILITY.json', provenance)"
        if source.count(old)!=1:raise ValueError('Original observer boundary changed')
        return source.replace(old,new)
    observer=c.module('observed_cli',observer_source)
    observer.COMPAT_BOOT_IO=BOOT_IO;observer.INSTALL_COMPAT=lambda:compat.install(pin)
    return observer.run(preparation,admission['stage'],a.attempt,admission.get('storage_policy'),
                        admission.get('checkpoint_commitments'),a.read_cap)


if __name__=='__main__':raise SystemExit(main())
