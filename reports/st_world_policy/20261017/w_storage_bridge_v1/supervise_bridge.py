#!/usr/bin/env python3
"""Versioned entry to the unchanged one-shot supervisor; never auto-admits."""
from pathlib import Path
import sys

from storage_bridge import HERE, Reads, fixed, upstream_module, validate


def build_supervisor():
    # The unchanged supervisor still validates its original helper inventory,
    # source/environment, stage lease, locks, migration and checkpoint receipts.
    upstream_module('prepare_bindings')
    original=upstream_module('supervise_once')
    validate_original=original.validate
    supervise_original=original.supervise
    observed=fixed(Reads())['upstream']['observed_cli.py']['path']
    state={}

    def validate_stage(admission, preparation, stage, now):
        pin=admission.get('storage_compatibility')
        if pin is not None:
            validate(pin,preparation,stage,admission.get('storage_policy'),admission=admission)
        result=validate_original(admission,preparation,stage,now)
        state.update(admission=admission,preparation=preparation,stage=stage)
        return result

    def supervise_child(argv, **kwargs):
        # Only the original fixed observer call is redirected. No Popen/process
        # monkeypatch: original explicit resource supervisor remains the caller.
        if len(argv)!=10 or argv[0]!=original.PYTHON or argv[1]!=observed or argv[2::2]!=[
                '--admission','--admission-sha256','--attempt','--read-cap']:
            raise ValueError('Unexpected original supervised observer invocation')
        if not state: raise ValueError('Stage must pass original and bridge preflight first')
        admission=state['admission'];pin=admission.get('storage_compatibility')
        if pin is not None:
            validate(pin,state['preparation'],state['stage'],admission['storage_policy'],admission=admission)
            argv=[*argv];argv[1]=str(HERE/'observed_cli.py')
        return supervise_original(argv,**kwargs)

    original.validate=validate_stage
    original.supervise=supervise_child
    return original


if __name__=='__main__':
    raise SystemExit(build_supervisor().main())
