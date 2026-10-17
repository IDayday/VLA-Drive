#!/usr/bin/env python3
"""Exact TRAIN64 artifact consumer for the original CPU fit300; no launch approval."""
from pathlib import Path
import sys

HERE=Path(__file__).absolute().parent
BRIDGE=HERE.parent/'w_storage_bridge_v1'
sys.path.insert(0,str(BRIDGE))
from storage_bridge import Reads,upstream_module
import train_artifact_receipt as receipt

BRIDGE_SHA='a6aa3a2ba5430ef588cd9811a4da373d64d05c17f042cade40324c810909089f'
FIT_PIN={'path':str(receipt.JOB/'controls_581399eb_v2_metadata_only/FIT_REGISTRATION.json'),
    'sha256':'7984d1ddedb9c769caa2649fd285af7782fe464f0c6a8b6c10f546fe64940d1e'}
KEY='train_artifact_reconciliation'
SOURCES='train_reconciliation_sources'


def load_original():
    Reads().raw(BRIDGE/'storage_bridge.py',BRIDGE_SHA)
    upstream_module('prepare_bindings')
    return upstream_module('supervise_once')


def install(original):
    """Only two preflight hooks; original observer, compute and monitor stay intact."""
    original_validate=original.validate
    original_stage_inputs=original.stage_inputs

    def require_fit(admission,stage):
        if (stage!='fit' or admission.get('stage')!='fit' or KEY not in admission
                or any(k in admission for k in ('manual_retry','previous_stage_result',
                    'previous_artifact_reconciliation','storage_compatibility'))):
            raise ValueError('Only explicit fixed TRAIN64 reconciliation to fit is supported')
        receipt.pin_shape(admission[KEY])
        sources=admission.get(SOURCES)
        if not isinstance(sources,dict) or set(sources)!={'supervise_fit.py','train_artifact_receipt.py'}:
            raise ValueError('Exact two new consumer source pins required')
        reads=Reads()
        for name,sha in sources.items():
            receipt.pin_shape({'path':str(HERE/name),'sha256':sha})
            reads.raw(HERE/name,sha)
        reads.raw(BRIDGE/'storage_bridge.py',BRIDGE_SHA)

    def validate_stage(admission,preparation,stage,now):
        require_fit(admission,stage)
        limits=original_validate(admission,preparation,stage,now)
        if (admission['preparation']!=receipt.PREPARATION_PIN
                or preparation.get('fit_registration')!=FIT_PIN):
            raise ValueError('Original exact preparation and fit registration required')
        original.read(FIT_PIN['path'],FIT_PIN['sha256'])
        receipt.validate_receipt(admission[KEY],
            failed_pin={'path':str(receipt.FAILED_PATH),'sha256':receipt.FAILED_SHA},
            preparation_pin=admission['preparation'],cache_root=receipt.TRAIN_ROOT,read=original.read)
        return limits

    def stage_inputs(admission,stage):
        if KEY not in admission:return original_stage_inputs(admission,stage)
        require_fit(admission,stage)
        original.check_manual_fields(admission,stage)
        previous,previous_pin=original.read(receipt.FAILED_PATH,receipt.FAILED_SHA)
        accepted=receipt.validate_receipt(admission[KEY],failed_pin=previous_pin,
            preparation_pin=admission['preparation'],cache_root=receipt.TRAIN_ROOT,read=original.read)
        if accepted['eligible_consumer']!='fit':raise ValueError('Wrong artifact consumer')
        accounting=dict(role_limit_bytes=original.FIT_READ_LIMIT,prior_read_bytes=0,sources=[],
            remaining_read_bytes=original.FIT_READ_LIMIT,upstream_artifact_reconciliation=admission[KEY],
            upstream_supervisor_status_preserved=previous['status'],
            scope='sum per attempt max(hash admitted + nominal torch.load, proc rchar); includes in-flight admitted hash; not physical IO')
        return original.JOB/'stages/fit',accounting,previous_pin

    original.validate=validate_stage
    original.stage_inputs=stage_inputs
    return original


if __name__=='__main__':
    raise SystemExit(install(load_original()).main())
