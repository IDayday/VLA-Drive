"""Check sealed native evaluation metadata and inline scene/feedback linkage.

Native trajectories and simulator feedback live inside each sealed scene JSON;
there is no separate trajectory payload whose checksum could be substituted.
These checks do not rerun inference/scoring or alter a legitimate score.
"""
import math
import numpy as np
from starVLA.rl.qwen_sourceflow.contracts import fingerprint

CORE_METRICS=('score','no_at_fault_collisions','drivable_area_compliance',
    'time_to_collision_within_bound','ego_progress','comfort','driving_direction_compliance')


def verify_identity(record):
    body=dict(record);seal=body.pop('sha256',None)
    if seal!=fingerprint(body):raise ValueError('Changed evaluation identity seal')
    expected=dict(schema='native_qwen_flow_evaluation_v1',metric='canonical_NAVSIM_v1.1',
        seed=2101,inference='single_candidate_source_shift5_ODE_K10')
    if any(record.get(k)!=v for k,v in expected.items()):
        raise ValueError('Unregistered native evaluation protocol')
    protocol=dict(record['protocol']);sha=protocol.pop('sha256',None)
    if (sha!=fingerprint(protocol) or
            protocol.get('name')!='navsim_v1.1_independent_reference_candidate'):
        raise ValueError('Changed canonical scoring protocol seal')
    return record


def verify_scene(record,identity,*,token=None,log=None):
    row=dict(record);seal=row.pop('sha256',None)
    if seal!=fingerprint(row):raise ValueError('Changed inline native scene payload')
    if row['identity']!=identity['sha256'] or row['status']!='OK':
        raise ValueError('Scene has wrong evaluation identity or invalid status')
    if (token is not None and row['token']!=token) or (log is not None and row['log']!=log):
        raise ValueError('Scene differs from locked token/log population')
    feedback=row['feedback']
    if (feedback.get('status')!='OK' or feedback.get('token')!=row['token'] or
            feedback.get('log')!=row['log'] or
            feedback.get('protocol_sha256')!=identity['protocol']['sha256'] or
            feedback.get('protocol')!=identity['protocol']['name']):
        raise ValueError('Official feedback is associated with a different scene/protocol')
    for key in CORE_METRICS:
        if key not in feedback['metrics'] or not math.isfinite(float(feedback['metrics'][key])):
            raise ValueError('Missing/nonfinite official metric: '+key)
    for key in ('normalized_action','requested_trajectory'):
        value=np.asarray(row[key],dtype=np.float64)
        if value.shape!=(8,3) or not np.isfinite(value).all():
            raise ValueError('Invalid native action/request payload: '+key)
    actual_request=np.asarray(feedback['requested_trajectory'],dtype=np.float64)
    if not np.array_equal(actual_request,np.asarray(row['requested_trajectory'],dtype=np.float64)):
        raise ValueError('Scored request differs from saved policy request')
    return row
