"""Full source/PG/Adam/RNG boundaries plus separate synthetic-query identity."""
from starVLA.rl.qwen_world_performance import recovery as base
from starVLA.rl.qwen_world_performance.nominal_improvement import QUALITY_CONFIG
from starVLA.rl.qwen_world_verified_quality import calibration as C,recipe as S6
from starVLA.rl.qwen_world_safety_exploration.retention_material import digest
from .recipe import validate_recipe,PROTOCOL,PROTOCOL_SHA256
from .runtime import COUNTER_SCHEMA
from . import material as M


def validate_boundary(state,window,identity,optimizer=None,scheduler=None,*,legacy=False,expected_rank=None):
    validate_recipe(identity['recipe'])
    if (identity.get('structured_progress_protocol')!=PROTOCOL or identity.get('structured_progress_protocol_sha256')!=PROTOCOL_SHA256
            or identity.get('verified_quality_protocol')!=S6.PROTOCOL
            or identity.get('quality_protocol_sha256')!=S6.QUALITY_PROTOCOL_SHA256
            or identity.get('performance_implementation_protocols',{}).get('nominal')!=QUALITY_CONFIG
            or identity.get('performance_implementation_protocols',{}).get('calibration')!=C.PROTOCOL):raise ValueError('Wrong S7/S6 component protocol identity')
    reader=M.get_reader()
    if identity.get('structured_GT_source_identity')!=reader.identity:raise ValueError('Wrong immutable TRAIN GT data identity')
    if window is not None:
        epoch=(state['valid_scene_exposures']-window['valid_global_scenes'])/identity['recipe']['research']['performance']['population']
        for leaf in window['samples']:M.validate_nominal(leaf,rank=window['rank'],expected_epoch=epoch)
    result=base.validate_boundary(state,window,identity,optimizer,scheduler,legacy=legacy,expected_rank=expected_rank,
        recipe_validator=validate_recipe,nominal_config=QUALITY_CONFIG,calibration_validator=C.validate_calibration_state,
        nominal_validator=M.validate_nominal)
    own=state.get('structured_progress');rank=result['rank']
    if not isinstance(own,dict) or set(own)!= {'schema','protocol_sha256','rank','ledger'} or own['schema']!=COUNTER_SCHEMA or own['protocol_sha256']!=PROTOCOL_SHA256 or own['rank']!=rank:
        raise ValueError('Wrong native S7 rank-private query ledger')
    ledger=own['ledger']
    if len(ledger)!=state['collections'] or [v['collection'] for v in ledger]!=list(range(1,state['collections']+1)):
        raise ValueError('S7 queries must commit exactly once per fresh window')
    for entry in ledger:
        for k in ('local','global_counts'):
            if set(entry[k])!=set(('queries','triggered','enabled','GT_rejected','GT_qualified','B1_qualified')) or any(type(v)is not int or v<0 for v in entry[k].values()):raise ValueError('Invalid S7 cost/activation counters')
            c=entry[k]
            if c['queries']>16*c['triggered'] or c['enabled']>c['triggered'] or c['GT_qualified']+c['B1_qualified']>c['queries']:raise ValueError('S7 query/teacher count outside actual pools')
    global_ledger=[dict(collection=x['collection'],counts=x['global_counts']) for x in ledger]
    result.update(structured_protocol_sha256=PROTOCOL_SHA256,structured_global_ledger_sha256=digest(global_ledger),
        structured_total_global_queries=sum(x['global_counts']['queries'] for x in ledger),
        structured_total_local_queries=sum(x['local']['queries'] for x in ledger))
    if window is not None:
        if window.get('structured_progress_protocol_sha256')!=PROTOCOL_SHA256:raise ValueError('Pending S7 pool protocol changed')
        local=M.local_counts(window)
        if local!=window['structured_progress_counts_local'] or local!=ledger[-1]['local'] or window['structured_progress_counts_global']!=ledger[-1]['global_counts']:
            raise ValueError('Pending synthetic queries/activation differ from sealed fresh collection')
        result.update(structured_pending_local=local,structured_pending_global=window['structured_progress_counts_global'])
    return result


def validate_ranks(summaries):
    base.validate_ranks(summaries)
    for k in ('structured_protocol_sha256','structured_global_ledger_sha256','structured_total_global_queries'):
        if any(s[k]!=summaries[0][k] for s in summaries):raise ValueError('S7 ranks disagree on science/global cost ledger')
    if summaries[0]['structured_protocol_sha256']!=PROTOCOL_SHA256:raise ValueError('Not S7 scientific protocol')
    if sum(s['structured_total_local_queries'] for s in summaries)!=summaries[0]['structured_total_global_queries']:raise ValueError('Local synthetic queries do not sum to global actual cost')
    if summaries[0]['pending']:
        global_counts=summaries[0]['structured_pending_global']
        if any(s['structured_pending_global']!=global_counts for s in summaries):raise ValueError('Different pending global synthetic counts')
        if {k:sum(s['structured_pending_local'][k] for s in summaries) for k in global_counts}!=global_counts:raise ValueError('Pending global count is not sum of rank-private actual pools')
    return summaries
