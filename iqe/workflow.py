"""Workflow rounds and expert expansion attempts have separate identities."""
from .io import read_json, digest, atomic_json, BlockedError
from .contracts import require


def expansion_history(pipeline, before_round):
    latest=pipeline.registry.latest(pipeline.registry.read())
    rows=[]
    for eid,event in latest.items():
        if eid=='expert_0' or event['created_round']>=before_round:
            continue
        result=event.get('gates',{}).get('oracle')
        if result is not None:
            rows.append({'expert_id':eid,'workflow_round':event['created_round'],
                         'oracle_gate':result['oracle_gate'],'status':event['status']})
    return sorted(rows,key=lambda row:row['workflow_round'])


def check_expansion(pipeline, number, selector_only=False, resume=False):
    latest=pipeline.registry.latest(pipeline.registry.read())
    pending=[e for e in latest.values() if 0 < e['created_round'] < number and e['status'] in {'training','frozen'}]
    if pending and not selector_only:
        raise BlockedError('UNRESOLVED_EXPERT_EXPANSION: finish/audit the preceding attempt before another expansion')
    history=expansion_history(pipeline,number)
    no_gain=0
    for row in reversed(history):
        if row['oracle_gate']:
            break
        no_gain+=1
    if pipeline.mode=='full' and not selector_only:
        if not pipeline.config['execution']['scientific_expansion']:
            raise BlockedError('SCIENTIFIC_EXPANSION_DISABLED: selector updates remain available')
        if no_gain>=pipeline.config['gates']['consecutive_no_gain_expansion_limit']:
            raise BlockedError('NO_COVERAGE_GAIN: consecutive real expansion attempts reached configured limit')
    pool={eid:h for eid,h in pipeline.registry.candidate_pool().items()
          if pipeline.registry.latest(pipeline.registry.read())[eid]['created_round']<number}
    require(selector_only or len(pool)<pipeline.config['model']['max_experts_including_base'] or resume,
            'configured accepted candidate budget reached')
    path=pipeline.round_root(number)/'WORKFLOW.json'
    value={'workflow_round':number,'kind':'selector_only' if selector_only else 'expert_expansion',
           'expansion_attempt_id':None if selector_only else 1+sum(0<e['created_round']<number for e in latest.values()),
           'prior_expansion_attempts':history,'consecutive_no_gain_attempts':no_gain,
           'old_candidate_pool':pool,'old_candidate_pool_version':digest(pool),
           'selector_update_id':number,'serving_version_source':'previous_serving.json; actual ACTIVE snapshot (temporary bundle only in smoke)'}
    if path.exists():
        require(read_json(path)==value,'workflow identity changed')
    atomic_json(path,value,immutable=True)
    return value
