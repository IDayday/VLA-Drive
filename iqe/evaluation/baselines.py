"""Offline selector controls; Oracle and uniform random never enter deployment."""
import numpy as np
from ..selector import SelectionRule
from .selection import selection_report


def evaluate_baselines(pipeline, number, role):
    d=pipeline.bank_arrays(number,role)
    scores,valid=d['scores'].numpy(),d['valid'].numpy()
    rng=np.random.default_rng(pipeline.config['seed'])
    choices={'always_base':np.zeros(len(scores),int),
        'uniform_random_offline':np.array([rng.choice(np.flatnonzero(v)) for v in valid]),
        'oracle_offline':np.where(valid,scores,-np.inf).argmax(1)}
    if number>0:
        module,rule,_=pipeline.calibrated_selector(number)
        candidates,prediction=pipeline.prediction_arrays(module,d)
        for label,selector in [('value_only_scorer',SelectionRule(0,{},{})),('calibrated_full_scorer',rule)]:
            _,ids,_=selector(candidates,prediction)
            choices[label]=np.array([d['expert_ids'].index(e) for e in ids])
    previous=pipeline.previous_indices(number,role,d)
    results={name:selection_report(scores,valid,selected,previous,{c:v.numpy() for c,v in d['components'].items()},
        {c:v.numpy() for c,v in d['component_valid'].items()},[s.source_group_id for s in d['scenes']]) for name,selected in choices.items()}
    return {'role':role,'pool_hash':d['manifest']['pool_hash'],'results':results,'offline_only':['uniform_random_offline','oracle_offline'],
            'science':'UNTESTED' if pipeline.mode=='smoke' else 'REPORTED'}
