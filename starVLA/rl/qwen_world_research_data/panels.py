"""Score-independent, nested physical-log-balanced TRAIN diagnostic panels."""
from collections import Counter, defaultdict
import json
from pathlib import Path

from .manifest import atomic_json, file_sha, fingerprint, physical, read_sealed, sealed

def balanced_order(tokens, logs, *, seed):
    groups=defaultdict(list)
    for token in tokens:groups[physical(logs[token])].append(token)
    order=sorted(groups,key=lambda log:fingerprint(dict(panel_seed=seed,physical_log=log)))
    for log in order:
        groups[log].sort(key=lambda token:fingerprint(dict(panel_seed=seed,token=token)))
    result=[];offset=0
    while len(result)<len(tokens):
        for log in order:
            if offset<len(groups[log]):result.append(groups[log][offset])
        offset+=1
    return result

def prepare_panels(data_root, output, *, seed=420301, sizes=(1024,4096)):
    data_root=Path(data_root).resolve();output=Path(output)
    data_path=data_root/'manifest.json';data=json.loads(data_path.read_text())
    complete=json.loads((data_root/'COMPLETE').read_text())
    data_sha=file_sha(data_path)
    if (data['schema']!='world_rl_research_current_observations_v1'
            or complete.get('status')!='COMPLETE' or complete['manifest_sha256']!=data_sha):
        raise ValueError('Diagnostic panels require complete explicit research NAVTRAIN observations')
    pop_path=Path(data['source_population_path'])
    if file_sha(pop_path)!=data['population_sha256']:raise ValueError('Population changed before panel selection')
    pop=json.loads(pop_path.read_text());tokens=data['tokens'];logs=data['token_logs']
    expected=set(pop['train_tokens']+pop['dev_tokens'])
    test=set(pop['navtest_tokens']);test_logs={physical(pop['token_logs'][t]) for t in test}
    if (set(tokens)!=expected or len(tokens)!=len(expected) or set(tokens)&test
            or any(physical(logs[t]) in test_logs for t in tokens)):
        raise ValueError('TRAIN diagnostic universe must be full legal NAVTRAIN and NAVTEST-disjoint')
    ordering=balanced_order(tokens,logs,seed=seed)
    requested=list(dict.fromkeys([int(v) for v in sizes]+[len(tokens)]))
    all_counts=Counter(physical(logs[t]) for t in tokens);result={}
    for count in requested:
        if not 0<count<=len(tokens):raise ValueError('Diagnostic panel size outside population')
        chosen=ordering[:count] if count<len(tokens) else list(tokens)
        counts=Counter(physical(logs[t]) for t in chosen)
        body=dict(schema='world_rl_research_evaluation_census_v1',cohort='TRAIN_DIAGNOSTIC',
                  name=f'TRAIN_DIAGNOSTIC_{count:06}',count=count,tokens=chosen,
                  token_logs={t:logs[t] for t in chosen},data_manifest_sha256=data_sha,
                  data_root=str(data_root),source_population_sha256=data['population_sha256'],
                  selection_rule='whole physical-log round-robin, SHA256 seed/token order; fixed before any policy score; nested 1024 within4096',
                  seed=seed,train_feedback_allowed=True,used_for_actor_update=True,
                  diagnostic_scope='full NAVTRAIN' if count==len(tokens) else 'fixed physically stratified panel; not a full-population scene-weighted estimate',
                  physical_logs=len(counts),physical_log_scene_counts=dict(counts),
                  universe_physical_log_scene_counts=dict(all_counts),
                  old_DEV_is_train_diagnostic=bool(set(chosen)&set(pop['dev_tokens'])),
                  old_DEV_scene_count=len(set(chosen)&set(pop['dev_tokens'])),
                  NAVTEST_token_overlap=0,NAVTEST_physical_log_overlap=0)
        value=sealed(body);name=f'TRAIN_DIAGNOSTIC_{count:06}.json';dest=output/name
        if dest.exists():
            if read_sealed(dest)!=value:raise FileExistsError('Fixed diagnostic census cannot change: '+str(dest))
        else:atomic_json(dest,value)
        result[count]=dict(path=str(dest.resolve()),count=count,sha256=value['sha256'],file_sha256=file_sha(dest),physical_logs=len(counts))
    return result
