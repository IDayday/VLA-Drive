"""Seal score-driven stage censuses using only legal NAVTRAIN diagnostics.

The mixture prioritizes absolute low scores and losses relative to the fixed IL
baseline, while keeping an explicit uniform probability floor for unmeasured or
already competent scenes. Each census is fixed before its optimizer stage.
"""
from collections import Counter
import bisect
import json
import math
from pathlib import Path
import random

from .manifest import atomic_json, file_sha, fingerprint, read_sealed, sealed, SCHEMA

BANK_SCHEMA = 'world_rl_research_diagnostic_bank_v1'
STAGE_SCHEMA = 'world_rl_research_stage_census_v1'

def validate_bank(path, *, manifest, legal_tokens):
    bank = read_sealed(path)
    if (bank.get('schema') != BANK_SCHEMA
            or bank.get('cohort') not in ('NAVTRAIN','TRAIN','RL_TRAIN_DIAGNOSTIC')
            or bank.get('data_manifest_sha256') != manifest['data_manifest_sha256']
            or bank.get('score_units') != '0..1'):
        raise ValueError('Adaptive selection requires matched legal TRAIN/NAVTRAIN diagnostics')
    records = bank.get('records')
    if not isinstance(records,dict) or not records or not set(records) <= legal_tokens:
        raise ValueError('Diagnostic bank contains an unregistered, held-out, or duplicate token')
    for token, row in records.items():
        value = row.get('score')
        if type(value) not in (int,float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid canonical diagnostic score for '+token)
    return bank

def priority_weights(tokens, current_scores, baseline_scores, *, uniform_fraction=.35,
                     low_score_fraction=.4, regression_fraction=.25):
    fractions = [uniform_fraction,low_score_fraction,regression_fraction]
    if (any(not math.isfinite(v) or v < 0 for v in fractions)
            or uniform_fraction <= 0 or not math.isclose(sum(fractions),1.,abs_tol=1e-12)):
        raise ValueError('Priority mixture must sum to one with a positive uniform floor')
    n = len(tokens)
    if not n or len(tokens)!=len(set(tokens)):
        raise ValueError('Priority population must be unique and nonempty')
    legal = set(tokens)
    if not set(current_scores) <= legal or not set(baseline_scores) <= legal:
        raise ValueError('Diagnostic scores outside training population')
    for values in (current_scores,baseline_scores):
        if any(type(v) not in (float,int) or not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
            raise ValueError('Scores must be canonical finite values in 0..1')
    low = {t:1-current_scores[t] for t in current_scores}
    regression = {t:max(baseline_scores[t]-current_scores[t],0.)
                  for t in current_scores.keys() & baseline_scores.keys()}
    low_sum, regression_sum = sum(low.values()),sum(regression.values())
    # If an adaptive component has no evidence, assign its mass to uniform
    # coverage rather than inventing failures for unmeasured scenes.
    uniform_mass = uniform_fraction + (low_score_fraction if low_sum==0 else 0) + (regression_fraction if regression_sum==0 else 0)
    weights = [uniform_mass/n
               + (low_score_fraction*low.get(t,0)/low_sum if low_sum else 0)
               + (regression_fraction*regression.get(t,0)/regression_sum if regression_sum else 0)
               for t in tokens]
    if not math.isclose(sum(weights),1.,abs_tol=1e-10) or min(weights) <= 0:
        raise ValueError('Invalid normalized adaptive sampling distribution')
    return weights,dict(uniform_fraction=uniform_fraction,low_score_fraction=low_score_fraction,
                        regression_fraction=regression_fraction,effective_uniform_mass=uniform_mass,
                        measured_scenes=len(current_scores),paired_baseline_scenes=len(current_scores.keys()&baseline_scores.keys()),
                        observed_regressed_scenes=sum(v>0 for v in regression.values()),
                        unknown_scenes=n-len(current_scores),min_selection_probability=min(weights),
                        max_selection_probability=max(weights),importance_correction=False,
                        objective='explicit adaptive training distribution; not an unbiased full-NAVTRAIN gradient')

def stage_census(tokens, weights, exposures, seed, *, world_size=8, adaptive=False):
    """Allow useful repetition across batches, while keeping each batch distinct.

    Stage-wide sampling without replacement would cap a 512-scene diagnostic
    panel at 12.5% of a 4096-exposure stage, defeating the intended prioritization.
    Adaptive stages therefore resample across batches. Within each global batch
    draws are conditioned on distinct scene identity to preserve scene diversity.
    Initial uniform stages use complete shuffled pool cycles for broad coverage.
    """
    if exposures <= 0 or len(weights)!=len(tokens) or any(v<=0 for v in weights):
        raise ValueError('Invalid stage population, probabilities, or exposure budget')
    if world_size <= 0 or exposures % world_size or len(tokens) < world_size:
        raise ValueError('Distinct-scene batches require an aligned budget and sufficient scenes')
    rng=random.Random(seed);result=[]
    if adaptive:
        cumulative=[];total=0.
        for value in weights:
            total+=value;cumulative.append(total)
        for _ in range(exposures//world_size):
            selected=set()
            while len(selected)<world_size:
                index=min(bisect.bisect_right(cumulative,rng.random()*total),len(tokens)-1)
                if index not in selected:
                    selected.add(index);result.append(tokens[index])
        return result
    while len(result)<exposures:
        ranked=sorted(range(len(tokens)),key=lambda i:-math.log(max(rng.random(),1e-300))/weights[i])
        result.extend(tokens[i] for i in ranked[:exposures-len(result)])
    return result

def prepare_stage(stream_manifest, stage_index, *, current_bank=None, baseline_bank=None,
                  uniform_fraction=.35,low_score_fraction=.4,regression_fraction=.25,
                  world_size=8):
    path=Path(stream_manifest);m=read_sealed(path)
    if m['schema']!=SCHEMA or m['mode']!='adaptive':
        raise ValueError('Stage census requires a separately registered adaptive stream')
    exposures=m['stage_exposures']
    if stage_index<0 or world_size<=0 or exposures%world_size:
        raise ValueError('Stage census does not align with the declared global batch')
    tokens=m['tokens'];legal=set(tokens);current={};baseline={};sources=[]
    protocol=None
    for label,bank_path in [('current',current_bank),('baseline',baseline_bank)]:
        if bank_path is None:continue
        bank=validate_bank(bank_path,manifest=m,legal_tokens=legal)
        current_protocol=fingerprint(bank['inference'])
        if protocol is not None and protocol!=current_protocol:
            raise ValueError('Current/baseline diagnostics must use the same sampler and seed protocol')
        protocol=current_protocol
        values={t:r['score'] for t,r in bank['records'].items()}
        if label=='current':current=values
        else:baseline=values
        sources.append(dict(role=label,path=str(Path(bank_path).resolve()),sha256=file_sha(bank_path),
                            bank_sha256=bank['sha256'],policy_sha256=bank['policy_sha256'],
                            cohort=bank['cohort'],count=len(values)))
    if bool(current_bank)!=bool(baseline_bank):
        raise ValueError('Adaptive stages require both current and fixed-IL diagnostic banks, or neither')
    weights,diag=priority_weights(tokens,current,baseline,uniform_fraction=uniform_fraction,
                                 low_score_fraction=low_score_fraction,regression_fraction=regression_fraction)
    stream_sha=file_sha(path)
    seed=int(fingerprint(dict(seed=m['seed'],stage_index=stage_index,stream_sha256=stream_sha))[:16],16)
    census=stage_census(tokens,weights,exposures,seed,world_size=world_size,adaptive=bool(current_bank))
    counts=Counter(census)
    stage=dict(schema=STAGE_SCHEMA,stream_manifest_sha256=stream_sha,stage_index=stage_index,
               start_cursor=stage_index*exposures,scene_exposures=exposures,
               ordered_tokens=census,mode='adaptive' if current_bank else 'uniform_initial',
               world_size=world_size,seed=seed,sources=sources,priority_diagnostics=diag,
               unique_scenes=len(counts),max_scene_repeats=max(counts.values()),
               selection=('weighted draws with replacement across batches; distinct scenes within each global batch; positive uniform probability floor'
                          if current_bank else 'uniform shuffled complete-pool cycles'),
               population='TRAIN/NAVTRAIN diagnostics only; never NAVTEST failures',
               census_selected_before_optimizer_stage=True)
    out=Path(m['stage_registry'])/f'stage{stage_index:06}.json'
    if out.exists():
        raise FileExistsError('Adaptive census is immutable; cannot overwrite '+str(out))
    atomic_json(out,sealed(stage))
    return read_sealed(out)
