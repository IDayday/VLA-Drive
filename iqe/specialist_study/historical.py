"""Matched-K Oracle analysis from immutable official per-scene labels only."""
from itertools import combinations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from iqe.io import atomic_json, atomic_bytes, file_hash, read_json
from iqe.contracts import require

POLICIES=('official_rl','interaction','road_direction','safe_progress')
SAFETY=('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound')


def bootstrap(delta, logs, repetitions, seed):
    unique,index=np.unique(logs,return_inverse=True)
    sums=np.bincount(index,weights=delta); counts=np.bincount(index)
    rng=np.random.default_rng(seed); result=[]
    for _ in range(repetitions):
        pick=rng.integers(len(unique),size=len(unique));result.append(sums[pick].sum()/counts[pick].sum())
    return np.quantile(result,[.025,.975]).tolist()


def subset_rows(scores, safe, logs, *, version, split, repetitions=2000, seed=11235):
    require(scores.shape==safe.shape and scores.shape[1]==4 and np.isfinite(scores).all(), 'historical K4 labels incomplete')
    out=[]
    for n in range(4):
        for added in combinations(range(1,4),n):
            cols=(0,)+added; v=scores[:,cols]; oracle=v.max(1); good=(v>=.9-1e-8)&safe[:,cols]
            row={'version':version,'split':split,'policies':'+'.join(POLICIES[i] for i in cols),'K':len(cols),
                 'scenes':len(scores),'logs':len(set(logs)),'base_PDMS_01':scores[:,0].mean(),'oracle_PDMS_01':oracle.mean(),
                 'oracle_gain_01':(oracle-scores[:,0]).mean(),'common_zero':int((v<=1e-8).all(1).sum()),
                 'safety_recovery':int((~safe[:,0]&safe[:,cols].any(1)).sum()),
                 'new_safe_high_quality':int((~((scores[:,0]>=.9)&safe[:,0])&good.any(1)).sum())}
            lo,hi=bootstrap(oracle-scores[:,0],logs,repetitions,seed)
            row.update(gain_ci_low_01=lo,gain_ci_high_01=hi)
            for i in range(4):
                row[POLICIES[i]+'_alone_01']=scores[:,i].mean()
                if i in added:
                    other=[j for j in cols if j!=i]
                    row[POLICIES[i]+'_leave_one_out_01']=(oracle-scores[:,other].max(1)).mean()
                    row[POLICIES[i]+'_unique_success']=int(((scores[:,i]>=.9)&safe[:,i]&~(((scores[:,other]>=.9)&safe[:,other]).any(1))).sum())
            out.append(row)
    return out


def audit(config):
    from .recipes import load
    recipe=load(config,'historical')
    require(recipe['include_Base_in_every_subset'] and recipe['use_official_labels_only'] and not recipe['reuse_K4_prediction_for_subsets'], 'historical subset protocol')
    require(recipe['bootstrap_repetitions']==config.bootstrap_repetitions and recipe['bootstrap_seed']==config.bootstrap_seed,'bootstrap configuration mismatch')
    root=Path(config.historical_root); sources=[]; rows=[]
    # Verify published evidence index before following actual asset paths.
    summary=Path(config.historical_summary)
    for name in ('README.md','SOURCE_MANIFEST.json','EVIDENCE.json','COHORTS.csv','VERIFICATION.json'):
        require((summary/name).is_file(), 'missing historical summary: '+name)
        sources.append({'path':str(summary/name),'sha256':file_hash(summary/name)})
    require(read_json(summary/'VERIFICATION.json')['status']=='PASS', 'historical publication did not pass verification')
    for split in ('navtrain','navtest'):
        original=root/f'research_iteration_v2/drivor_simscale134k_four_teacher_full_v1/analysis/{split}/per_scene.csv'
        frame=pd.read_csv(original)
        cols=('official_rl','interaction_ema','road_direction_ema','progress_comfort_ema')
        scores=np.stack([frame[c+'_score'].to_numpy() for c in cols],1)
        safe=np.stack([np.stack([frame[c+'_'+s].to_numpy()>=1-1e-8 for s in SAFETY],1).all(1) for c in cols],1)
        require(not frame.token.duplicated().any(),'duplicate historical scene')
        rows+=subset_rows(scores,safe,frame.source_log.to_numpy(),version='first_ema10k',split=split,repetitions=config.bootstrap_repetitions,seed=config.bootstrap_seed)
        sources.append({'path':str(original),'sha256':file_hash(original)})
        refined=root/f'research_iteration_v2/teacher_oracle_complementarity_v5/evaluation_specialists_v2/runs/{split}/{split}/scores.csv'
        fresh=pd.read_csv(refined)
        model_col=next((c for c in ('policy','model','model_name') if c in fresh),None)
        require(model_col is not None,'unrecognized refined per-candidate bank schema')
        cols=('official_rl','interaction_specialist_ema','road_direction_specialist_ema','progress_comfort_specialist_ema')
        frames=[fresh[fresh[model_col]==c].set_index('token') for c in cols]
        ids=frame.token.tolist()
        require(all(not f.index.duplicated().any() and set(f.index)==set(ids) for f in frames),'refined candidate ID population mismatch')
        frames=[f.loc[ids] for f in frames]
        require(all(np.array_equal(f.source_log.to_numpy(),frame.source_log.to_numpy()) for f in frames),'refined log identity mismatch')
        scorecol='score' if 'score' in frames[0] else 'PDMS_01'
        scores=np.stack([f[scorecol].to_numpy() for f in frames],1)
        safe=np.stack([np.stack([f[s].to_numpy()>=1-1e-8 for s in SAFETY],1).all(1) for f in frames],1)
        rows+=subset_rows(scores,safe,frame.source_log.to_numpy(),version='refined_ema4k',split=split,repetitions=config.bootstrap_repetitions,seed=config.bootstrap_seed)
        sources.append({'path':str(refined),'sha256':file_hash(refined)})
    out=config.root/'historical'; table=pd.DataFrame(rows)
    atomic_bytes(out/'HISTORICAL_SUBSET_RESULTS.csv',table.to_csv(index=False).encode(),immutable=True)
    atomic_json(out/'SOURCES.json',{'sources':sources,'historical_navtest_descriptive_only':True,'new_inference_calls':0,'used_DrivoR_predictions':False},immutable=True)
    text='Historical matched-K Oracle analysis\n\nScores are official [0,1] labels. No DrivoR predictions were truncated or rerun. Historical Navtest is descriptive only and is not used for the current pilot or hyperparameters.\n\n'+table[['version','split','policies','oracle_PDMS_01','oracle_gain_01','common_zero']].to_csv(index=False)
    atomic_bytes(out/'HISTORICAL_MATCHED_K_REPORT.md',text.encode(),immutable=True)
    return {'status':'COMPLETE','subsets':len(rows),'sources':len(sources)}
