"""Complete-NAV checkpoint selection only. Never supplies TRAIN IDs or feedback.

Run once after a completed result or call refresh_selection from the CPU eval
owner. All checkpoints and results remain immutable and retained.
"""
import argparse,fcntl,json,os,time
from pathlib import Path
import numpy as np
from .launch import ROOT,read,sha,publish,inspect_native
from starVLA.rl.qwen_world_research_data.manifest import fingerprint,sealed,read_sealed

POLICY=dict(schema='world_performance_complete_NAV_checkpoint_selection_v1',count=12146,physical_logs=44,
    seed=2101,invalid_scenes=0,fixed_high_threshold=.9,stable_score_band_PDMS=.10,
    reference='first actual COMPLETE NAV publication per logical run; frozen once',
    best_score='maximum full-population PDMS; ties choose earlier native update',
    best_stable='within bestPDMS-.10; lexicographic minimum fixed-high gross loss, newly-zero count; then higher PDMS, earlier update',
    last='largest native update with COMPLETE verified NAV; unscored last-native is separate',
    NavtestSelected=True,used_for_training_sample_selection=False,
    caveat='Fixed high cohort was conditionally selected on first model; stability diagnostics do not prove causal forgetting',
    keep_all_checkpoints=True)


def validate_summary(value):
    body=dict(value);digest=body.pop('sha256',None)
    if(digest!=fingerprint(body)or value.get('status')!='COMPLETE'or value.get('cohort')!='NAVTEST'
       or value.get('schema')!='world_rl_research_evaluation_summary_v1'
       or value.get('scenes')!=12146 or value.get('expected_scenes')!=12146
       or value.get('physical_logs')!=44 or value.get('invalid_scenes')!=0
       or value.get('seed')!=2101 or value.get('train_feedback_allowed')is not False):
        raise ValueError('Only exact COMPLETE12146/44logs/invalid0 canonical NAV may select checkpoints')
    if not isinstance(value.get('PDMS'),(int,float))or not np.isfinite(value['PDMS'])or not 0<=value['PDMS']<=100:
        raise ValueError('Invalid complete NAV score')
    return value


def compact(run,entry,cache_root):
    output=Path(entry['evaluation']).resolve();summary_path=output/'SUMMARY.json';spec_path=output/'SPEC.json';identity_path=output/'identity.json'
    if not output.is_relative_to(ROOT/'evaluations'):raise ValueError('Foreign evaluation output')
    summary=validate_summary(read(summary_path));spec=read_sealed(spec_path);identity=read_sealed(identity_path)
    if(spec['arm']!=run['id']or spec['updates']!=summary['RL_updates']or spec['source']!=run['source']
       or spec['config_sha256']!=run['config_sha256']or spec['cohort']!='NAVTEST'or spec['count']!=12146
       or spec['seed']!=2101 or entry['spec_sha256']!=spec['sha256']or entry['world']!=spec['world']
       or identity['sha256']!=summary['evaluation_identity_sha256']or identity['model_sha256']!=summary['model_sha256']
       or identity['token_sha256']!=summary['token_sha256']):raise ValueError('Changed evaluation/model/spec binding')
    cp=inspect_native(run,spec['checkpoint']['path'])
    if(Path(identity['native_RL_checkpoint']).resolve()!=Path(cp['path'])
       or Path(identity['native_parent_checkpoint']).resolve()!=Path(run['parent_checkpoint']).resolve()
       or Path(spec['parent_checkpoint']).resolve()!=Path(run['parent_checkpoint']).resolve()):
        raise ValueError('Selected model/native IL/RL checkpoint differs')
    if cp['updates']!=spec['updates']or cp['files_sha256']!=spec['checkpoint']['files_sha256']:raise ValueError('Selected native checkpoint changed')
    source=dict(summary_sha256=sha(summary_path),spec_sha256=sha(spec_path),identity_sha256=sha(identity_path))
    stem=cache_root/identity['sha256'];meta_path=stem.with_suffix('.json');data_path=stem.with_suffix('.npz')
    if meta_path.exists():
        meta=read_sealed(meta_path)
        if meta['source']!=source or meta['npz_sha256']!=sha(data_path):raise ValueError('Selection compact cache changed')
        with np.load(data_path,allow_pickle=False)as a:tokens=a['tokens'].copy();scores=a['scores'].copy()
    else:
        from tools.qwen_world_safety_exploration.evaluate import load_evaluation
        checked,rows=load_evaluation(output)
        tokens=np.array(checked['tokens']);scores=np.array([rows[t]['feedback']['metrics']['score']for t in tokens],dtype=np.float64)
        if len(tokens)!=12146 or len(set(tokens))!=12146 or not np.isfinite(scores).all()or np.any((scores<0)|(scores>1)):
            raise ValueError('Incomplete/invalid complete-scene scores')
        if abs(float(scores.mean())*100-summary['PDMS'])>1e-9:raise ValueError('Full scene scores and SUMMARY disagree')
        cache_root.mkdir(parents=True,exist_ok=True);tmp=data_path.with_suffix('.npz.tmp.'+str(os.getpid()))
        with tmp.open('wb')as f:np.savez_compressed(f,tokens=tokens,scores=scores)
        os.replace(tmp,data_path)
        publish(meta_path,sealed(dict(schema='world_performance_selection_compact_v1',source=source,npz_sha256=sha(data_path),count=12146)))
    return dict(label=entry['label'],updates=spec['updates'],PDMS=summary['PDMS'],checkpoint=cp,
        summary=str(summary_path),summary_sha256=source['summary_sha256'],spec_sha256=spec['sha256'],
        evaluation_identity_sha256=identity['sha256'],model_sha256=identity['model_sha256'],
        exposure_epoch=entry.get('registered_exposure_epoch'),valid_scene_exposures=entry.get('actual_valid_scene_exposures'),
        tokens=tokens,scores=scores,publication_mtime_ns=summary_path.stat().st_mtime_ns,compact=str(data_path))


def choose_points(points,reference_scores,threshold=.9,band=.10):
    """Pure full-population point selection, no score/candidate/model generation."""
    reference_scores=np.asarray(reference_scores,dtype=np.float64);high=reference_scores>=threshold
    out=[]
    for p in points:
        scores=np.asarray(p['scores'],dtype=np.float64)
        if scores.shape!=reference_scores.shape:raise ValueError('Different scene populations')
        q={k:v for k,v in p.items()if k not in('tokens','scores')}
        q['retention']=dict(fixed_high_count=int(high.sum()),high_gross_loss_PDMS=100*float(np.maximum(reference_scores[high]-scores[high],0).mean())if high.any()else 0.,
            high_net_delta_PDMS=100*float((scores[high]-reference_scores[high]).mean())if high.any()else 0.,
            new_zero=int(((reference_scores>0)&(scores==0)).sum()),repaired_zero=int(((reference_scores==0)&(scores>0)).sum()),
            high_to_zero=int((high&(scores==0)).sum()),zero=int((scores==0).sum()))
        out.append(q)
    best=min(out,key=lambda p:(-p['PDMS'],p['updates']))
    pool=[p for p in out if p['PDMS']>=best['PDMS']-band-1e-12]
    stable=min(pool,key=lambda p:(p['retention']['high_gross_loss_PDMS'],p['retention']['new_zero'],-p['PDMS'],p['updates']))
    return dict(best_score_checkpoint=best,best_stable=stable,last=max(out,key=lambda p:p['updates']),points=out)


def refresh_selection(registry=None,evaluation_state=None):
    registry=read(ROOT/'RUNS.json')if registry is None else registry
    evaluation_state=read(ROOT/'owners/RESEARCH_EVALUATION/STATE.json',{})if evaluation_state is None else evaluation_state
    directory=ROOT/'selections';directory.mkdir(parents=True,exist_ok=True)
    with(directory/'selection.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runs={r['id']:r for r in registry['runs']};results={};global_points=[]
        for arm,run in runs.items():
            entries=[e for e in evaluation_state.get('evaluations',{}).values()if e.get('arm')==arm and e.get('status')=='COMPLETE'and e.get('cohort')=='NAVTEST']
            if not entries:results[arm]=dict(status='WAITING_COMPLETE_NAVTEST');continue
            points=[compact(run,e,directory/'compact')for e in entries];points.sort(key=lambda p:(p['publication_mtime_ns'],p['updates']))
            folder=directory/arm;folder.mkdir(parents=True,exist_ok=True);refpath=folder/'FIXED_REFERENCE.json'
            if refpath.exists():
                reference=read_sealed(refpath);base=next((p for p in points if p['evaluation_identity_sha256']==reference['evaluation_identity_sha256']),None)
                if base is None or base['summary_sha256']!=reference['summary_sha256']:raise ValueError('Fixed first-complete reference disappeared/changed')
            else:
                base=points[0];reference=dict(schema=POLICY['schema'],run=arm,evaluation_identity_sha256=base['evaluation_identity_sha256'],summary_sha256=base['summary_sha256'],
                    checkpoint=base['checkpoint'],threshold=.9,high_tokens=base['tokens'][base['scores']>=.9].tolist(),
                    token_sha256=fingerprint(base['tokens'].tolist()),NavtestSelected=True,used_for_training_sample_selection=False)
                publish(refpath,sealed(reference))
            if any(not np.array_equal(p['tokens'],base['tokens'])for p in points):raise ValueError('Different per-scene token order/census')
            selected=choose_points(points,base['scores'])
            for name in('best_score_checkpoint','best_stable','last'):
                publish(folder/(name+'.json'),sealed(dict(schema=POLICY['schema'],run=arm,pointer=name,**selected[name],policy=POLICY,reference=str(refpath),updated_unix=time.time())))
            native=read(Path(run['root'])/'latest.json')
            if native:selected['last_native_checkpoint']=dict(path=native['checkpoint'],updates=native['updates'],
                scored=any(Path(native['checkpoint']).resolve()==Path(p['checkpoint']['path']).resolve()for p in points),meaning='Recovery pointer; never substitutes for complete evaluated last')
            results[arm]=dict(status='COMPLETE_SELECTION_FOR_AVAILABLE_RESULTS',**selected);global_points.append(dict(run=arm,**selected['best_score_checkpoint']))
        state=dict(schema=POLICY['schema'],status='COMPLETE_SELECTION_FOR_AVAILABLE_RESULTS'if global_points else'WAITING_COMPLETE_NAVTEST',policy=POLICY,runs=results,at_unix=time.time())
        if global_points:state['global_best_score_checkpoint']=min(global_points,key=lambda p:(-p['PDMS'],p['updates']))
        publish(directory/'STATE.json',sealed(state));return state


def main():
    argparse.ArgumentParser(__doc__).parse_args();state=refresh_selection();print(json.dumps(dict(status=state['status'],runs=len(state['runs']))))
if __name__=='__main__':main()
