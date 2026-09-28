import pytest
from tools.foresight.score_pdms import requested_population,identity_hash


def test_official_population_is_current_manifest_not_available_score_subset():
    current=[{'token':'a','log':'log1'},{'token':'b','log':'log2'},{'token':'c','log':'log2'}]
    metric=[dict(r,cache_path=r['token']+'.pkl') for r in reversed(current)]
    export={'current_identity':{'index_sha256':identity_hash(current)},'limit':0}
    rows=requested_population(current,metric,export)
    assert [r['token'] for r in rows]==['a','b','c']
    with pytest.raises(ValueError):requested_population(current,metric[:2],export)
    with pytest.raises(ValueError):requested_population(current[::-1],metric,export)
    export['limit']=2
    assert [r['token'] for r in requested_population(current,metric,export)]==['a','b']
    metric[0]['log']='wrong'
    export['limit']=0
    with pytest.raises(ValueError):requested_population(current,metric,export)


def test_sampling_failures_keep_full_denominator_and_only_zero_failed_run():
    from tools.foresight.summarize_experiments import collapse_sampling_runs,METRICS
    def row(log,value,status='ok'):
        return dict(log=log,status=status,**{m:value for m in METRICS})
    runs={seed:{'a':row('log1',1.),'b':row('log2',.5)} for seed in range(42,47)}
    runs[42]['a']=row('log1',float('nan'),'failed')
    rows,summary=collapse_sampling_runs(runs,tuple(range(42,47)))
    assert rows[0]['score']==pytest.approx(.8)
    assert summary['metrics']['score']==pytest.approx(.65)
    assert summary['failed_samples']==1 and summary['failed_scenes']==1 and not summary['valid']
    assert summary['zero_sample_fraction']==.1
    del runs[43]['b']
    with pytest.raises(ValueError):collapse_sampling_runs(runs,tuple(range(42,47)))


def test_paired_uncertainty_clusters_logs_after_averaging_inference_seeds():
    from tools.foresight.summarize_experiments import paired_difference,METRICS
    def row(token,log,value):
        return dict(token=token,log=log,failed_sampling_runs=0,**{m:value for m in METRICS})
    baseline=[row('a','one',.2),row('b','one',.2),row('c','two',.2)]
    first=[row('a','one',.4),row('b','one',.4),row('c','two',.1)]
    result,scenes,logs=paired_difference(first,baseline,bootstrap=1000)
    assert result['metrics']['score']['delta']==pytest.approx(.1)
    assert result['metrics']['score']['log_cluster_95_interval']==pytest.approx([-.1,.2])
    assert (result['wins'],result['losses'],result['ties'])==(2,1,0)
    assert len(scenes)==3 and len(logs)==2 and logs[0]['scenes']==2
    # Duplicating each scene within its original log must not manufacture precision.
    twice=lambda rows:rows+[dict(r,token=r['token']+'_repeat') for r in rows]
    repeated=paired_difference(twice(first),twice(baseline),bootstrap=1000)[0]
    assert repeated['metrics']['score']['log_cluster_95_interval']==result['metrics']['score']['log_cluster_95_interval']
    single=paired_difference(first[:2],baseline[:2],bootstrap=1000)[0]
    assert single['metrics']['score']['log_cluster_95_interval'] is None
