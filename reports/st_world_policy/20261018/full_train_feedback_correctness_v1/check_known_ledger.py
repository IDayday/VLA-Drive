"""One-read independent integration check; explicitly excludes other ledgers."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import tempfile
import time
from unittest.mock import patch

resource.setrlimit(resource.RLIMIT_CPU,(110,110))
START=time.monotonic()
HERE=Path(__file__).resolve().parent
KEY='a53a189d6603e316f5eaf2254fe85d3903e82ec27dce9167b143e6eddfff33c0'
SHA='8019dd29b17efa79c1fd1a7211bdf09a0b5be93931db75cd4db95ad9a3fb122f'


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--plan',required=True)
    args=p.parse_args()
    source=Path(args.source_root)/'tools/st_world_policy/analyze_full_feedback_support.py'
    raw=source.read_bytes()
    assert raw==subprocess.check_output(['git','-C',args.source_root,'show',args.source_commit+':tools/st_world_policy/analyze_full_feedback_support.py'])
    source_sha=hashlib.sha256(raw).hexdigest()
    spec=importlib.util.spec_from_file_location('review_full_feedback_support',source)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    plan_raw=Path(args.plan).read_bytes();plan=m.validate_plan(plan_raw)
    assert plan['run']==str(m.RUN)
    item=next(i for shard in plan['shards'] for i in shard['items'] if i['ledger']==KEY)
    assert item['payload_bytes']==3441888 and item['payload_sha256_declared']==SHA
    output=HERE/'KNOWN_LEDGER_REPLAY.json'
    assert not output.exists(),'one authorized real-ledger read already consumed'
    old_affinity=os.sched_getaffinity(0)
    m.configure_cpu(min(old_affinity))
    payload_path=m.RUN/'official_feedback'/KEY/'attempt_00/RESULTS.pt'
    counts=[];real_open=Path.open
    def guard(path,*a,**kw):
        if path.suffix=='.pt':
            assert path==payload_path,'no other real payload may be opened'
            counts.append(str(path));assert len(counts)==1,'only one read is authorized'
        return real_open(path,*a,**kw)
    try:
        with tempfile.TemporaryDirectory(prefix='independent_known_ledger_') as tmp:
            b=m.Budget(tmp,plan['plan_id'],'independent_known_ledger',time.time()+105,8<<20,1<<20,False)
            with patch.object(Path,'open',guard):
                result=m.verify_and_analyze(plan,item,b)
            legacy_path=Path('/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261018/consequence_next_hypothesis_v1/RESULTS.json')
            legacy_raw=legacy_path.read_bytes()
            assert hashlib.sha256(legacy_raw).hexdigest()=='f1e54540107e6f76d2b9ccd3d5eb9c4d53207b1dae8e49e1a3f245557bbbe986'
            legacy=json.loads(legacy_raw)
            assert result['reward_value_decomposition']==legacy['advantage_value_decomposition']
            assert result['centered_A_float32']==[x['C1_advantages_float32'] for x in legacy['per_scene']]
            assert result['scene_log_pairs']==[[x['scene_id'],x['source_log']] for x in legacy['per_scene']]
            assert result['EP_only_scene_presentations']==16 and result['event_candidate_counts_D_C_T']==[0,0,0]
            assert result['candidate_presentations']==128 and result['scene_presentations']==32
            for condition,row in result['terminal_ties'].items():
                assert {k:v for k,v in row.items() if k!='empty_event_pairs'}==legacy['terminal_ties'][condition]
                assert row['empty_event_pairs']==row['eligible_pairs']-row['nonempty_event_pairs']
            pooled=m.amplitude_statistics([result])
            A=m.np.array(result['centered_A_float32'],dtype=m.np.float32)
            ranges=m.np.ptp(A,axis=1)
            assert ranges.tolist()==result['within_group_A_range_float32']
            populations={'all_scene_presentations':m.np.ones(32,dtype=bool),
                         'nonzero_advantage_scene_presentations':m.np.any(A!=0,axis=1)}
            for label,selected in populations.items():
                for key,values in [('abs_A_per_candidate',m.np.abs(A[selected]).astype('float64').flatten()),
                                   ('within_group_A_range',ranges[selected].astype('float64'))]:
                    stat=pooled[label][key]
                    assert stat['count']==len(values)
                    assert abs(stat['mean']-float(values.mean()))<1e-16
                    assert abs(stat['RMS']-float(m.np.sqrt(m.np.mean(values**2))))<1e-16
                    quantiles=m.np.quantile(values,m.QUANTILES,method='linear')
                    assert m.np.allclose(quantiles,list(stat['quantiles'].values()),atol=1e-16,rtol=0)
            assert len(counts)==1 and not m.torch.cuda.is_initialized()
            usage=resource.getrusage(resource.RUSAGE_SELF)
            assert usage.ru_maxrss*1024<=2<<30
            out={'status':'PASS_ONE_AUTHORIZED_REAL_LEDGER_ONLY','reviewer':'/root/road_proxy_correctness_review',
                 'source_commit':args.source_commit,'source_sha256':source_sha,
                 'plan_sha256':hashlib.sha256(plan_raw).hexdigest(),'plan_id':plan['plan_id'],
                 'ledger':KEY,'payload_sha256':SHA,'payload_reads':1,'payload_bytes':3441888,
                 'legacy_matches_exact':True,'pooled_quantiles_match_numpy_linear':True,
                 'population':{'scenes':32,'candidates':128,'C1_active_scenes':16,'EP_only_scenes':16,'D_C_T_events':[0,0,0]},
                 'pooled_amplitude':pooled,'input_pins':result['input_pins'],'budget':dict(b.state),
                 'other_metadata_bytes':len(raw)+len(plan_raw)+len(legacy_raw),
                 'total_actual_asset_and_metadata_read_bytes':b.state['actual_read_bytes']+len(raw)+len(plan_raw)+len(legacy_raw),
                 'wall_seconds':time.monotonic()-START,'cpu_seconds':usage.ru_utime+usage.ru_stime,
                 'peak_rss_bytes':usage.ru_maxrss*1024,'full_ledger_scan':False,
                 'model_loads':0,'new_score_calls':0,'optimizer_updates':0,'DEV_Navtest_reads':0,
                 'formal_training_approved':False}
            assert out['total_actual_asset_and_metadata_read_bytes']<=8<<20
            output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    finally:
        os.sched_setaffinity(0,old_affinity)
    print(json.dumps({k:out[k] for k in ('status','payload_reads','total_actual_asset_and_metadata_read_bytes','cpu_seconds','peak_rss_bytes')}))


if __name__=='__main__':main()
