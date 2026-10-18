"""Exact current v2 admission, reusing independently checked signing helpers."""
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util,json,os,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
helper=HERE.parent/'T3_cpu_preparation_review/sign_actual.py'
spec=importlib.util.spec_from_file_location('reviewed_t3_sign_helpers',helper)
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
h.HERE=HERE
h.AUTHOR=Path('/mnt/project/VLA-Drive-st-teacher-channel-control-20261017/reports/st_world_policy/20261018/temporal_T3_cpu_preparation_v2')
h.PROPOSAL=Path('/mnt/project/st-world-policy-20261016/resources/T3_cpu_preparation_v2_actual_20261018T0035/ADMISSION_CURRENT_PROPOSAL.json')
h.PROPOSAL_SHA='d02545a2b9139ae65e1a53b1ab7e7265010fbe8ecea723527e3da46315126721'
def main():
    a=h.proposal()
    prior=h.readpin({'path':str(h.AUTHOR/'ADMISSION_PROPOSAL.json'),'sha256':'86962148e0d564ce1aea8d59d75ca3e3be60fc1593321a374d66d0329f0c8be0'})
    restored=deepcopy(a)
    for k in ('issued_at_unix','latest_start_unix','valid_until_unix'):restored[k]=prior[k]
    assert restored==prior
    review=h.readpin({'path':str(HERE/'REVIEW.json'),'sha256':'a926e95328e75061fe1fed73e74e83e662bdb3579b9d33666916a9ba76c4c40f'})
    assert review['status']=='PASS' and review['actual_admission_issued'] is False
    p=h.verify_all(a)
    h.readpin(p['control_reservation']['retained_failed_v1_plan'])
    failure=h.readpin(p['versioned_attempt']['previous_failed_result'])
    assert failure['status']=='FAILED_NO_AUTO_RETRY' and failure['resources']['cleanup']['live_owned_remaining']==0
    r=p['control_reservation']
    assert r['prior_reserved_bytes']==7880<<20 and r['total_reserved_bytes']==7888<<20
    assert r['retained_failed_v1_bytes']==r['total_bytes']==8<<20
    cp=h.readpin({'path':str(h.CP),'sha256':h.CP_SHA})
    assert cp['status']=='CURRENT' and cp['remaining_bytes']==2703<<30
    assert time.time()<cp['valid_until_unix'] and a['valid_until_unix']<=cp['valid_until_unix']
    stat=os.statvfs('/mnt/project');free=stat.f_bavail*stat.f_frsize
    required=(2703+1024+768+8)<<30  # Retain entire metadata envelope as well.
    assert free>=required
    signed=deepcopy(a);signed['status']=h.STATUS;signed['reviewer']=h.REVIEWER
    h.assert_signed_matches(signed,h.proposal())
    bad=deepcopy(signed);bad['controller_sources'][0]['sha256']='0'*64
    try:h.assert_signed_matches(bad,h.proposal())
    except AssertionError:nested_negative=True
    else:raise AssertionError('Mutated nested controller pin accepted')
    assert a==h.proposal()
    s=importlib.util.spec_from_file_location('actual_v2_readonly',h.AUTHOR/'prepare_once.py')
    w=importlib.util.module_from_spec(s);s.loader.exec_module(w);c=w.controller()
    class Reads:
        def raw(self,path,expected):
            assert h.sha(path)==expected;return h.raw(path)
        def document(self,pin):return h.readpin(pin)
    c.validate(signed,p,None,Reads(),time.time())
    h.verify_all(signed);h.assert_signed_matches(signed,h.proposal())
    target=HERE/'ACTUAL_ADMISSION.json';h.atomic_new(target,signed)
    emitted=json.loads(h.raw(target))
    h.assert_signed_matches(emitted,h.proposal());h.verify_all(emitted)
    result={'status':'PASS','reviewer':h.REVIEWER,'signed_admission':h.pin(target),
      'source_proposal':h.pin(h.PROPOSAL),'created_at_utc':datetime.now(timezone.utc).isoformat(),
      'review':h.pin(HERE/'REVIEW.json'),'checkpoint_commitments':h.pin(h.CP),
      'validated_to_signed_changes':['status','reviewer'],'post_write_fresh_disk_comparison_and_pins':'PASS',
      'nested_mutation_negative_test':nested_negative,'actual_bound_validate':'PASS',
      'prior_v1_failure_retained':p['versioned_attempt']['previous_failed_result'],
      'administrative_reservation_MiB':{'prior_with_v1':7880,'new_v2':8,'total':7888,'limit':8192},
      'shared_free_bytes_observed':free,'separate_retained_reserves_including_whole_metadata_bytes':required,
      'scope':'One CPU-only exact32 T3 metadata preparation v2; Lead executes. No training/GPU admission.',
      'actual_preparation_started':False,'ledger_read_or_write':False,'GPU_or_SSH':False}
    h.atomic_new(HERE/'ACTUAL_ADMISSION_REVIEW.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
