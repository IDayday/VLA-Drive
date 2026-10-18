"""Small T3 v2 binding review. Never runs preparation, storage ledger or GPU."""
import hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
AUTHOR=Path('/mnt/project/VLA-Drive-st-teacher-channel-control-20261017')
P=AUTHOR/'reports/st_world_policy/20261018/temporal_T3_cpu_preparation_v2'
O=AUTHOR/'reports/st_world_policy/20261017/temporal_T3_cpu_preparation_v1'
def pin(p):
    assert p.stat().st_size<4<<20
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def readpin(p):
    assert pin(Path(p['path']))==p
    return json.loads(Path(p['path']).read_text())
def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=AUTHOR,text=True).strip()=='075972d4188ce3b6fc4afbae631eb3fcef349e4b'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=AUTHOR).strip()
    expected={'PLAN.json':'0fd1b5d93e02008f97959fb29b6f37eedd2eae898cb97468ceca8ad41ab8f631',
      'prepare_once.py':'80da95a21f7918b0524d1d71045da13d43c1828747f046e62d9bdb26e418bd01',
      'ADMISSION_PROPOSAL.json':'86962148e0d564ce1aea8d59d75ca3e3be60fc1593321a374d66d0329f0c8be0'}
    for n,h in expected.items():assert pin(P/n)['sha256']==h
    p=json.loads((P/'PLAN.json').read_text());a=json.loads((P/'ADMISSION_PROPOSAL.json').read_text());old=json.loads((O/'PLAN.json').read_text())
    assert p['inputs']==old['inputs']
    for v in p['inputs'].values():readpin(v)
    for v in p['source_files'].values():assert pin(Path(v['path']))==v
    for v in a['controller_sources']:assert pin(Path(v['path']))==v
    assert a['plan']==pin(P/'PLAN.json') and a['control_reservation']==p['control_reservation']
    assert a['status']=='PROPOSAL_NOT_APPROVED' and a['reviewer'] is None
    assert a['issued_at_unix'] is a['latest_start_unix'] is a['valid_until_unix'] is None
    version=p['versioned_attempt'];failure=readpin(version['previous_failed_result'])
    assert failure['status']=='FAILED_NO_AUTO_RETRY' and failure['resources']['cleanup']['live_owned_remaining']==0
    assert Path(version['previous_failed_root']).is_dir() and not Path(p['output_root']).exists()
    assert p['output_root']!=old['output_root'] and p['training_output_root']!=old['training_output_root']
    compat=readpin(version['mathematics_compatibility'])
    baseline=HERE.parent/'T3_cpu_preparation_review/native_migration_patch'
    review=readpin({'path':str(baseline/'REVIEW.json'),'sha256':'4a8292bc0b023ef06492ac49658bdb665e3b746d07c84c360ff2a771e9d6ce12'})
    migration=json.loads((baseline/'SOURCE_MIGRATION_PREFLIGHT.json').read_text())
    assert compat==migration['mathematics_compatibility']
    assert review['reviewed_commit']==p['runtime']['source_sha']=='cf516259c8efc24503727f14b9e330ed32231293'
    r=p['control_reservation']
    assert r['prior_reserved_bytes']==old['control_reservation']['total_reserved_bytes']==7880<<20
    assert r['total_reserved_bytes']==r['prior_reserved_bytes']+r['control_reservation_bytes']==7888<<20
    assert r['retained_failed_v1_bytes']==r['control_reservation_bytes']==8<<20
    assert r['remaining_bytes']==r['metadata_envelope_bytes']-r['total_reserved_bytes']==304<<20
    for k in ('prior_evidence','history_control_adoption','retained_failed_v1_plan'):readpin(r[k])
    report={'status':'PASS','reviewer':'/root/temporal_narrow_review',
      'reviewed_binding_commit':'075972d4188ce3b6fc4afbae631eb3fcef349e4b',
      'binding':pin(P/'prepare_once.py'),'plan':pin(P/'PLAN.json'),'unsigned_proposal':pin(P/'ADMISSION_PROPOSAL.json'),
      'inherited_native_review':pin(baseline/'REVIEW.json'),
      'narrow_diff':'Wrapper exactly four runtime/output/commit/PLAN replacements; launch shell byte identical. No new controller semantics.',
      'actual_metadata_inputs_unchanged':True,'all_current_source_input_pins_verified':True,
      'failed_v1_result':version['previous_failed_result'],'failed_v1_evidence_retained':True,
      'reservations_MiB':{'prior_including_failed_v1':7880,'new_v2':8,'total':7888,'envelope':8192,'remaining':304},
      'focused_tests':{'count':3,'failures':0,'errors':0,'log':pin(HERE/'TEST.log')},
      'runtime':p['runtime'],'limits':p['limits'],'max_output_bytes':p['max_output_bytes'],
      'actual_admission_issued':False,'actual_preparation_started':False,'GPU_or_SSH':False,'ledger_read_or_write':False,
      'limits_of_review':['Only CPU metadata preparation is covered. Future T3 PROFILE4 config is not training authorization.',
        'Native CLI proof and negative compatibility tests are inherited from real cf516259 independent preflight; not repeated in this binding review.',
        'Current host/time admission and systems resource accounting remain required before Lead single invocation.',
        'Prior v1 failure root and full8MiB reservation retained; no delete/retry under that identity.']}
    target=HERE/'REVIEW.json';assert not target.exists();target.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'PASS','report':pin(target)}))
if __name__=='__main__':main()
