"""Independent small metadata/source audit; never runs either preparation CLI."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
AUTHOR=Path('/mnt/project/VLA-Drive-st-teacher-channel-control-20261017')
P=AUTHOR/'reports/st_world_policy/20261017/temporal_T3_cpu_preparation_v1'
EXPECTED={
 'prepare_once.py':'72129bd4b20bc0dcff61b623fb28c368142fc25efbe3be0d65aa2664567b3e0a',
 'PLAN.json':'f699cc48d009791349a65788c9a07c057364fd2edd1e3c085233b7151b91d72d',
 'ADMISSION_PROPOSAL.json':'99c50b3de157ed22f97c22a8653b04f89f0452d3aac9d046c10676d4afbc5bd4'}
seen={}
def raw(path):
    p=Path(path)
    assert p.is_absolute() and p.stat().st_size <= 4<<20
    b=p.read_bytes(); seen[str(p)]=len(b)
    assert sum(seen.values()) < 128<<20
    return b
def pinned(pin):
    b=raw(pin['path']); assert hashlib.sha256(b).hexdigest()==pin['sha256'],pin
    return b
def jsonpin(pin):return json.loads(pinned(pin))
def pin(path):return {'path':str(path),'sha256':hashlib.sha256(raw(path)).hexdigest()}
def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=AUTHOR,text=True).strip()=='658a96108a80187ae332781903feea7801dbf569'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=AUTHOR).strip()
    for name,sha in EXPECTED.items():pinned({'path':str(P/name),'sha256':sha})
    plan=json.loads(raw(P/'PLAN.json')); proposal=json.loads(raw(P/'ADMISSION_PROPOSAL.json'))
    assert proposal['status']=='PROPOSAL_NOT_APPROVED' and proposal['reviewer'] is None
    assert proposal['issued_at_unix'] is proposal['latest_start_unix'] is proposal['valid_until_unix'] is None
    assert proposal['plan']==pin(P/'PLAN.json')
    for v in plan['source_files'].values():pinned(v)
    inputs={k:jsonpin(v) for k,v in plan['inputs'].items()}
    for v in proposal['controller_sources']:pinned(v)
    runtime=Path(plan['runtime']['path'])
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=runtime,text=True).strip()==plan['runtime']['source_sha']=='676e3e88d48d9444ab2cb928242d196381631b26'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=runtime).strip()
    assert plan['limits']=={'wall_seconds':300,'cleanup_seconds':40,'read_bytes':4<<30,'rss_bytes':2<<30}
    assert plan['max_output_bytes']==8<<20 and plan['cpu_affinity']==[2,3]
    assert plan['gpu'] is plan['T4_automatic'] is plan['training_started'] is False
    h,m,s=inputs['history_complete'],inputs['history_manifest'],inputs['selected']
    assert h['status']=='COMPLETE' and h['scenes']==32 and h['logs']==2
    assert h['history_manifest_sha256']==plan['inputs']['history_manifest']['sha256']
    assert h['history_identity']==m['identity']
    assert [(r['token'],r['log']) for r in m['scenes']]==[(r['token'],r['log']) for r in s['scenes']]
    assert len(s['scenes'])==32 and s['split']==m['split']=='train'
    assert len({r['token'] for r in s['scenes']})==32
    assert not Path(plan['output_root']).exists()
    r=plan['control_reservation']
    assert r['proposed_total_reserved_bytes']==r['conservative_prior_bytes']+r['retained_history_artifact_and_control_bytes']+r['total_bytes']
    assert r['total_bytes']==8<<20 and r['terminal_control_bytes_included']==1<<20
    assert r['remaining_bytes']==r['metadata_envelope_bytes']-r['proposed_total_reserved_bytes']>=0
    pinned(r['prior_evidence']);pinned(r['history_control_adoption'])
    result={'status':'PASS','reviewer':'/root/temporal_narrow_review',
      'reviewed_at_utc':datetime.now(timezone.utc).isoformat(),
      'reviewed_commit':'658a96108a80187ae332781903feea7801dbf569',
      'controller':pin(P/'prepare_once.py'),'plan':pin(P/'PLAN.json'),'unsigned_proposal':pin(P/'ADMISSION_PROPOSAL.json'),
      'source_inputs_pins_verified':len(plan['source_files'])+len(plan['inputs']),
      'history32_actual':'COMPLETE,32 unique scenes/2logs; exact ordered scene/log list matches original indexed32; metadata hashes match.',
      'scope':'Only T3 exact32 CPU metadata preparation; no GPU/training/teacher generation; future four-update config is not launch authorization.',
      'existing_entry_reuse':['qualify_indexed_provider prepare-profile-migration','temporal_composition --cell T3'],
      'resource_bounds':plan['limits'],'output_bytes':plan['max_output_bytes'],
      'resource_arithmetic':r,
      'read_cost':'Static owner/system estimate1,401,661,260 primary hash bytes plus1,590,864 PCA statistics; actual aggregate4GiB ceiling monitored. No measured preparation cost claimed.',
      'corrected_findings':[
        {'id':'T3_F1','problem':'Initial COMMIT/PLAN a8f8cb differed from actual already-reviewed676e3e88; original strict parent guard would reject before output creation.','fix':'Bind actual clean676e3e88; unchanged pinned numerical/source files; no rollback.'},
        {'id':'T3_F2','problem':'Per-child inventory restarted device baseline, weakening original mount-escape guard.','fix':'Every direct child device checked against root before original strict recursive inventory; synthetic mismatch test passes.'}],
      'focused_regression':{'tests':9,'failures':0,'errors':0,'log':pin(HERE/'FOCUSED_TESTS.log'),'kind':'Author narrow fixtures independently rerun, not real preparation or model qualification'},
      'inherited_not_rerun':['43 temporal +8 history wrapper +4 independent refusals','history32 current W source compatibility review','original9ef4b two-CLI supervisor/4b2f monitor and cleanup'],
      'new_boundary_review':['Only local configured -> .configured.staging-* link allowed; canonical stage outputs pinned; all other symlinks/shared hardlinks/mount escape reject; failed staging remains charged.',
        'Two commands share original deadline, lock, aggregate logical-read/RSS/owned-descendant cleanup, single-shot refusal and bounded terminal receipt. First failure prevents second CLI.',
        'CLI --max-seconds1800 defines future diagnostic training config; preparation remains300+40 seconds.'],
      'unique_small_metadata_source_bytes':sum(seen.values()),
      'admission_issued':False,'actual_preparation_run':False,'gpu_or_ssh_access':False,'ledger_read_or_write':False,
      'remaining_conditions':['Systems must confirm actual8MiB administrative reservation and current disk/CPU availability.',
        'Lead/independent actual host/time admission still required; unsigned proposal intentionally rejected.',
        'Actual CPU preparation may fail safely at resource cap; no preparation COMPLETE or future training approval is inferred.']}
    target=HERE/'REVIEW.json';assert not target.exists();target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'PASS','review':pin(target),'small_read_bytes':sum(seen.values()),'admission_issued':False}))
if __name__=='__main__':main()
