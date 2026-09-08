"""Bind small measured artifacts to the tested snapshot; missing runs remain BLOCKED."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from navsim.agents.EpisodeDrive.planreg_v2.runtime import source_fingerprint


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--artifacts',required=True);p.add_argument('--tested-commit',required=True)
    p.add_argument('--output',default='reports/planreg_wm_v2_review');a=p.parse_args()
    root=Path(a.artifacts);output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    identity=dict(tested_code_commit=a.tested_commit,source_fingerprint=source_fingerprint(),attachments='NOT_FOUND_NOT_READ')
    def read(path):
        path=root/path
        if not path.is_file():return dict(status='BLOCKED',reason='Artifact not produced',path=str(path))
        result=json.loads(path.read_text());return dict(result,artifact_path=str(path),artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    def write(name,data):
        (output/(name+'.json')).write_text(json.dumps(dict(identity,**data),indent=2,allow_nan=False))
    for name in ('SCORER_INTEGRATION_PARITY','LONG_TARGET_PARITY','LR_RESOLUTION','INITIALIZATION_AUDIT'):
        data=read('final_audit/'+name+'.json')
        if name=='INITIALIZATION_AUDIT':data['shared_pair']=read('final_shared_pair.json')
        write(name,data)
    migration=read('final_migration.ckpt.audit.json');write('MIGRATION_COVERAGE',migration)
    tree=ET.parse(root/'logs/final_regression.xml');suite=tree.getroot().find('testsuite')
    tests=dict(tests=int(suite.attrib['tests']),failures=int(suite.attrib['failures']),errors=int(suite.attrib['errors']),skipped=int(suite.attrib['skipped']),seconds=float(suite.attrib['time']))
    unit_pass=not(tests['failures'] or tests['errors'])
    ids=[c.attrib['classname']+'::'+c.attrib['name'] for c in suite.findall('testcase') if c.find('failure') is None and c.find('error') is None and c.find('skipped') is None]
    write('REGRESSION_RESULTS',dict(status='PASS' if unit_pass else 'FAIL',counts=tests,passed_test_ids=ids,
        input_type='CPU unit, two-process Gloo and CUDA numerical regression; not all tests are full-model tests'))
    write('DDP_ACCUMULATION_PARITY',dict(status='PASS' if unit_pass and any('review_ddp' in n for n in ids) else 'BLOCKED',
        production='runtime.accumulated_batches + losses.world_model_loss',tests=[n for n in ids if 'distributed' in n or 'review_ddp' in n],
        real_ddp=read('final_ddp4/validation.json')))
    smoke=read('final_smoke32/validation.json')
    write('REAL_SMOKE_SUMMARY',dict(smoke=smoke,metadata=read('final_smoke32/run_metadata.json'),
        initial_probe=read('final_smoke32/probe_initial.json'),final_probe=read('final_smoke32/probe_final.json'),
        precision=read('final_precision.json'),status='PASS' if smoke.get('optimizer_steps')==32 else 'BLOCKED'))
    write('GRADIENT_ROUTING',dict(step0=read('final_smoke32/gradient_audit_step000000.json'),
        step1=read('final_smoke32/gradient_audit_step000001.json'),
        production='runtime.component_gradient_audit; separate L_traj, L_scorer, weighted L_WM on one batch before clipping'))
    resume=read('final_resume/resume_parity.json');export=read('final_agentinput_export.json')
    write('RESUME_EXPORT_REPORT',dict(resume=resume,real_agent_input_export=export))
    profile=read('GB128_PROFILE.json');write('GB128_PROFILE',profile)
    required_resume=('model_equal','optimizer_moments_equal','scheduler_equal','sampler_progress_equal','rng_equal')
    ready=unit_pass and smoke.get('optimizer_steps')==32 and all(resume.get(n) is True for n in required_resume) and export.get('status')=='PASS'
    ready=ready and read('final_ddp4/validation.json').get('optimizer_steps')==4
    write('FINAL_STATUS',dict(ENGINEERING_READY=ready,READY_FOR_FORMAL_TRAINING=False,PERFORMANCE='NOT_EVALUATED',
        formal_blockers=['Complete 103288-scene progressive-long cache and matching raw-GT statistics not built in this bounded task',
                         'No uncontended real GB128 layout lock; all local GPUs were occupied by existing candidate export jobs'],
        tests=tests,not_a_performance_claim=True))
    # Preserve small red/green evidence and final logs, not checkpoints/caches.
    for name in ('red.log','red.xml','green_initial.log','final_regression.log','final_regression.xml','final_scorer_component.log'):
        source=root/'logs'/name
        if source.is_file():
            destination=output/'logs'/name;destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(source.read_bytes())


if __name__=='__main__':main()
