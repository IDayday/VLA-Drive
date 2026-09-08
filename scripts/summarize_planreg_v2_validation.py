"""Publish small measured evidence only; do not copy weights, datasets or feature caches."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--artifacts',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--tests',default='final_pytest.xml')
    args=parser.parse_args();root=Path(args.artifacts);output=Path(args.output)
    if output.exists():raise FileExistsError('New immutable evidence directory required')
    output.mkdir(parents=True)
    def read(path):return json.loads((root/path).read_text())
    def write(name,value):(output/name).write_text(json.dumps(value,indent=2))
    def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    runs={}
    for name in ('final_smoke32','ddp2_accum2_smoke4'):
        report=read(name+'/validation.json');meta=read(name+'/run_metadata.json')
        runs[name]={k:v for k,v in report.items() if k not in ('records','trainable_updates')}
        runs[name].update(loaded_git_commit=meta['git_commit'],world_size=meta['world_size'],global_batch=meta['global_batch'],
            changed_trainable_tensors=sum(v>0 for v in report['trainable_updates'].values()),
            trainable_tensor_count=len(report['trainable_updates']),
            loss_first={k:v for k,v in report['records'][0].items() if k.endswith('loss')},
            loss_last={k:v for k,v in report['records'][-1].items() if k.endswith('loss')},
            evidence_sha256=sha(root/name/'validation.json'))
    test=ET.parse(root/args.tests).getroot()
    suites=list(test.iter('testsuite'))
    totals={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','errors','failures','skipped')}
    shared=read('shared_pair.json');pair=read('vlm_pair.json')
    report=dict(base_commit='d9ca73f3d61f059285fcbf12a5bc81177ee350d7',
        evidence_assembly_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        tests=totals,test_report_sha256=sha(root/args.tests),runs=runs,
        resume_accumulate2=read('final_resume_accum2/resume_parity.json'),export=read('final_smoke32/export_replay.json'),
        shared_trainable_initial_state_bitwise_equal=shared['trainable_initial_state_bitwise_equal'],
        shared_optimizer_group_counts_equal=shared['group_counts_equal'],
        shared_init_sha256=shared['reports'][0]['shared_sha256'],vlm_config_differences=pair['differences'],
        original_worktree_head=subprocess.check_output(['git','-C','/mnt/project/DriveVLA-M0','rev-parse','HEAD'],text=True).strip(),
        original_worktree_status=subprocess.check_output(['git','-C','/mnt/project/DriveVLA-M0','status','--short'],text=True).splitlines(),
        full_formal_layout='NOT_RUN',full_training='NOT_RUN',navtest='NOT_RUN',performance_claim='NOT_EVALUATED',
        caveat='The 32-scene cache is a bounded functional test, not a full-data or final-layout benchmark.')
    write('VALIDATION_SUMMARY.json',report)
    write('PRECISION_CONTRACT.json',read('final_PRECISION_CONTRACT.json'))
    write('RUNTIME_PROVENANCE.json',read('final_smoke32/runtime_provenance.json'))
    write('VLM_PAIR_AUDIT.json',pair)
    write('OPTIMIZER_GROUPS.json',read('final_smoke32/run_metadata.json')['optimizer_groups'])
    write('SAME_BATCH_GRADIENT.json',read('final_resume_accum2/reference/gradient_audit_step000000.json'))
    write('LOGGED_MOTION_AUDIT.json',read('logged_motion_audit.json'))
    # Invoke the existing fixed-source executable audit, not an invented parity summary.
    from scripts.audit_drivor_scorer_parity import run_audit
    parity=run_audit(Path('/mnt/project/external/DrivoR'),seed=20260901)
    write('SCORER_PARITY.json',parity)
    if not parity['passed'] or totals['errors'] or totals['failures']:raise AssertionError('Validation evidence contains failure')
    print(json.dumps(dict(tests=totals,scorer_parity=parity['passed'],output=str(output)),indent=2))


if __name__=='__main__':main()
