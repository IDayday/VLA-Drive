"""Prospective remaining11 registration with an explicit old first-slot prefix.

CPU-only frozen coordinator build_plan reclassifies existing raw statistics; no
GPU dispatch, retry, old byte rewriting or numerical threshold change.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess

import prepare_execution_e2_envelope as pins

ROOT = Path('/mnt/project/VLA-Drive-structured-world-fgtr-perf-fullgrad-driver-v3-frozen-20261011')
GIT = 'c86285df9c9356729c2c5af96b6caf47421a1f9d'
SHA = '50160a20e75746014a7e3779532e7d15d4c01c09e2c9b3bf8c5f461577d56602'
MANIFEST_SHA = '60de82bd6e04003ccbbaeb4103d846f6def67971e45db68f7f99e6c9b89abfbc'
TARGET = '/mnt/project/structured-world-fgtr-execution-artifacts/20261011/p0_gradient_v3_zt4_actual'


def chronology_rule(jobs, prefix, frozen_UTC):
    expected = {(g,s,r) for g in ('G1','G3') for s in ('initial','mid') for r in range(3)}
    keys = [(j['case'].split('/')[0],j['case'].split('/')[1],j['repeat']) for j in jobs]
    pins.require(len(keys) == len(set(keys)) == 12 and set(keys) == expected and keys[0] == ('G1','initial',0), 'Original12 slot coverage/order changed')
    pins.require(prefix['job'] == jobs[0], 'Only actual first old slot may be exempt')
    return {'remaining_slots_before_A': [list(k) for k in keys[1:]],
            'remaining_slot_start_requirement': 'All actual A/AA/B role starts and each new PROCESS >= '+frozen_UTC,
            'prefix_exemption': {'slot':list(keys[0]), 'reason':'Already executed under its ORIGINAL pre-A enrollment and dispatcher; new freeze is NOT before this old A',
                                 'new_enrollment_applies':False, 'GPU_reexecution_allowed':False}}


def enroll(manifest_path, original_forward_enrollment, output):
    out = Path(output).resolve(); pins.require(not out.exists(), 'NEW external enrollment receipt directory required')
    pins.ref(manifest_path,MANIFEST_SHA); manifest = pins.load(manifest_path)
    original = pins.document(pins.ref(original_forward_enrollment))
    original_binding = pins.document(original['forward_binding'])
    pins.require(original['actual869'] == pins.ref(pins.GRAD_BINDING,pins.GRAD_BINDING_SHA), 'Original869 context changed')
    pins.require(manifest['recommended_actual_output'] == TARGET and not Path(TARGET).exists(), 'New shared output exists; cannot retrospectively enroll11')
    pins.require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip() == GIT
                 and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(), 'New coordinator not cold')
    script = ROOT/'tools/structured_world/run_p0_fullgrad_probes.py'; pins.ref(script,SHA)
    pins.require(manifest['fullgradient_source_binding'] == pins.ref(pins.GRAD_BINDING,pins.GRAD_BINDING_SHA), 'Producer registration changed')
    driver = pins.module(script,SHA,'_actual_remaining11_coordinator')
    plan = driver.build_plan(manifest,TARGET)
    prefix = plan['observed_prefix']
    pins.require(plan['new_GPU_jobs'] == 11 and prefix and Path(prefix['job']['folder']).is_relative_to(Path(original_binding['output_root'])), 'Exactly first original slot plus11new required')
    numeric_path = Path('/mnt/project/structured-world-fgtr-execution-artifacts/20261011/p0_gradient_v2_zt4_actual/UNCHANGED_NUMERIC_COMPONENT_REVIEW.json')
    numeric = pins.load(numeric_path)
    pins.require(numeric['numeric_thresholds_changed'] is False and numeric['passed'] is False and numeric['failures'], 'Preserved real numeric failure missing')
    now = datetime.now(timezone.utc).isoformat(); rule = chronology_rule(plan['jobs'],prefix,now)
    pins.require(not Path(TARGET).exists(), 'Output appeared during CPU enrollment')
    out.mkdir(parents=True)
    pins.write_new(out/'PLAN_CPU_ONLY.json',plan)
    value = {'schema':'execution_E2_forward_remaining11_enrollment_v1','registered_UTC':now,'kind':'forward',
        'registration':pins.ref(pins.REG,pins.REG_SHA),'criteria_changed':False,
        'coordinator':{'root':str(ROOT),'git_sha':GIT,'script':pins.ref(script,SHA)},
        'producer':pins.ref(*pins.PRODUCERS['forward']),'producer_git_sha':pins.PINS['fullgrad'][1],
        'actual_full869':pins.ref(pins.GRAD_BINDING,pins.GRAD_BINDING_SHA),
        'model_full431':pins.document(pins.ref(pins.GRAD_BINDING,pins.GRAD_BINDING_SHA))['model'],
        'manifest':pins.ref(manifest_path,MANIFEST_SHA),'plan':pins.ref(out/'PLAN_CPU_ONLY.json'),
        'new_output':TARGET,'new_output_absent_at_enrollment':True,'new_GPU_jobs_registered':11,
        'prior_firstslot_original_enrollment':pins.ref(original_forward_enrollment),'prior_firstslot_original_binding':original['forward_binding'],
        'original_dispatch_source':'ccbda534adc63e71425143c61a98af9381b31416 / 0e0d786fa7f2d380b77f24e880e02fe98311b4b95c044209f48addab0643f886',
        'actual_preserved_prefix':prefix,'chronology':rule,
        'mandatory_numeric_negative':pins.ref(numeric_path),'numeric_failures':numeric['failures'],
        'policy':'Diagnostic continuation only; original strict failure and norm_relative failure persist. No early or whole formal pass',
        'GPU_jobs_started':0,'passed':False,'formal_authorized':False,'trainer_consumable':False,
        'internal_budget':manifest['native_budget'],'budget_scope':'Original30GPUh finite phase includes reused firstslot actualcost; userGPUhlimitnull, oldhistoryUNKNOWN_NOT_ZERO',
        'new_observation_consumer_ONLY': 'c862 metadata/order verifier; original producer/model/raw statistics and old dispatcher provenance unchanged'}
    pins.write_new(out/'ENROLLMENT.json',value)
    command=[manifest['python'],'-B',str(script),'--manifest',str(Path(manifest_path).resolve()),'--output',TARGET,'--execute']
    pins.write_new(out/'HANDOFF.json',{'enrollment':pins.ref(out/'ENROLLMENT.json'),'command_for_parent_ONLY':command,
        'own_enrollment_helper':pins.ref(__file__),'source_code_stable':True,'actual_CPU_prefix_reclassification_completed':True,
        'new_GPU_dispatch_performed':False,'formal_acceptance_granted':False})
    return {'enrollment':pins.ref(out/'ENROLLMENT.json'),'new_GPU_jobs':11,'reused_old_slot':1,'GPU_jobs_started':0,'passed':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path);p.add_argument('--original-forward-enrollment',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();print(__import__('json').dumps(enroll(args.manifest,args.original_forward_enrollment,args.output)))


if __name__ == '__main__':main()
