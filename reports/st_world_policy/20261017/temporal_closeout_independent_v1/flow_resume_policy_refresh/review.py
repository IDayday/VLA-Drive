"""Narrow, read-only policy-refresh audit; no launcher import or dispatch."""
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import subprocess
import time

OUT = Path(__file__).resolve().parent
ROOT = Path('/mnt/project/st-world-policy-20261016')
RUNTIME = Path('/mnt/project/VLA-Drive-st-flow-context-runtime-v3-20261017')
LAUNCHER = Path('/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261018/flow_resume_policy_refresh_v1/launch.py')
SNAPSHOT = ROOT/'resources/policy_refresh_20261018T0014/DISPATCH_POLICY_SNAPSHOT.json'
OLD = ROOT/'resources/launch_reviewed_flow_v1.py'
POLICY_SHA = '29ce8b89aec0e20d0b429d6a026634ec59e6ddee5933106d2ce9e378fa322d55'
NEW_SHA = 'c15714b2f0dba80024de94bb4b878fc31e4a231c9b83094b084163b1709b4bd2'
OLD_SHA = '6a03442b26eb01a5de9e2ea5e3e1f508ca69d2959ba18cf37903e73295ce3db2'
HASHES = {}

def sha(path):
    path = Path(path)
    if str(path) not in HASHES:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda: f.read(1 << 20), b''):
                h.update(b)
        HASHES[str(path)] = h.hexdigest()
    return HASHES[str(path)]

def pin(path):
    return {'path': str(path), 'sha256': sha(path)}

def read(path):
    return json.loads(Path(path).read_text())

def verify(p):
    assert sha(p['path']) == p['sha256'], p['path']

def reject(code, env):
    try:
        exec(compile(ast.Module(body=[code], type_ignores=[]), '<actual-guard>', 'exec'), env)
    except AssertionError:
        return
    raise AssertionError('Expected actual launcher guard rejection')

def main():
    start = time.monotonic()
    assert sha(OLD) == OLD_SHA and sha(LAUNCHER) == NEW_SHA
    assert sha(SNAPSHOT) == sha('/mnt/project/server_dispatch_policy.json') == POLICY_SHA
    old, new = OLD.read_text(), LAUNCHER.read_text()
    # Whole-source transform, not a selected-line claim. Everything outside the
    # explicit approval replacement and receipt-schema rename must be identical.
    expected = old.replace("p.add_argument('--registration', required=True)\n", "p.add_argument('--registration', required=True)\np.add_argument('--approval', required=True)\n")
    begin, end = expected.index('reviews = {}\n'), expected.index('host = socket.gethostname()\n')
    replacement = new[new.index('# Resume only.'):new.index('host = socket.gethostname()\n')]
    expected = expected[:begin] + replacement + expected[end:]
    expected = expected.replace('atomic_json(approval, proposal, immutable=True)\n', '')
    expected = expected.replace("'schema':'st_reviewed_formal_flow_launch_v1'", "'schema':'st_reviewed_formal_flow_policy_refresh_launch_v1'")
    assert expected == new
    tree = ast.parse(new)
    asserts = [n for n in tree.body if isinstance(n, ast.Assert)]
    resume = next(n for n in asserts if ast.unparse(n.test) == 'a.resume')
    boundary = next(n for n in asserts if ast.unparse(n.test).startswith('approval.is_relative_to('))
    from types import SimpleNamespace
    reject(resume, {'a': SimpleNamespace(resume=False)})
    reg = ROOT/'formal/consequence/registrations/C1_fulltrain_seed42_official_4096_v1/REGISTRATION.json'
    for bad in (reg.parent/'RUN_APPROVAL.json', reg.parent.parent/'C4_fulltrain_seed42_official_4096_v1/policy_refresh/APPROVAL.json'):
        reject(boundary, {'regpath': reg, 'approval': bad})
    exec(compile(ast.Module(body=[boundary], type_ignores=[]), '<actual-guard>', 'exec'), {'regpath': reg, 'approval': reg.parent/'policy_refresh/v1/APPROVAL.json'})
    policy = read(SNAPSHOT)
    host = 'training-vlawm-tsinghua-zt4-worker-0'
    assert host in policy['allowed_aliases']
    assert host.removesuffix('-worker-0') not in policy['global_dispatch_prohibitions']
    assert policy['idle_pressure_holds'][host.removesuffix('-worker-0')]['status'] == 'disabled_by_user_resource_release'
    assert policy['idle_pressure_holds'][host.removesuffix('-worker-0')]['scope'] == 'pressure only; preserve genuine training/evaluation'
    host_guard = next(n for n in asserts if ast.unparse(n.test).startswith("host in policy['allowed_aliases']"))
    auth = read(ROOT/'resources/AUTHORIZATION_20261016T203442.json')
    reject(host_guard, {'host':'training-vla-zt3-worker-0', 'policy':policy, 'auth':auth})
    exec(compile(ast.Module(body=[host_guard], type_ignores=[]), '<actual-guard>', 'exec'), {'host':host, 'policy':policy, 'auth':auth})
    spec = importlib.util.spec_from_file_location('reviewed_gate', RUNTIME/'tools/st_world_policy/check_formal_approval.py')
    gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=RUNTIME,text=True).strip() == '6714b1e9fb2e0c8059b90b8b4d48cd40796fe345'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=RUNTIME).strip()
    source_fp = gate.source_fingerprint(RUNTIME)
    reports = []
    for arm in ('C1', 'C4'):
        d = ROOT/f'formal/consequence/registrations/{arm}_fulltrain_seed42_official_4096_v1'
        regpath = d/'REGISTRATION.json'; reg = read(regpath)
        original = read(d/'reviews/CORRECTNESS.json')
        approval = read(d/'RUN_APPROVAL.json')
        assert original['status'] == 'PASS' and original['role'] == 'correctness'
        assert original['fingerprint'] == approval['fingerprint']
        for p in reg['artifacts'].values(): verify(p)
        fingerprint = {'registration_sha256':sha(regpath),'source_fingerprint':source_fp,
            'artifacts':{k:v['sha256'] for k,v in reg['artifacts'].items()},
            'execution':reg['execution'],'run_id':reg['run_id']}
        assert fingerprint == original['fingerprint']
        assert reg['source_fingerprint'] == source_fp
        assert reg['execution']['max_updates'] == reg['execution']['stop_after'] == 4096
        assert reg['execution']['world_size'] == 4 and reg['execution']['global_batch'] == 32
        cfg = read(reg['artifacts']['config']['path']); assert cfg['execution'] == reg['execution']
        lease = read(reg['artifacts']['resource_lease']['path'])
        assert lease['hostnames'] == [host]
        assert lease['valid_from_unix'] <= time.time() < lease['valid_until_unix'] - 28800 - 780
        assert lease['cuda_visible_devices_by_host'][host] == ('4,5,6,7' if arm == 'C1' else '0,1,2,3')
        technical = [p for p in original['evidence_artifacts'] if p['path'] != '/mnt/project/server_dispatch_policy.json']
        for p in technical: verify(p)
        for role,p in approval['reviews'].items():
            verify(p)
            assert read(p['path'])['fingerprint'] == fingerprint
        result = {'schema':'st_world_independent_role_review_v1','status':'PASS','role':'correctness',
            'reviewer_agent':'/root/temporal_narrow_review',
            'reviewed_at_utc':datetime.now(timezone.utc).isoformat(), 'fingerprint':deepcopy(fingerprint),
            'decision_scope':f'{arm} same-lineage explicit native resume only to original absolute4096. Policy snapshot refresh and thin launcher only; no new model or training definition.',
            'evidence_artifacts':[pin(d/'reviews/CORRECTNESS.json'),pin(d/'RUN_APPROVAL.json'),pin(LAUNCHER),pin(SNAPSHOT),pin(OLD),pin(__file__)]+technical,
            'policy_refresh':{'current_snapshot':pin(SNAPSHOT),'old_policy_sha256':'79cca8cfdb98ed7f46af554633c4734e61def4ee02ff656220415013b2082097',
                'old_policy_snapshot_available':False,'full_policy_diff_performed':False,
                'current_authorization_verified':True,'pressure_only_disabled':True,'scientific_fingerprint_unchanged':True},
            'launcher_mechanical_diff':'Only required --approval, resume-only and same-registration policy_refresh boundary, replacement of auto approval construction by original strict gate, and launch receipt schema rename.',
            'negative_checks':['no --resume rejected','old root RUN_APPROVAL path rejected','other-run approval path rejected','withdrawn host rejected'],
            'limits':deepcopy(original['limits'])+[
                'Original correctness evidence is inherited except its stale mutable dispatch-policy reference, replaced by this immutable current snapshot. Old report/approval unchanged.',
                'Fresh host policy, GPU UUID/idle locks, lease28800+780, six invocations, fifteen native directories, same native identity and disk reservations are still checked by launcher. No live GPU/resource claim in this read-only review.',
                'Lead still must assemble approval with distinct evidence/efficiency roles. This report does not launch a job, modify ledger, claim endpoint completion or grant C0/C2/C3 expansion.']}
        target = OUT/f'{arm}_CORRECTNESS_POLICY_REFRESH.json'
        assert not target.exists()
        target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        reports.append(pin(target))
    summary={'status':'PASS','reviewer':'/root/temporal_narrow_review','reports':reports,
        'whole_source_mechanical_transform_equal':True,'four_negative_guard_tests_passed':True,
        'allowed_boundary_and_current_host_tests_passed':True,
        'unique_hashed_bytes':sum(Path(p).stat().st_size for p in HASHES),
        'elapsed_seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'jobs_started':False,'gpu_access':False,'ssh':False,'ledger_read_or_write':False,
        'training_manifest':'Only streaming content hash, no scene/image/context payload interpretation.'}
    (OUT/'RESULT.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__': main()
