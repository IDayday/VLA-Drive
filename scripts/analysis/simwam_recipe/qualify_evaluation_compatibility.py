"""Qualify only the already-reviewed accounting edit, without new inference."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch

from scripts.analysis.simwam_recipe.evaluation_compatibility import match_protocol
from starVLA.rl.simwam_recipe.contracts import file_sha, fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json

BASE = '12b8e0d75691d7bed04ab02072020e130f711a3b'
OLD = 'f9bfa2d205bd9548beca787730c8cf7508f2412998e72f7665848e062b218163'
NEW = 'e4e8536ec2d9aa7185309d934a2747c479d6e3678fc2113339473802a9c34fe8'
EVALUATOR = 'scripts/analysis/simwam_recipe/evaluate_policy.py'


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    repo = Path.cwd()
    out = Path(a.output).resolve()
    out.relative_to(repo)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'compatibility.json').exists():
        raise FileExistsError('Keep the original qualification receipt')
    old = subprocess.check_output(['git', 'show', BASE + ':' + EVALUATOR])
    new = (repo / EVALUATOR).read_bytes()
    if hashlib.sha256(old).hexdigest() != OLD or hashlib.sha256(new).hexdigest() != NEW:
        raise ValueError('This qualification covers only the reviewed accounting patch')
    old_tree, new_tree = ast.parse(old), ast.parse(new)
    def definition(tree, name):
        return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    def calls(tree):
        selected = {'VLAAgent','KeyedDataset','RewardService','configure_numerics',
                    'prepare_policy_observation','seed_for','torch.randn',
                    'model.encode_policy_condition','model.action_model._euler_sample',
                    'model.predict_action_infer_1d','shifted_ode','decode_request','service.score',
                    'WanContinuousFlowMatchScheduler','scheduler.build_inference_schedule',
                    'head.predict_velocity','scheduler.step'}
        return [ast.dump(n, include_attributes=False) for n in ast.walk(tree)
                if isinstance(n, ast.Call) and ast.unparse(n.func) in selected]
    assert ast.dump(definition(old_tree,'shifted_ode')) == ast.dump(definition(new_tree,'shifted_ode'))
    # ast.walk order is unchanged for these exact scientific calls; logging
    # statements do not become part of the numerical graph or random stream.
    assert calls(old_tree) == calls(new_tree)
    paths = ['infer.py','starVLA/dataloader/navsim_dataset.py',
             'starVLA/rl/flow_grpo/observation.py','starVLA/rl/flow_grpo/data.py',
             'starVLA/rl/flow_grpo/reproducibility.py',
             'starVLA/rl/safe_flow_improvement/policy_adapter.py',
             'starVLA/rl/simwam_recipe/reward_readouts.py',
             'starVLA/rl/simwam_recipe/vendor/scheduler_continuous.py']
    paths += subprocess.check_output(['git','ls-files','starVLA/model'], text=True).splitlines()
    hashes = {}
    for name in sorted(set(paths)):
        if not name.endswith('.py'):
            continue
        baseline = subprocess.check_output(['git','show',BASE+':'+name])
        if (repo/name).read_bytes() != baseline:
            raise ValueError('Numerical dependency changed: '+name)
        hashes[name] = file_sha(repo/name)
    evidence_root = Path('/mnt/project/simwam-audit-recovery-20261015/qualification')
    prior = evidence_root/'evaluation_ledger_acceptance.json'
    prior_record = json.loads(prior.read_text())
    assert prior_record['status']=='PASS' and prior_record['source_sha256']==NEW
    sample_root = evidence_root/'evaluation_ledger_v1'
    sample_identity = json.loads((sample_root/'manifest.json').read_text())
    ref_root = Path('/mnt/project/simwam-recipe-20261015/evaluations')/('SFT_update000000_'+sample_identity['cohort'])
    reference = json.loads((ref_root/'manifest.json').read_text())
    assert sample_identity['source_sha256']==NEW and reference['source_sha256']==OLD
    samples = list((sample_root/'scenes').glob('*/*.pt'))
    assert len(samples)==1
    sample_seals = []
    for path in samples:
        refpath = ref_root/'scenes'/path.parent.name/path.name
        for f,identity in [(path,sample_identity),(refpath,reference)]:
            seal=json.loads(f.with_suffix('.json').read_text())
            assert seal['identity_sha256']==identity['sha256'] and seal['payload_sha256']==file_sha(f)
        current=torch.load(path,map_location='cpu',weights_only=False)
        original=torch.load(refpath,map_location='cpu',weights_only=False)
        for key in ['initial_noise','normalized_actions']:
            assert torch.equal(current[key],original[key])
        assert current['noise_key']==original['noise_key']
        for mode in current['records']:
            x,y=current['records'][mode],original['records'][mode]
            assert x['status']==y['status']=='OK' and x['metrics']==y['metrics']
            for key in ['requested_trajectory','simulated_states']:
                assert np.array_equal(np.asarray(x[key]),np.asarray(y[key]))
        sample_seals.append(dict(candidate_payload_sha256=file_sha(path),reference_payload_sha256=file_sha(refpath)))
    patch=out/'evaluation_accounting.patch'
    patch.write_bytes(subprocess.check_output(['git','diff',BASE,'--',EVALUATOR]))
    proof=dict(status='PASS',reference_source_sha256=OLD,evaluation_source_sha256=NEW,
               scientific_calls_and_shifted_ode_AST_identical=True,
               unchanged_numerical_dependency_count=len(hashes),
               rechecked_saved_actions_metrics_and_simulation_exact=True,
               real_scenes_rechecked=1,full_population_equivalence_claimed=False,
               previous_acceptance_sha256=file_sha(prior),sample_payload_hashes=sample_seals,
               new_inference=0,new_scoring=0,
               scope='Manual exact-source diff review plus unchanged dependencies and previously executed raw-observation smoke; not generic equivalence of arbitrary programs')
    atomic_json(out/'qualification.json',proof)
    fields=('seed','modes','profile','feature_cache','actor','scoring')
    receipt=dict(schema='simwam_evaluation_accounting_compatibility_v1',status='QUALIFIED_LOGGING_ONLY',
                 reference_source_sha256=OLD,evaluation_source_sha256=NEW,
                 fixed_protocol={k:reference[k] for k in fields},unchanged_numerical_files=hashes,
                 evidence_files=[dict(role=role,path=str(path.relative_to(repo)),sha256=file_sha(path))
                                 for role,path in [('qualification',out/'qualification.json'),('source_diff',patch)]],
                 explanation='Only exact audited invocation/locking/cost-accounting changes; original manifests and source hashes retained')
    receipt['sha256']=fingerprint(receipt)
    target=out/'compatibility.json'
    atomic_json(target,receipt)
    current=json.loads(Path('/mnt/project/simwam-x-anchor-20261015/evaluations/R_CANONICAL_X1_update005000_'+reference['cohort']+'/manifest.json').read_text())
    checks=[]
    def reject(name,identity,ref,path):
        try:match_protocol(identity,ref,path)
        except (ValueError,KeyError,FileNotFoundError):checks.append(name)
        else:raise AssertionError('Accepted '+name)
    match_protocol(reference,reference)
    checks.append('same_source_keeps_strict_path')
    reject('source_difference_rejected_without_receipt',current,reference,None)
    match_protocol(current,reference,target)
    checks.append('qualified_exact_pair_accepted')
    for key in ('seed','modes','profile','scoring','tokens_sha256','actor','feature_cache'):
        mutation=copy.deepcopy(current);mutation[key]='MUTATED'
        reject('reject_'+key,mutation,reference,target)
    mutation=copy.deepcopy(current);mutation['source_sha256']='0'*64
    reject('reject_unknown_source',mutation,reference,target)
    with tempfile.TemporaryDirectory() as temp:
        fake=Path(temp)/'receipt.json'
        changed=copy.deepcopy(receipt);changed['status']='UNKNOWN'
        atomic_json(fake,changed)
        reject('reject_tampered_receipt',current,reference,fake)
        changed=copy.deepcopy(receipt);changed['evidence_files'][0]['sha256']='0'*64
        changed.pop('sha256');changed['sha256']=fingerprint(changed)
        atomic_json(fake,changed)
        reject('reject_changed_evidence',current,reference,fake)
    atomic_json(out/'tests.json',dict(status='PASS',checks=checks,new_inference=0,new_scoring=0))
    print(json.dumps(dict(status='PASS',receipt=str(target),checks=len(checks),numerical_files=len(hashes))))


if __name__=='__main__':
    main()
