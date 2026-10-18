"""Trace native-contract failure through small source/metadata only."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
CURRENT=Path('/mnt/project/VLA-Drive-st-temporal-combination-20261017')
PARENT=Path('/mnt/project/VLA-Drive-st-indexed-qualified-runtime-v1-20261017')
RESULT=Path('/mnt/project/st-world-policy-20261016/audits/temporal_T3_indexed32_preparation_v1/controls/RESULT.json')
def pin(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def tree(p):return ast.parse(p.read_text())
def dump(x):return ast.dump(x,include_attributes=False)
def main():
    source=CURRENT/'tools/st_world_policy/qualify_indexed_provider.py'
    groups={n.targets[0].id:ast.literal_eval(n.value) for n in tree(source).body
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
        and n.targets[0].id in ('CONTROL_PATHS','MATH_PATHS')}
    rows=[]
    for group,paths in groups.items():
        for rel in paths:
            a,b=pin(PARENT/rel),pin(CURRENT/rel)
            rows.append({'group':group,'relative':rel,'old':a,'new':b,'equal':a['sha256']==b['sha256']})
    mismatches=[r for r in rows if r['group']=='MATH_PATHS' and not r['equal']]
    assert len(mismatches)==1 and mismatches[0]['relative']=='starVLA/dataloader/st_world_history.py'
    old,new=tree(PARENT/mismatches[0]['relative']),tree(CURRENT/mismatches[0]['relative'])
    old_functions={n.name:dump(n) for n in old.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    new_functions={n.name:dump(n) for n in new.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    assert set(new_functions)-set(old_functions)=={'_packed_parent','_packed_version','read_packed_history_range'}
    assert [k for k in old_functions if old_functions[k]!=new_functions[k]]==['HistoryReader']
    oc=next(n for n in old.body if isinstance(n,ast.ClassDef) and n.name=='HistoryReader')
    nc=next(n for n in new.body if isinstance(n,ast.ClassDef) and n.name=='HistoryReader')
    om={n.name:n for n in oc.body if isinstance(n,ast.FunctionDef)}
    nm={n.name:n for n in nc.body if isinstance(n,ast.FunctionDef)}
    assert set(om)==set(nm) and [k for k in om if dump(om[k])!=dump(nm[k])]==['read']
    o,n=om['read'],deepcopy(nm['read'])
    # Preserve exact legacy file operations and all post-read semantic checks:
    # strip only the new recognized-layout guard and packed alternative.
    branch=n.body[2];assert isinstance(branch,ast.If)
    assert "packed_log_v1" in ast.unparse(branch.test)
    assert "Unknown history record layout" in ast.unparse(branch.orelse[0])
    n.body=n.body[:2]+branch.orelse[1:]+n.body[3:]
    assert dump(o)==dump(n)
    actual=json.loads(RESULT.read_text())
    assert actual['status']=='FAILED_NO_AUTO_RETRY'
    resource=actual['resources']
    assert resource['returncode']==2 and resource['cleanup']['live_owned_remaining']==0
    assert 'PROFILE migration changed protected mathematical/reader/PCA/sampler source' in resource['stderr']
    report={'status':'CHANGES_REQUESTED','reviewer':'/root/temporal_narrow_review',
      'actual_failure':pin(RESULT),'failure_stage':'First native prepare-profile-migration; second temporal_composition CLI not reached',
      'actual_read_bytes':actual['charged_rchar_bytes'],'remaining_owned_processes':0,
      'source_inventory':rows,'only_math_path_delta':mismatches[0],
      'semantic_source_comparison':{
        'unchanged_all_original_top_level_functions_except_HistoryReader':True,
        'unchanged_HistoryReader_methods_except_read':True,
        'legacy_read_operations_and_following_record_semantic_checks_AST_identical':True,
        'added_features':'packed_log_v1 bounded byte-range storage plus safe no-follow reader; no target equations/teacher/PCA/sampler delta'},
      'why_nine_tests_missed_it':'The two-CLI fixture mocked runpy.run_module and only asserted command ordering/SystemExit/logging. It never executed prepare_control_migration or validate_control_migration against the real archived/current source pair. Current pin/HEAD checks did not prove archived/current protected-path equality.',
      'reviewer_coverage_gap':'Independent narrow re-run and pin review inherited those mocks and did not inspect MATH_PATHS pair equality before actual admission. This is an acknowledged review omission, not evidence that source hashes should be bypassed.',
      'required_bounded_fix':[
        'Preserve the failed output/receipt and one-shot identity; no implicit retry or overwritten evidence.',
        'Add an explicit, narrow migration proof for this exact already-reviewed legacy-to-packed history storage transition. Preserve all other MATH_PATHS byte equality and existing parent-plan source checks.',
        'Do not delete history from protected math/reader list or accept arbitrary hashes/control relocation; bind exact old/new source and independent packed compatibility evidence.',
        'Test actual validate_control_migration on actual small metadata/in-memory migration, with negative changed history semantics and unrelated math source; do not mock away the contract validator.',
        'Any subsequent preparation needs its own reviewed output namespace/current resource admission. No GPU/model or teacher tensor work is authorized here.'],
      'actual_preparation_retried':False,'gpu_or_ssh_access':False,'ledger_read_or_write':False}
    target=HERE/'ACTUAL_PREPARATION_FAILURE_AUDIT.json'
    assert not target.exists();target.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':report['status'],'report':pin(target),'math_mismatches':len(mismatches)}))
if __name__=='__main__':main()
