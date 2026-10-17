"""Independent tiny controls only. No index/model/real ledger mutation."""
import ast
import argparse
import copy
import importlib.util
from pathlib import Path
import pytest
OWNER=Path('/mnt/project/VLA-Drive-st-full-future-index-prep-20261017/reports/st_world_policy/20261017/full_future_index_preparation_v1')
spec=importlib.util.spec_from_file_location('owner_index_fixture', OWNER/'test_prepare_index_controls.py')
o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)
m=o.m
proposal_inputs=o.proposal_inputs

@pytest.mark.parametrize('role', ['dev','train'])
def test_actual_parser_accepts_exact_whole_root_argv_without_importing_indexer(role):
    src=Path('/mnt/project/VLA-Drive-st-metadata-build-train-v1-20261017/tools/st_world_policy/index_teacher_metadata.py')
    tree=ast.parse(src.read_bytes());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parser')
    namespace=dict(argparse=argparse,Path=Path,MiB=1<<20,GiB=1<<30,__doc__='independent AST-only CLI check')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(src),'exec'),namespace)
    helper=m.load_helper();argv,req=m.index_arguments(helper,role,'/fixture/TEACHER_REQUESTS.json','a'*64,'/fixture/index')
    args=namespace['parser']().parse_args(argv[3:])
    assert args.max_disk_bytes==req['output_bytes']+req['staging_bytes']
    assert args.max_scenes==m.COUNTS[role] and args.cache_root is None

@pytest.mark.parametrize('case',['callback','dependency_bytes','late_train_budget','old_claim','wrong_request'])
def test_late_invalid_train_or_preserved_claim_produces_neither_registration(proposal_inputs,case):
    helper,inputs,out,bindings,old,ledger,pin=proposal_inputs
    source=helper.document(inputs['roles']['train']['source_pins'])
    if case in ('callback','dependency_bytes'):
        name='tools/st_world_policy/index_teacher_metadata.py' if case=='callback' else 'starVLA/dataloader/st_world_metadata_index.py'
        path=Path(source['files'][name]['path'])
        path.write_text("METADATA_STORAGE_CALLBACK_PROTOCOL='wrong'\n" if case=='callback' else '# corrupt source bytes\n')
        if case=='callback':
            import hashlib
            source['files'][name]['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            inputs['roles']['train']['source_pins']=pin('changed_source.json',source)
    elif case=='late_train_budget':
        resource=helper.document(inputs['resource_plan']);resource['roles']['train']['index_claim_inclusive_mib']-=1
        inputs['resource_plan']=pin('wrong_budget.json',resource)
    elif case=='old_claim':
        ledger['claims'][old['claims'][0]['root']]['claim']['claim_id']='replaced'
        inputs['old_ledger']=pin('wrong_claim.json',ledger)
        # fixture shares claim dict: repin original policy with the immutable old identity.
    else: bindings['train']['request_identity']='c'*64
    with pytest.raises((ValueError,KeyError)):
        m.index_proposal(helper,inputs,out,bindings)
    assert not list(out.iterdir())

def test_original_helper_source_pin_is_actual():
    import hashlib
    source=OWNER.parent/'full_future_metadata_preparation_v1/prepare_controls.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==m.HELPER_SHA
    assert 'torch' not in __import__('sys').modules
