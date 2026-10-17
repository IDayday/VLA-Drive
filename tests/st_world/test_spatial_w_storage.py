"""Temporary CPU controls only: no real ledger, model, image or W extraction."""
import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.st_world_policy import cache_storage_budget as b
from tools.st_world_policy import metadata_storage as m
from tools.st_world_policy import spatial_w_storage as w
from test_metadata_storage_guard import fixture as metadata_fixture, write


def seal(doc, key='identity'):
    doc = {k:v for k,v in doc.items() if k != key}
    return {**doc, key:b._digest(doc)}


def pin(path, doc):
    return dict(path=str(path), sha256=write(path, doc))


def signature_fixture(tmp_path, arm='S-P0', split='train', *, hidden=8, revision='v1'):
    """Real control schemas. Missing tensor/image files intentionally never opened."""
    armroot = tmp_path / (arm + '-' + revision)
    source = seal(dict(schema='ddpolicy_current_cameras_v1', split=split, index_sha256=b._digest(split)))
    current = tmp_path / ('current-' + split)
    write(current / 'identity.json', source)
    rows = [dict(token=f'{split}{i}', log='fixture-log', current_record_sha256=b._digest([split,i]))
            for i in range(64 if split == 'train' else 16)]
    manifest = pin(tmp_path / (split + '-manifest.json'), dict(schema='st_fixed_scene_list_v1',
        split=split, source_index_hash=source['index_sha256'], scenes=rows))
    observation = seal(dict(schema='st_observation_only_v1', current_source=source['identity'],
        manifest_sha256=manifest['sha256'], split=split, image_pins={r['token']:[
            [str(tmp_path / 'absent-images' / (r['token'] + '_' + str(view))), b._digest([r,view])]
            for view in range(3)] for r in rows}, history=None, targets_read=False, teacher_cache_read=False))
    base = seal(dict(schema='st_world_bound_s0_v1',
        parent_identity=dict(config=dict(framework=dict(qwenvl=dict(vl_hidden_dim=hidden)))),
        inference_contract={'precision':'FP32_master_eval_TF32_off','flow_steps':10}), 'binding_sha256')
    base_ref = pin(armroot / 'base.json', base)
    cfg = dict(schema='st_training_v1', execution=dict(scope='FORMAL', world_size=1),
        space_time=dict(cell=arm), data=dict(history_root=None), runtime=dict(output_root=str(armroot / 'runs')))
    cfg_ref = pin(armroot / 'config.json', cfg)
    reg_ref = pin(armroot / 'registration.json', dict(run_id='native', artifacts=dict(config=cfg_ref,base_binding=base_ref)))
    run = armroot / 'runs' / 'native'
    run_id = seal(dict(schema='st_world_training_v1', config=cfg, source_sha='1'*40,
        registration_sha256=reg_ref['sha256'], base_binding=base['binding_sha256'], weights_usable_for_formal_init=True))
    write(run / 'identity.json', run_id)
    files = {name:dict(bytes=1,sha256=b._digest(name)) for name in
        ('mp_rank_00_model_states.pt','bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt','rng_rank0.pt')}
    receipt = seal(dict(schema='st_native_checkpoint_v1', world_size=1, files=files,
        state=dict(identity=run_id['identity'],tag='step005000',completed=5000,termination_reason='MAX_STEPS')), 'receipt_identity')
    receipt_ref = pin(run / 'checkpoints' / 'step005000' / 'COMPLETE.json', receipt)
    checkpoint = seal(dict(schema='st_world_deployment_checkpoint_v1',run=str(run),tag='step005000',
        run_identity=run_id['identity'],completed=5000,receipt_identity=receipt['receipt_identity'],
        base_binding=base['binding_sha256'],native_checkpoint_sha256=b._digest({k:v['sha256'] for k,v in files.items()}),
        native_receipt_sha256=receipt_ref['sha256'],training_source_sha='1'*40,scope='FORMAL',
        weights_usable_for_formal_init=True,inference_contract=base['inference_contract'],
        deployment_source_contract={'frozen.py':'a'*64}), 'sha256')
    binding = seal(dict(schema='st_spatial_w_binding_v1', source_sha='2'*40,
        environment=dict(base=dict(schema='st_training_environment_v1',python='fixture',packages={'torch':'fixture'},
                                   cuda_runtime=None,cudnn=None),scikit_learn='fixture'), device='cpu',
        training_registration=reg_ref,checkpoint_tag='step005000',checkpoint_identity=checkpoint['sha256'],arm=arm,
        observations=dict(current_root=str(current),manifest=manifest,identity=observation['identity'],split=split),
        hidden_dim=hidden,memory_tokens=288 if arm == 'S-P3' else 144,max_seconds=600))
    signature = seal(dict(schema=w.SIGNATURE_SCHEMA,binding=binding,checkpoint=checkpoint,observations=observation,
        rows=rows,precision='FP32_native_master_TF32_off',targets_read=False,
        memory_mask='all learned W slots valid; no label-derived mask'))
    signature_path = armroot / (split + '-signature.json')
    signature_sha = write(signature_path,signature)
    root = tmp_path / 'cache' / ('w-' + arm + '-' + split + '-' + revision)
    claim = w.claim_from_signature(root=root,claim_id=f'w-{arm}-{split}-{revision}',
        signature_path=signature_path,signature_sha256=signature_sha)
    return SimpleNamespace(**locals())


def fixture(tmp_path, cells=(('S-P0','train'),('S-P0','dev')), hidden=8):
    m0 = metadata_fixture(tmp_path)
    m.migrate_policy(**m0.kwargs)
    before = json.loads(m0.old.state_path.read_text())
    old_path, old_doc = m0.new_path, m0.new_doc
    old_sha = hashlib.sha256(old_path.read_bytes()).hexdigest()
    ledger_sha = hashlib.sha256(m0.old.state_path.read_bytes()).hexdigest()
    signatures = [signature_fixture(tmp_path, arm, split, hidden=hidden) for arm,split in cells]
    new_doc = copy.deepcopy(old_doc)
    new_doc['claims'].extend(s.claim for s in signatures)
    new_doc['frozen_w_package'] = dict(schema=w.PACKAGE_SCHEMA,maximum_bytes=b.GIB,
        baseline_policy_sha256=old_sha,baseline_ledger_sha256=ledger_sha,
        baseline_claims={r['root']:copy.deepcopy(r) for r in old_doc['claims']},
        baseline_legacy_reservations=copy.deepcopy(old_doc['legacy_reservations']),
        metadata_package_sha256=b._digest(old_doc['metadata_package']),
        members={s.claim['root']:copy.deepcopy(s.claim) for s in signatures})
    new_path = tmp_path / 'v6-w.json'; new_sha = write(new_path,new_doc)
    inventory_path = tmp_path / 'w-inventory.json'
    inventory_sha = write(inventory_path,dict(schema='st_metadata_pilot_inventory_v1',roots=[]))
    q = copy.deepcopy(m0.q)
    q['bindings'] = dict(old_policy_sha256=old_sha,new_policy_sha256=new_sha,old_ledger_sha256=ledger_sha,
        pilot_inventory_sha256=inventory_sha,checkpoint_sha256=m0.checkpoint_sha)
    q['checked_writer_roots'] = [r['root'] for r in old_doc['claims'] + old_doc['legacy_reservations']]
    q_path = tmp_path / 'w-quiescence.json'; q_sha = write(q_path,q)
    kwargs = dict(old_policy=old_path,old_policy_sha256=old_sha,new_policy=new_path,new_policy_sha256=new_sha,
        expected_old_ledger_sha256=ledger_sha,quiescence_receipt=q_path,quiescence_sha256=q_sha,
        pilot_inventory=inventory_path,pilot_inventory_sha256=inventory_sha,
        checkpoint_receipt=m0.checkpoint_path,checkpoint_sha256=m0.checkpoint_sha)
    return SimpleNamespace(**locals())


def refresh(f):
    f.new_sha=write(f.new_path,f.new_doc)
    f.kwargs['new_policy_sha256']=f.new_sha
    f.q['bindings']['new_policy_sha256']=f.new_sha
    f.kwargs['quiescence_sha256']=write(f.q_path,f.q)


def test_migration_is_typed_separate_and_old_claims_history_preserved(tmp_path):
    f=fixture(tmp_path)
    result=m.migrate_policy(**f.kwargs)
    state=json.loads(f.m0.old.state_path.read_text())
    assert state['claims']==f.before['claims']
    assert state['metadata_migrations'][:-1]==f.before['metadata_migrations']
    assert state['core']['metadata_package']==f.old_doc['metadata_package']
    assert result['metadata_package_reserved_bytes']==4*m.MIB
    assert result['frozen_w_package_reserved_bytes']==sum(b.StorageBudget._total(s.claim) for s in f.signatures)
    assert m.migrate_policy(**f.kwargs)==result
    with pytest.raises(b.StorageBudgetError,match='different policy'):b.StorageBudget(f.old_path).initialize()
    budget=b.StorageBudget(f.new_path)
    for s in f.signatures:
        required={k:s.claim[k] for k in w.AMOUNTS}
        handle=budget.reserve(s.root,s.signature['identity'],**required)
        assert handle.receipt['reserved_bytes']==sum(required.values())
        with handle.writing(16384):
            s.root.mkdir(parents=True)
            write(s.root/'identity.json',s.signature)
        assert budget.reserve(s.root,s.signature['identity'],**required).claim==s.claim
    assert len(json.loads(budget.state_path.read_text())['claims'])==3
    with m.metadata_writing(**{**f.m0.guard,'policy_path':f.new_path,'policy_sha256':f.new_sha}) as ctx:
        ctx.check(before_commit=True)
    assert json.loads(budget.state_path.read_text())['core']['metadata_package']['maximum_bytes']==4*b.GIB


@pytest.mark.parametrize('stage',['after_pending','after_snapshot','after_state','after_receipt'])
def test_w_migration_fault_retry_keeps_original_snapshot_and_claims(tmp_path,stage):
    f=fixture(tmp_path)
    def fault(name):
        if name==stage:raise RuntimeError(stage)
    with pytest.raises(RuntimeError):m.migrate_policy(**f.kwargs,fault=fault)
    result=m.migrate_policy(**f.kwargs)
    assert result==m.migrate_policy(**f.kwargs)
    state=json.loads(f.m0.old.state_path.read_text())
    assert state['claims']==f.before['claims']
    assert len(state['metadata_migrations'])==2
    assert len(list(f.m0.old.state_dir.glob('*.snapshot.json')))==2


@pytest.mark.parametrize('change',['no_package','wrong_kind','unknown_kind','signature_sha','signature_content',
    'shape','dtype','amount','overlap','metadata_baseline','w_baseline','w_metadata_pin','max_plus_one'])
def test_wrong_admission_rejected_without_ledger_mutation(tmp_path,change):
    f=fixture(tmp_path)
    row=f.new_doc['claims'][-1]
    if change=='no_package':del f.new_doc['frozen_w_package']
    if change=='wrong_kind':row['identity_kind']=m.REG_SCHEMA
    if change=='unknown_kind':row['identity_kind']='unreviewed_tensor'
    if change=='signature_sha':row['signature_sha256']='f'*64
    if change=='signature_content':Path(row['signature_path']).write_text('{}')
    if change=='shape':row['tensor_schema']['W']['shape'][0]+=1
    if change=='dtype':row['tensor_schema']['W']['dtype']='bfloat16'
    if change=='amount':row['staging_bytes']+=1
    if change=='overlap':row['root']=f.new_doc['claims'][-2]['root']+'/child'
    if change=='metadata_baseline':f.new_doc['metadata_package']['maximum_bytes']-=1
    if change=='w_baseline':f.new_doc['frozen_w_package']['baseline_claims'].pop(f.old_doc['claims'][0]['root'])
    if change=='w_metadata_pin':f.new_doc['frozen_w_package']['metadata_package_sha256']='f'*64
    if change=='max_plus_one':f.new_doc['frozen_w_package']['maximum_bytes']=b.GIB+1
    if change in ('shape','dtype','amount','overlap'):
        old_root=str(f.signatures[-1].root)
        del f.new_doc['frozen_w_package']['members'][old_root]
        f.new_doc['frozen_w_package']['members'][row['root']]=copy.deepcopy(row)
    refresh(f)
    with pytest.raises((b.StorageBudgetError,KeyError)):m.migrate_policy(**f.kwargs)
    assert hashlib.sha256(f.m0.old.state_path.read_bytes()).hexdigest()==f.ledger_sha


def test_declared_w_aggregate_exact_boundary_and_one_byte_over(tmp_path):
    f=fixture(tmp_path)
    total=sum(b.StorageBudget._total(s.claim) for s in f.signatures)
    f.new_doc['frozen_w_package']['maximum_bytes']=total
    refresh(f)
    assert b.StorageBudget(f.new_path).core['frozen_w_package']['maximum_bytes']==total
    f.new_doc['frozen_w_package']['maximum_bytes']-=1;refresh(f)
    with pytest.raises(b.StorageBudgetError,match='aggregate frozen W'):b.StorageBudget(f.new_path)


def test_single_campaign_cap_counts_all_declarations_exactly_once(tmp_path):
    f=fixture(tmp_path)
    expected=f.new_doc['ledger_reserve_bytes']+sum(r['reserved_bytes'] for r in f.new_doc['legacy_reservations'])+sum(
        b.StorageBudget._total(r) for r in f.new_doc['claims'])
    f.new_doc['cache_cap_bytes']=expected;refresh(f)
    assert b.StorageBudget(f.new_path).core['cache_cap_bytes']==expected
    f.new_doc['cache_cap_bytes']-=1;refresh(f)
    with pytest.raises(b.StorageBudgetError,match='all declared'):b.StorageBudget(f.new_path)


def test_all_eight_exact_scopes_formula_and_ninth_rejected(tmp_path):
    # Larger campaign cap is fixed at initial creation for this local fixture.
    f=fixture(tmp_path,cells=[(arm,split) for arm in w.ARMS for split in ('train','dev')],hidden=2048)
    f.new_doc['cache_cap_bytes']=768*b.GIB;refresh(f)
    budget=b.StorageBudget(f.new_path)
    assert w.validate_package(budget.core,budget.declarations)==546104992
    duplicate=copy.deepcopy(f.new_doc['claims'][-1]);duplicate['root']+='-extra'
    duplicate['claim_id']='ninth';duplicate['cache_identity']='b'*64
    f.new_doc['claims'].append(duplicate)
    f.new_doc['frozen_w_package']['members'][duplicate['root']]=copy.deepcopy(duplicate);refresh(f)
    with pytest.raises(b.StorageBudgetError,match='one to eight'):b.StorageBudget(f.new_path)


def test_duplicate_arm_split_and_train_dev_different_native_endpoint_reject(tmp_path):
    f=fixture(tmp_path)
    changed=signature_fixture(tmp_path,'S-P0','dev',revision='v2')
    del f.new_doc['frozen_w_package']['members'][f.new_doc['claims'][-1]['root']]
    f.new_doc['claims'][-1]=changed.claim
    f.new_doc['frozen_w_package']['members'][changed.claim['root']]=copy.deepcopy(changed.claim);refresh(f)
    with pytest.raises(b.StorageBudgetError,match='same arm checkpoint'):b.StorageBudget(f.new_path)
    changed=signature_fixture(tmp_path,'S-P0','train',revision='v2')
    del f.new_doc['frozen_w_package']['members'][f.new_doc['claims'][-1]['root']]
    f.new_doc['claims'][-1]=changed.claim
    f.new_doc['frozen_w_package']['members'][changed.claim['root']]=copy.deepcopy(changed.claim);refresh(f)
    with pytest.raises(b.StorageBudgetError,match='duplicate frozen W arm/split'):b.StorageBudget(f.new_path)


@pytest.mark.parametrize('change',['source','checkpoint','scope','completed','targets','precision','rows','population',
    'observation','mask','base_binding','runtime','native_receipt','registration','source_control','run_control'])
def test_wrong_signature_or_pinned_dependencies_rejected(tmp_path,change):
    s=signature_fixture(tmp_path)
    doc=copy.deepcopy(s.signature)
    if change=='source':doc['binding']['source_sha']='not-a-commit'
    if change=='checkpoint':doc['binding']['checkpoint_identity']='a'*64
    if change=='scope':doc['checkpoint']['scope']='SMOKE'
    if change=='completed':doc['checkpoint']['completed']=4999
    if change=='targets':doc['targets_read']=True
    if change=='precision':doc['precision']='BF16_then_float'
    if change=='rows':doc['rows'][0]['token']='foreign'
    if change=='population':doc['binding']['observations']['split']='navtest'
    if change=='observation':doc['observations']['history']='a'*64
    if change=='mask':doc['memory_mask']='labels define valid mask'
    if change=='base_binding':doc['checkpoint']['base_binding']='a'*64
    if change=='runtime':doc['binding']['environment']={}
    if change=='native_receipt':Path(s.receipt_ref['path']).write_text('{}')
    if change=='registration':Path(s.reg_ref['path']).write_text('{}')
    if change=='source_control':write(s.current/'identity.json',seal({**s.source,'index_sha256':'a'*64}))
    if change=='run_control':write(s.run/'identity.json',seal({**s.run_id,'source_sha':'f'*40}))
    doc['checkpoint']=seal(doc['checkpoint'],'sha256')
    doc['binding']=seal(doc['binding']);doc['observations']=seal(doc['observations']);doc=seal(doc)
    with pytest.raises((b.StorageBudgetError,KeyError)):w.validate_signature(doc)


def test_later_old_claim_mutation_and_frozen_package_change_reject(tmp_path):
    f=fixture(tmp_path);m.migrate_policy(**f.kwargs)
    row=f.signatures[0].claim;budget=b.StorageBudget(f.new_path)
    budget.reserve(row['root'],row['cache_identity'],**{k:row[k] for k in w.AMOUNTS})
    f.new_doc['claims'][0]['output_bytes']+=1;refresh(f)
    with pytest.raises(b.StorageBudgetError,match='baseline claim'):b.StorageBudget(f.new_path)
    f.new_doc['claims'][0]['output_bytes']-=1
    f.new_doc['frozen_w_package']['maximum_bytes']-=1;refresh(f)
    with pytest.raises(b.StorageBudgetError,match='different policy'):b.StorageBudget(f.new_path).initialize()


def test_frozen_w_cannot_be_disguised_as_metadata_registration(tmp_path):
    s=signature_fixture(tmp_path)
    row={**s.claim,'identity_kind':m.REG_SCHEMA,'registration_path':str(s.signature_path),
         'registration_sha256':s.signature_sha,'cache_identity':s.signature_sha}
    with pytest.raises(b.StorageBudgetError,match='registration/output'):m._registration(row)


def test_shape_budget_matches_unchanged_extractor_and_no_payload_open(tmp_path,monkeypatch):
    from tools.st_world_policy import spatial_probe as probe
    s=signature_fixture(tmp_path)
    for arm in w.ARMS:
        for scenes in (16,64):
            tokens=288 if arm=='S-P3' else 144
            assert w._requirements(scenes,tokens,2048)==probe.cache_requirements(scenes,tokens,2048)
    original=Path.open
    opened=[]
    def checked(path,*args,**kwargs):
        opened.append(str(path))
        assert path.suffix=='.json' and path.name!='index.json'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',checked)
    binding,required=w.validate_signature(s.signature)
    assert required==probe.cache_requirements(64,binding['memory_tokens'],binding['hidden_dim'])
    assert opened and all('absent-images' not in p for p in opened)


def test_actual_cache_plan_signature_and_extractor_reserve_call_are_compatible(tmp_path,monkeypatch):
    """Use real plan/extract entry points, mock only native/data execution.

    No tensor output/teacher/native model is produced: stop immediately at the
    loader boundary after the existing extractor publishes its small identity.
    """
    from tools.st_world_policy import spatial_probe as p, checkpoints
    from starVLA.dataloader import st_world_observation
    f=fixture(tmp_path,cells=(('S-P0','train'),))
    s=f.signatures[0]
    class ObservationFixture:
        rows=s.rows
        identity=s.observation
        def __init__(self,*args):pass
        def __len__(self):return len(self.rows)
        def __getitem__(self,index):raise AssertionError('No image/data execution allowed')
    monkeypatch.setattr(st_world_observation,'STObservationDataset',ObservationFixture)
    monkeypatch.setattr(checkpoints,'checkpoint_identity',lambda *_:(s.cfg,s.base,s.checkpoint))
    monkeypatch.setattr(p,'source_runtime',lambda *a,**k:None)
    signature,data,required=p.cache_plan(s.binding)
    # JSON serializable representation must be exactly the original signature.
    assert signature==s.signature
    assert required=={k:s.claim[k] for k in w.AMOUNTS}
    m.migrate_policy(**f.kwargs)
    def stop(*a,**k):raise RuntimeError('intentional before native loader')
    monkeypatch.setattr(checkpoints,'load_world_checkpoint',stop)
    with pytest.raises(RuntimeError,match='before native loader'):
        p.extract(s.binding,s.root,f.new_path, checkpoint_commitments=dict(
            path=str(f.m0.checkpoint_path), sha256=f.m0.checkpoint_sha))
    assert json.loads((s.root/'identity.json').read_text())==s.signature
    assert not list(s.root.glob('*.pt'))
    state=json.loads(f.m0.old.state_path.read_text())
    assert state['claims'][str(s.root)]['claim']==s.claim


@pytest.mark.parametrize('kind',['quiescence','checkpoint','free'])
def test_w_migration_preserves_resource_fail_closed_checks(tmp_path,monkeypatch,kind):
    f=fixture(tmp_path)
    if kind=='quiescence':
        f.q['valid_until_unix']=0
        f.kwargs['quiescence_sha256']=write(f.q_path,f.q)
    if kind=='checkpoint':
        doc=json.loads(f.m0.checkpoint_path.read_text());doc['valid_until_unix']=0
        sha=write(f.m0.checkpoint_path,doc);f.kwargs['checkpoint_sha256']=sha
        f.q['bindings']['checkpoint_sha256']=sha;f.kwargs['quiescence_sha256']=write(f.q_path,f.q)
    if kind=='free':monkeypatch.setattr(b.shutil,'disk_usage',lambda _:SimpleNamespace(free=1))
    with pytest.raises(b.StorageBudgetError):m.migrate_policy(**f.kwargs)
    assert hashlib.sha256(f.m0.old.state_path.read_bytes()).hexdigest()==f.ledger_sha


def test_per_write_validates_controls_without_tensor_hash_or_unbounded_scan(tmp_path,monkeypatch):
    f=fixture(tmp_path);m.migrate_policy(**f.kwargs)
    s=f.signatures[0]
    handle=b.StorageBudget(f.new_path).reserve(s.root,s.signature['identity'],**{k:s.claim[k] for k in w.AMOUNTS})
    original=Path.open
    opened=[]
    def small_only(path,*args,**kwargs):
        opened.append(path)
        assert path.suffix in ('.json','.lock') or '.tmp' in path.name
        assert path.name!='index.json' and 'absent-images' not in str(path)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',small_only)
    with handle.writing(8192):pass
    assert opened
    # Failed attempts retain their charge, rather than releasing it as if the
    # writer had never started; admission reconciles stat only on explicit retry.
    charged=json.loads(f.m0.old.state_path.read_text())['claims'][str(s.root)]['charged_bytes']
    with pytest.raises(RuntimeError):
        with handle.writing(8192):raise RuntimeError('partial writer')
    assert json.loads(f.m0.old.state_path.read_text())['claims'][str(s.root)]['charged_bytes']==charged+8192


def test_package_requires_existing_ledger_and_cannot_change_state_dir(tmp_path):
    f=fixture(tmp_path)
    f.new_doc['state_dir']=str(tmp_path/'second-ledger');refresh(f)
    with pytest.raises(b.StorageBudgetError,match='explicit migration'):b.StorageBudget(f.new_path).initialize()
    with pytest.raises(b.StorageBudgetError,match='immutable campaign'):m.migrate_policy(**f.kwargs)


def test_signature_pin_swap_and_source_link_rejected(tmp_path):
    f=fixture(tmp_path)
    path=f.signatures[0].signature_path
    path.write_text(path.read_text()+' ')
    with pytest.raises(b.StorageBudgetError,match='SHA mismatch'):b.StorageBudget(f.new_path)
    s=signature_fixture(tmp_path/'separate')
    path=s.current/'identity.json';target=path.with_name('other.json')
    path.rename(target);path.symlink_to(target)
    with pytest.raises(b.StorageBudgetError,match='symlink'):w.validate_signature(s.signature)


def test_admitted_w_claim_cannot_be_swapped_for_a_valid_different_endpoint(tmp_path):
    f=fixture(tmp_path,cells=(('S-P0','train'),))
    m.migrate_policy(**f.kwargs)
    old=f.signatures[0]
    budget=b.StorageBudget(f.new_path)
    budget.reserve(old.root,old.signature['identity'],**{k:old.claim[k] for k in w.AMOUNTS})
    updated=signature_fixture(tmp_path,'S-P0','train',revision='new-endpoint')
    f.new_doc['claims'][-1]={**updated.claim,'root':str(old.root),'claim_id':old.claim['claim_id']}
    refresh(f)
    with pytest.raises(b.StorageBudgetError,match='member declarations'):b.StorageBudget(f.new_path).initialize()
    f.new_doc['frozen_w_package']['members'][str(old.root)]=copy.deepcopy(f.new_doc['claims'][-1]);refresh(f)
    with pytest.raises(b.StorageBudgetError,match='different policy'):b.StorageBudget(f.new_path).initialize()


def test_w_cannot_hide_inside_original_metadata_baseline(tmp_path):
    f=fixture(tmp_path)
    row=f.signatures[0].claim
    f.new_doc['frozen_w_package']['baseline_claims'][row['root']]=row
    refresh(f)
    with pytest.raises(b.StorageBudgetError,match='cannot be hidden'):b.StorageBudget(f.new_path)


def test_add_later_arm_split_requires_same_explicit_migration_and_preserves_members(tmp_path):
    f=fixture(tmp_path,cells=(('S-P0','train'),))
    m.migrate_policy(**f.kwargs)
    before=json.loads(f.m0.old.state_path.read_text())
    new=copy.deepcopy(f.new_doc)
    s=signature_fixture(tmp_path,'S-P0','dev')
    new['claims'].append(s.claim)
    new['frozen_w_package']['members'][s.claim['root']]=copy.deepcopy(s.claim)
    path=tmp_path/'v7-w.json';sha=write(path,new)
    with pytest.raises(b.StorageBudgetError,match='different policy'):b.StorageBudget(path).initialize()
    old_ledger_sha=hashlib.sha256(f.m0.old.state_path.read_bytes()).hexdigest()
    q=copy.deepcopy(f.q)
    q['bindings'].update(old_policy_sha256=f.new_sha,new_policy_sha256=sha,old_ledger_sha256=old_ledger_sha)
    q['checked_writer_roots']=[r['root'] for r in f.new_doc['claims']+f.new_doc['legacy_reservations']]
    q_path=tmp_path/'v7-quiescence.json';q_sha=write(q_path,q)
    kwargs={**f.kwargs,'old_policy':f.new_path,'old_policy_sha256':f.new_sha,'new_policy':path,
        'new_policy_sha256':sha,'expected_old_ledger_sha256':old_ledger_sha,
        'quiescence_receipt':q_path,'quiescence_sha256':q_sha}
    result=m.migrate_policy(**kwargs)
    state=json.loads(f.m0.old.state_path.read_text())
    assert state['claims']==before['claims']
    assert len(state['metadata_migrations'])==3
    assert state['core']['metadata_package']==f.old_doc['metadata_package']
    assert result['frozen_w_package_reserved_bytes']==sum(b.StorageBudget._total(r) for r in new['frozen_w_package']['members'].values())
