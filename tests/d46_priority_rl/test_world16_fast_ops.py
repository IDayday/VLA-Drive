"""Focused CPU-only typed16->16 admission/entry tests; no actor or dispatch."""
import copy
import json
from pathlib import Path
import pytest
import yaml
from starVLA.rl.qwen_world_compare.contracts import fingerprint,file_sha
from tools.d46_priority_rl import world16_owner as o
from tools.qwen_world_compare import job_worker as w


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return value


def sealed(path,value):
    value=copy.deepcopy(value);value.pop('sha256',None);value['sha256']=fingerprint(value);return put(path,value)


@pytest.fixture
def chain(tmp_path,monkeypatch):
    monkeypatch.setattr(w,'D46_ROOT',tmp_path);monkeypatch.setattr(o,'ROOT',tmp_path)
    registration=tmp_path/'PARENT_REGISTRATION_v2.json';registration.write_text('{}')
    contract_sha=file_sha(registration);monkeypatch.setattr(w,'D46_CONTRACT_SHA',contract_sha)
    cfg=dict(arm='D3_FPO',parent='D_IL46K',parent_checkpoint=str(tmp_path/'IL46_PARENT'),
        parent_registration=dict(path=str(registration),file_sha256=contract_sha),training_uses_navtest=False,
        global_scenes=64,group=16,inner_updates=2,target_epochs=6.,mc_samples=4,clip=.01)
    cp8=tmp_path/'runs/D46_D3_FPO/checkpoints/update000002_inner0'
    old=sealed(cp8/'identity.json',dict(schema='d44_native_rl_v1',architecture_id='D',world=8,qualification=False,
        parent_model_sha256=w.D46_PARENT_SHA,scope=dict(trainable_names=['lora']),data=dict(NAVTEST_overlap=0),recipe=cfg,source_files={'fixture.py':'8'*64}))
    put(cp8/'files.json',{'optimizer.pt':'a'*64,'identity.json':file_sha(cp8/'identity.json')})
    put(cp8/'COMPLETE',dict(world=8,phase='READY',next_inner=0,updates=2,files_sha256=file_sha(cp8/'files.json')))
    tx=dict(schema='d46_exact_native_READY_world8_to16_v1',source_world=8,target_world=16,same_algorithm=True,
        old_owner_may_adopt=False,virtual_sampler_lanes=8,global_scenes=64,group=16,inner_updates=2,parent_policy_unchanged=True,
        source_checkpoint=str(cp8),origin_updates=2,source_files_sha256=file_sha(cp8/'files.json'),source_run_identity_sha256=old['sha256'],
        common_optimizer_sha256='a'*64,source_qualification=False,target_qualification=False)
    base=tmp_path/'runs/D46_D3_FPO_WORLD16';value=copy.deepcopy(old);value.update(world=16,fresh_Adam=False,native_world_transition=tx,source_files={'fixture.py':'b'*64})
    baseid=sealed(base/'identity.json',value)
    transition=sealed(base/'WORLD_TRANSITION.json',dict(schema=tx['schema'],source_identity_sha256=old['sha256'],source_checkpoint=str(cp8),
        source_files_sha256=tx['source_files_sha256'],origin_updates=2,target_identity_sha256=baseid['sha256'],common_optimizer_sha256='a'*64,
        native_weights_and_Adam_inherited=True,qualification=False,source_qualification=False,source_inventory=baseid['source_files']))
    cp16=base/'checkpoints/update000004_inner0';put(cp16/'identity.json',baseid)
    put(cp16/'files.json',{'optimizer.pt':'c'*64,'identity.json':file_sha(cp16/'identity.json')})
    put(cp16/'COMPLETE',dict(world=16,phase='READY',next_inner=0,updates=4,files_sha256=file_sha(cp16/'files.json')))
    def upgrade(name='FAST_Q',qualification=True,change=None):
        output=tmp_path/('qualification'if qualification else'runs')/name
        current=copy.deepcopy(baseid);current.update(qualification=qualification,source_files={'fixture.py':'d'*64})
        current['native_world_transition']['target_qualification']=qualification
        current['execution_upgrade']=dict(schema='d46_world16_execution_upgrade_v1',source_world16_checkpoint=str(cp16),
            source_identity_sha256=baseid['sha256'],source_checkpoint_identity_file_sha256=file_sha(cp16/'identity.json'),
            source_checkpoint_files_sha256=file_sha(cp16/'files.json'),common_optimizer_sha256='c'*64,origin_updates=4,
            source_qualification=False,target_qualification=qualification,old_inventory_sha256=fingerprint(baseid['source_files']),
            new_inventory_sha256=fingerprint(current['source_files']),source_world_transition_sha256=transition['sha256'],
            flags=dict(fast_prefix=True,fuse_fm_mc=True,async_scoring=False,rotary_cache=False,no_grad_prefix_cache=False))
        if change:change(current)
        current=sealed(output/'identity.json',current)
        receipt=sealed(output/'EXECUTION_UPGRADE.json',dict(schema='d46_world16_execution_upgrade_v1',execution_upgrade=current['execution_upgrade'],target_identity_sha256=current['sha256']))
        changed=copy.deepcopy(transition);changed.update(target_identity_sha256=current['sha256'],source_inventory=current['source_files'],qualification=qualification,execution_upgrade_sha256=receipt['sha256'])
        sealed(output/'WORLD_TRANSITION.json',changed)
        return output,current
    return dict(root=tmp_path,base=base,baseid=baseid,cp16=cp16,cfg=cfg,upgrade=upgrade)


def test_own8_formal16_READY_to_qualification16_worker_admitted(chain):
    output,current=chain['upgrade']()
    assert w.known_D46_world16_output(chain['base'])
    assert w.known_D46_fast16_output(output)
    assert w.known_owner_args(['python','-m','tools.d46_world16.train_fast','--output',str(output),'--qualification'])
    assert current['qualification']is True and current['execution_upgrade']['source_qualification']is False
    assert current['native_world_transition']['origin_updates']==2 and current['execution_upgrade']['origin_updates']==4
    assert not w.known_D46_fast16_output(chain['base'])


def test_resealed_semantic_tamper_and_unsupported_flags_are_refused(chain):
    changes={
        'recipe':lambda x:x['recipe'].update(mc_samples=2),
        'scope':lambda x:x['scope'].update(trainable_names=['different_adapter']),
        'Adam':lambda x:x['execution_upgrade'].update(common_optimizer_sha256='reset'),
        'sourcefile':lambda x:x['source_files'].update({'fixture.py':'changed'}),
        'flags':lambda x:x['execution_upgrade']['flags'].update(fast_prefix='true'),
        'async':lambda x:x['execution_upgrade']['flags'].update(async_scoring=True),
        'unsupported_cache':lambda x:x['execution_upgrade']['flags'].update(no_grad_prefix_cache=True),
    }
    for name,change in changes.items():
        output,_=chain['upgrade'](name,change=change)
        assert not w.known_D46_fast16_output(output),name
        assert not w.known_owner_args(['python','-m','tools.d46_world16.train_fast','--output',str(output)]),name
    output,_=chain['upgrade']('unsealed')
    receipt=json.loads((output/'EXECUTION_UPGRADE.json').read_text());receipt['tampered']=True;put(output/'EXECUTION_UPGRADE.json',receipt)
    assert not w.known_D46_fast16_output(output)
    output,_=chain['upgrade']('pending_source')
    seal=json.loads((chain['cp16']/'COMPLETE').read_text());seal.update(phase='INNER_COMMITTED',next_inner=1);put(chain['cp16']/'COMPLETE',seal)
    assert not w.known_D46_fast16_output(output)


def test_stage_routes_fast_only_with_explicit_optin_and_bound_receipt(chain,monkeypatch):
    output,identity=chain['upgrade']('FORMAL_FAST',qualification=False);root=chain['root']
    config=root/'cfg.yaml';config.write_text(yaml.safe_dump(chain['cfg']))
    source=root/'SCI';source.mkdir()
    monkeypatch.setattr(o.subprocess,'check_output',lambda args,**kw:'frozen'if'rev-parse'in args else'')
    def plan(output,identity,entry=None):
        value=dict(schema=o.PLAN,stage_id='fast_stage',arm='D3_FPO',output=str(output),science_source=str(source),source_sha='frozen',
            config=str(config),config_sha256=file_sha(config),identity_sha256=identity['sha256'],transition_receipt_sha256=file_sha(output/'WORLD_TRANSITION.json'),
            hosts=['training-node0','training-node1'],master_addr='172.16.1.1',master_port=29785,target_epochs=6.,worker_source=str(root/'OPS'),owner_deadline_unix=None)
        if entry is not None:value['entry']=entry
        path=root/('plan_'+str(entry).replace('.','_')+'.json');put(path,value)
        return o.load_plan(path,check_hold=False)
    bound=plan(output,identity,'tools.d46_world16.train_fast')
    command=o.command(bound,0,str(output/'checkpoints/update000004_inner0'))
    assert 'tools.d46_world16.train_fast'in command and'--qualification'not in command and'--stop-after-update'not in command
    with pytest.raises(ValueError,match='bound entry'):plan(output,identity)
    legacy=plan(chain['base'],chain['baseid'])
    assert 'tools.d46_world16.train'in o.command(legacy,1,'own_READY')
    assert 'tools.d46_world16.train_fast'not in o.command(legacy,1,'own_READY')
