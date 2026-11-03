"""CPU metadata only; no SSH/GPU/controller or live campaign mutations."""
import copy,json,time
from pathlib import Path
import pytest,yaml
from starVLA.rl.qwen_world_compare.contracts import fingerprint,file_sha
from tools.d46_priority_rl import world16_owner as o
from tools.qwen_world_compare import job_worker as w


def put(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))


def test_FPO_stage_plan_native_resume_and_same_logical_arm_ingest(monkeypatch,tmp_path):
    monkeypatch.setattr(o,'ROOT',tmp_path)
    old=tmp_path/'runs/D46_D3_FPO';output=tmp_path/'runs/D46_D3_FPO_WORLD16'
    cfg=dict(arm='D3_FPO',global_scenes=64,group=16,inner_updates=2,target_epochs=6.,mc_samples=4,clip=.01)
    config=tmp_path/'cfg.yaml';config.write_text(yaml.safe_dump(cfg))
    identity=dict(world=16,qualification=False,parent_model_sha256=o.MODEL_SHA,recipe=cfg,native_world_transition=dict(schema='d46_exact_native_READY_world8_to16_v1',old_owner_may_adopt=False,source_qualification=False,target_qualification=False,source_checkpoint=str(old/'checkpoints/update001500_inner0')))
    identity['sha256']=fingerprint(identity);put(output/'identity.json',identity)
    receipt=dict(target_identity_sha256=identity['sha256'],qualification=False);receipt['sha256']=fingerprint(receipt);put(output/'WORLD_TRANSITION.json',receipt)
    source=tmp_path/'SCI';source.mkdir()
    p=dict(schema=o.PLAN,stage_id='FPO_stage',arm='D3_FPO',output=str(output),science_source=str(source),source_sha='frozen',config=str(config),config_sha256=file_sha(config),identity_sha256=identity['sha256'],transition_receipt_sha256=file_sha(output/'WORLD_TRANSITION.json'),hosts=['training-node0','training-node1'],master_addr='172.16.1.1',master_port=29785,target_epochs=6.,worker_source=str(tmp_path/'OPS'),owner_deadline_unix=None)
    plan=tmp_path/'PLAN.json';put(plan,p)
    control=dict(execution_holds={'D3_FPO':'registered native16'},world16_stages={'D3_FPO':dict(plan=str(plan),plan_sha256=file_sha(plan))})
    put(tmp_path/'operations/EXECUTION_POLICY.json',control)
    campaign=dict(runs={'D3_FPO':dict(arm='D3_FPO',output=str(old),config=str(config),config_sha256=file_sha(config),source='original',jobs=[],qualification_jobs=[])},evaluations={})
    put(tmp_path/'CAMPAIGN.json',campaign)
    monkeypatch.setattr(o.subprocess,'check_output',lambda args,**kw:'frozen'if'rev-parse'in args else'')
    p=o.load_plan(plan);state=o.initial(p)
    point=dict(checkpoint=str(output/'checkpoints/update001500_inner0'),files_sha256='full_native',updates=1500,next_inner=0,phase='READY',valid_scene_exposures=48000,scene_cursor=48000)
    monkeypatch.setattr(o,'latest',lambda *a:point)
    monkeypatch.setattr(o,'allowed_hosts',lambda **kw:p['hosts'])
    monkeypatch.setattr(o,'capacity',lambda *a:list(range(8)))
    calls=[]
    def launch(host,source,job,cmd,*a,**kw):
        calls.append(cmd);r=dict(host=host,job=str(job),command=cmd,at=time.time());put(Path(job)/'LAUNCH.json',r);return r
    monkeypatch.setattr(o,'launch',launch)
    o.tick(p,state);o.ingest(campaign,control)
    assert len(calls)==2 and len(campaign['runs'])==1
    assert campaign['runs']['D3_FPO']['world16_stage']['optimizer']=='inherited_AdamW_no_reset'
    for cmd in calls:
        assert cmd[cmd.index('--resume')+1]==point['checkpoint']
        assert '--stop-after-update'not in cmd and'--qualification'not in cmd and'tools.d46_world16.train'in cmd


def test_FPO_worker_preserves_own_READY_recipe_scope_Adam_and_qualification_guards(monkeypatch,tmp_path):
    monkeypatch.setattr(w,'D46_ROOT',tmp_path)
    registration=tmp_path/'PARENT_REGISTRATION_v2.json';registration.write_text('{}')
    sha=file_sha(registration);monkeypatch.setattr(w,'D46_CONTRACT_SHA',sha)
    cp=tmp_path/'runs/D46_D3_FPO/checkpoints/update001500_inner0'
    old=dict(schema='d44_native_rl_v1',architecture_id='D',world=8,qualification=False,parent_model_sha256=w.D46_PARENT_SHA,scope=dict(trainable_names=['lora']),data=dict(NAVTEST_overlap=0),recipe=dict(parent='D_IL46K',arm='D3_FPO',parent_checkpoint=str(tmp_path/'IL46_PARENT'),parent_registration=dict(path=str(registration),file_sha256=sha),training_uses_navtest=False,mc_samples=4,clip=.01))
    old['sha256']=fingerprint(old);put(cp/'identity.json',old);put(cp/'files.json',{'optimizer.pt':'original_Adam'})
    put(cp/'COMPLETE',dict(world=8,phase='READY',next_inner=0,updates=1500,files_sha256=file_sha(cp/'files.json')))
    tx=dict(schema='d46_exact_native_READY_world8_to16_v1',source_world=8,target_world=16,same_algorithm=True,old_owner_may_adopt=False,virtual_sampler_lanes=8,global_scenes=64,group=16,inner_updates=2,parent_policy_unchanged=True,source_checkpoint=str(cp),origin_updates=1500,source_files_sha256=file_sha(cp/'files.json'),source_run_identity_sha256=old['sha256'],common_optimizer_sha256='original_Adam',source_qualification=False,target_qualification=True)
    target=tmp_path/'qualification/FPO16';value=dict(old);value.pop('sha256');value.update(world=16,fresh_Adam=False,qualification=True,native_world_transition=tx);value['sha256']=fingerprint(value);put(target/'identity.json',value)
    receipt=dict(schema=tx['schema'],source_identity_sha256=old['sha256'],source_checkpoint=str(cp),source_files_sha256=tx['source_files_sha256'],origin_updates=1500,target_identity_sha256=value['sha256'],common_optimizer_sha256='original_Adam',native_weights_and_Adam_inherited=True,qualification=True,source_qualification=False);receipt['sha256']=fingerprint(receipt);put(target/'WORLD_TRANSITION.json',receipt)
    assert w.known_D46_world16_output(target)
    for key,bad in [('scope',dict(trainable_names=['other'])),('recipe',dict(value['recipe'],mc_samples=2))]:
        changed=copy.deepcopy(value);changed.pop('sha256');changed[key]=bad;changed['sha256']=fingerprint(changed);put(target/'identity.json',changed)
        assert not w.known_D46_world16_output(target)
    put(target/'identity.json',value)
    changed=dict(receipt);changed.pop('sha256');changed['common_optimizer_sha256']='reset';changed['sha256']=fingerprint(changed);put(target/'WORLD_TRANSITION.json',changed)
    assert not w.known_D46_world16_output(target)
    put(target/'WORLD_TRANSITION.json',receipt)
    seal=json.loads((cp/'COMPLETE').read_text());seal['phase']='INNER_COMMITTED';put(cp/'COMPLETE',seal)
    assert not w.known_D46_world16_output(target)
