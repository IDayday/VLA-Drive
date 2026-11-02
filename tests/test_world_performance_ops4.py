"""CPU-only registration/export/typed-lineage contracts; no real dispatch."""
import copy,json
from pathlib import Path
import pytest,yaml
from tools.qwen_world_performance import launch as L,evaluate as E,execution as X
from starVLA.rl.qwen_world_research_data.manifest import fingerprint
from starVLA.rl.simwam_recipe.lineage import SCIENTIFIC_KEYS
from starVLA.rl.qwen_world_safety_exploration.recovery import tensor_digest


def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
def sealed(v):return dict(v,sha256=fingerprint(v))


def test_existing_seven_and_optional_only_eighth_family_with_exact_packed_export_receipt(tmp_path,monkeypatch):
    real=L.read(L.ROOT/'RUNS.json');L.validate_registry(real)
    # Recipe mathematics is separately qualified by the packed science source.
    # This fixture tests family admission without importing or changing it.
    base=next(r for r in real['runs']if r['id']=='S0_LARGE_SAFETYEP');cfg=yaml.safe_load(Path(base['config']).read_text())
    cfg['research']['arm']=L.FAST;cfg['research']['performance']['microbatch_scenes']=4
    path=tmp_path/'fast.yaml';path.write_text(yaml.safe_dump(cfg));fast=copy.deepcopy(base)
    fast.update(id=L.FAST,root=str(L.ROOT/'runs'/L.FAST),config=str(path),config_sha256=L.sha(path),trainer_module='tools.qwen_world_multiscene.train')
    orig=L.validate_run_recipe
    monkeypatch.setattr(L,'validate_run_recipe',lambda r:cfg if r['id']==L.FAST else orig(r))
    r=copy.deepcopy(real);r['runs'].append(fast);assert len(L.validate_registry(r)['runs'])==8
    bad=copy.deepcopy(r);bad['runs'][-1]['id']='UNREGISTERED_PACK'
    with pytest.raises(ValueError,match='Only original seven'):L.validate_registry(bad)
    cp=dict(path=str(tmp_path/'cp'),files_sha256='files',updates=808)
    receipt=dict(schema='world_packed_export_native_validation_v1',status='PASS',checkpoint=cp['path'],files_sha256='files',counts=dict(updates=808),micro_scenes=4)
    assert E.validate_native_export_receipt(fast,cp,receipt)==receipt
    with pytest.raises(ValueError,match='Packed export'):E.validate_native_export_receipt(fast,cp,dict(receipt,micro_scenes=1))
    with pytest.raises(ValueError,match='Wrong exact'):orig(dict(fast,trainer_module='arbitrary.train'))


def test_frozen_reference_execution_is_receipt_bound_native_only_and_uses_typed_flag(tmp_path,monkeypatch):
    X._PROOFS.clear();source=tmp_path/'execution';helper=source/X.CACHE_HELPER;helper.parent.mkdir(parents=True);helper.write_text('fixture source, not a real GPU qualification')
    root=tmp_path/'run';cp=root/'checkpoints/update000004_inner0';cp.mkdir(parents=True);(cp/'COMPLETE').write_text('{}')
    dump(cp/'files.json',{})
    old={k:None for k in SCIENTIFIC_KEYS};old.update(recipe=dict(research=dict(arm='S0_LARGE_SAFETYEP')),purpose='formal',world=8,source_files={},execution_identity={'fields':{}},execution_optimizations={})
    old=sealed(old);dump(root/'identity.json',old);dump(cp/'identity.json',old)
    inventory={X.CACHE_HELPER:L.sha(helper)};new=copy.deepcopy(old);new.pop('sha256')
    new.update(source_files=inventory,execution_identity=dict(fields=dict(module_root=str(source),source_files=inventory)),
               execution_optimizations=dict(frozen_IL_reference_cache=dict(schema='world_frozen_IL_reference_W3_terms_cache_v1')))
    new=sealed(new)
    single=tmp_path/'single.json';world=tmp_path/'world.json'
    dump(single,dict(schema=X.SINGLE,status='PASS'))
    dump(world,dict(schema=X.WORLD8,status='PASS',world=8,cache_helper_sha256=inventory[X.CACHE_HELPER],qualified_source_files_sha256=tensor_digest(inventory)))
    receipt=tmp_path/'receipt.json';dump(receipt,sealed(dict(schema='world_frozen_reference_cache_execution_migration_draft_v1',old_identity=old,new_identity=new,parent_files_sha256=L.sha(cp/'files.json'),evidence=[dict(path=str(p),sha256=L.sha(p))for p in (single,world)])))
    monkeypatch.setattr(X.subprocess,'check_output',lambda *a,**kw:'fixturecommit\n')
    run=dict(id='S0_LARGE_SAFETYEP',root=str(root),source='oldscientific5e',trainer_module='tools.qwen_world_performance.train',
             execution_source=str(source),execution_source_commit='fixturecommit',execution_optimizations=dict(schema=X.SCHEMA,frozen_reference_cache=True),
             execution_migration_receipt=str(receipt),execution_migration_receipt_sha256=L.sha(receipt),execution_parent_checkpoint=str(cp))
    cwd,flags=X.command_execution(run,str(cp));assert cwd==str(source)
    assert flags==['--frozen-reference-cache','--migration-receipt',str(receipt)]
    with pytest.raises(ValueError,match='native resume'):X.command_execution(run,None)
    with pytest.raises(ValueError,match='Arbitrary'):X.validate_execution_fields(dict(run,execution_flags=['--change-reward']))
    with pytest.raises(ValueError,match='FAST_PACK4'):X.validate_execution_fields(dict(run,id=L.FAST))
    dump(world,dict(schema=X.WORLD8,status='FAIL'))
    with pytest.raises(ValueError,match='evidence changed'):X.command_execution(run,str(cp))


def test_cached_attempt_return_binds_actual_cwd_and_keeps_scientific_provenance(tmp_path,monkeypatch):
    import time,inspect
    from tools.qwen_world_performance import continuation as C
    from tools.qwen_world_safety_exploration import job_worker
    monkeypatch.setattr(C,'ROOT',tmp_path)
    root=tmp_path/'runs/own';root.mkdir(parents=True);dump(root/'identity.json',dict(sha256='logical5e'))
    cp=dict(path=str(root/'checkpoints/cp'),updates=8,world=8,next_inner=0)
    monkeypatch.setattr(C,'inspect_native',lambda run,p:cp);dump(root/'latest.json',dict(checkpoint=cp['path']))
    run=dict(id='own',root=str(root),source='science5e',source_commit='5e',config_sha256='cfg',execution_source='cachedExec',execution_migration_receipt_sha256='cacheproof')
    for tag,cwd in [('cached','cachedExec'),('pre_migration','science5e')]:
        job=tmp_path/'jobs/own'/tag;command=['model','-m','tools.qwen_world_performance.train','--output',str(root)]
        node=dict(job=str(job),host='oldhost')
        if cwd!='science5e':node['execution_source']=cwd
        running=dict(status='RUNNING',host='oldhost',worker_pid=10,worker_start_ticks=20,command=command,cwd=cwd)
        dump(job/'RUNNING.json',running);dump(job/'DISPATCH.json',running)
        intent=C.make_migration_intent(run,node,'training-vlawm-tsinghua-zt5')
        dump(job/'OWN_NATIVE_CONTINUATION.json',intent);(job/'STOP_REQUESTED').touch()
        result=dict(running,returncode=0,end_unix=time.time()+1)
        assert intent['source']=='science5e'and intent['execution_source']==cwd
        assert C.verified_migration(run,node,result,cp)==intent
        assert C.verified_migration(run,node,dict(result,cwd='unreviewed'),cp)is None
    source=inspect.getsource(job_worker.task_ranks)
    assert "'tools.qwen_world_multiscene.train'"in source
    assert 'd44_native_rl'not in source
