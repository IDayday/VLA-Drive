"""Closed M2 guard-only migration metadata; no policy/optimizer/GPU loads."""
import copy,json
from pathlib import Path
import pytest,yaml
from tools.qwen_world_performance import journal_migration as J,execution as X
from starVLA.rl.simwam_recipe.lineage import SCIENTIFIC_KEYS
from starVLA.rl.qwen_world_research_data.manifest import fingerprint


def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))
def seal(x):return dict(x,sha256=fingerprint(x))
def fixture(tmp,monkeypatch):
 root=tmp/'P';rr=root/'runs/S3_LARGE_FPO';cp=rr/'checkpoints/update000404_inner0';cp.mkdir(parents=True);source=tmp/'fixedscience';guard=source/J.GUARD;guard.parent.mkdir(parents=True);guard.write_text('fixed guard CPU fixture');(source/'actor.py').write_text('identical actor')
 monkeypatch.setattr(J,'ROOT',root);monkeypatch.setattr(J,'GUARD_SHA',J.sha(guard));monkeypatch.setattr(J.subprocess,'check_output',lambda *a,**kw:'b'*40+'\n')
 cfg=dict(research=dict(arm='S3_LARGE_FPO',performance=dict(inner_updates=2)))
 config=root/'config.yaml';config.write_text(yaml.safe_dump(cfg));before={J.GUARD:'original_guard_sha','actor.py':J.sha(source/'actor.py')}
 fields=dict(source_files=before,module_root='originalfrozen',DDP='unchanged',checkpoint_schema=2)
 old={k:None for k in SCIENTIFIC_KEYS};old.update(recipe=cfg,purpose='formal',world=8,source_files=before,execution_identity=seal(dict(fields=fields,schema='simwam_execution_v2')))
 # execution identity fingerprints fields, as the real native Trainer does.
 old['execution_identity']['sha256']=fingerprint(fields);old=seal(old);dump(rr/'identity.json',old);dump(cp/'identity.json',old)
 dump(cp/'transaction.json',dict(ranks=[dict(rank=r,updates=404,next_inner=0,phase='READY',run=old['sha256'])for r in range(8)]))
 for n in['optimizer.pt']+[f'rank{r}.pt'for r in range(8)]:(cp/n).write_text('opaqueCPUfixture;notloaded')
 dump(cp/'files.json',{p.name:J.sha(p)for p in cp.iterdir()});dump(cp/'COMPLETE',dict(schema=2,world=8,phase='READY',updates=404,next_inner=0,files_sha256=J.sha(cp/'files.json')))
 delta={J.GUARD:dict(old=before[J.GUARD],new=J.GUARD_SHA)}
 own=dict(arm='S3_LARGE_FPO',checkpoint=str(cp),checkpoint_files_sha256=J.sha(cp/'files.json'),identity_sha256=old['sha256'],inner_updates=2,rank_prefixes=[dict(rank=r,rows=404,updates=[1,404],legacy_M4_reproduces_error=True)for r in range(8)])
 evidence=root/'operations/CPU_fixture.json';dump(evidence,seal(dict(schema=J.EVIDENCE,status='PASS',source_file_sha256=J.GUARD_SHA,actual_prefix_rows=6464,actual_runs=[own],source_inventory_deltas={'S3_LARGE_FPO':delta})))
 run=dict(id='S3_LARGE_FPO',root=str(rr),source='unchangedscientific',source_commit='old',trainer_module='tools.qwen_world_performance.train',config=str(config),config_sha256=J.sha(config),disposition='must be preserved',execution_output_root=str(rr))
 return root,run,source,evidence,cp


def test_closed_journal_only_proposal_restores_own404_and_pending_lineage_without_cache(tmp_path,monkeypatch):
 root,run,source,evidence,cp=fixture(tmp_path,monkeypatch);old=copy.deepcopy(run)
 result=J.prepare(run,source,evidence,root/'operations/proposal');assert run==old
 proposed=dict(run,**result['execution_fields']);proof=X.validate_execution_fields(proposed)
 assert proof['kind']==J.SCHEMA and proof['old_identity']==J.read(cp/'identity.json')
 cwd,flags=X.command_execution(proposed,str(cp));assert cwd==str(source)and flags==['--migration-receipt',str(root/'operations/proposal/MIGRATION_RECEIPT.json')]
 assert '--frozen-reference-cache'not in flags and result['new_initialization']is False and result['cache_enabled']is False
 with pytest.raises(ValueError,match='strict native resume'):X.command_execution(proposed,None)
 later=Path(run['root'])/'checkpoints/update000405_inner1';later.mkdir();dump(later/'files.json',{'other':'nativepayload'});dump(later/'COMPLETE',{})
 with pytest.raises(ValueError,match='audited parent'):X.command_execution(proposed,str(later))
 dump(later/'transaction.json',dict(migration_receipt_sha256=J.read(proof['receipt'])['sha256']))
 assert X.command_execution(proposed,str(later))==(cwd,flags)
 assert proposed['disposition']==run['disposition']and result['failed_attempts_retained']


def test_rejects_other_algorithms_scientific_delta_source_tamper_and_unbound_evidence(tmp_path,monkeypatch):
 root,run,source,evidence,cp=fixture(tmp_path,monkeypatch);result=J.prepare(run,source,evidence,root/'operations/proposal');proposed=dict(run,**result['execution_fields'])
 with pytest.raises(ValueError,match='restricted'):J.parent_material(dict(proposed,id='S1_NOMINAL_RESCUE'))
 with pytest.raises(ValueError,match='Arbitrary'):X.validate_execution_fields(dict(proposed,execution_flags=['--newInit']))
 old=J.read(cp/'identity.json');new=J.read(proposed['execution_migration_receipt'])['new_identity'];bad=copy.deepcopy(new);bad.pop('sha256');bad['parameter_contract']={'changed':True};bad=seal(bad)
 with pytest.raises(ValueError,match='scientific'):J.check_source_and_identity(proposed,old,bad)
 actor=source/'actor.py';actor.write_text('changed actor')
 with pytest.raises(ValueError,match='source payload'):X.validate_execution_fields(proposed)
 actor.write_text('identical actor');e=J.read(evidence);e['status']='FAIL';dump(evidence,e)
 with pytest.raises(ValueError,match='evidence file'):X.validate_execution_fields(proposed)
