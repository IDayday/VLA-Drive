"""Closed same-run M2 journal guard repair. No RUNS/owner/GPU mutation.

Only the inspected invocation.py guard may differ. The sealed404 adapters,
Adam/scheduler/all8 RNG and pending/stream buffers are restored by the unchanged
native Trainer; this adapter never constructs a new optimizer or policy.
"""
import argparse,copy,json,os,subprocess
from pathlib import Path
import yaml
from .launch import ROOT,read,sha
from starVLA.rl.simwam_recipe.lineage import scientific_identity,verify_migration
from starVLA.rl.qwen_world_research_data.manifest import fingerprint

SCHEMA='world_performance_M2_journal_guard_execution_v1'
RECEIPT='world_M2_journal_guard_execution_migration_v1'
PROTOCOL=dict(schema=SCHEMA,identity_derived_M2_journal_guard=True)
ARMS={'S2_CONSERVATIVE_AWFM','S3_LARGE_FPO'}
GUARD='starVLA/rl/simwam_recipe/invocation.py'
GUARD_SHA='734c2be8d96c73a8ead45a7cdc3e76c32d407e810efc95a7061bb2da536647a1'
EVIDENCE='world_performance_M2_journal_guard_CPU_evidence_v1'


def seal(identity):
 body=dict(identity);saved=body.pop('sha256',None)
 if not saved or saved!=fingerprint(body):raise ValueError('M2 migration identity/evidence seal changed')
 return identity


def parent_material(run):
 if run['id']not in ARMS or run['trainer_module']!='tools.qwen_world_performance.train':raise ValueError('Journal-only repair is restricted to exact S2/S3 M2 runs')
 root=Path(run['root']).resolve();parent=Path(run['execution_parent_checkpoint']).resolve()
 if root!=ROOT.resolve()/'runs'/run['id']or parent!=root/'checkpoints/update000404_inner0':raise ValueError('Only own exact sealed404 native parent')
 marker=read(parent/'COMPLETE',{});files=read(parent/'files.json',{});old=seal(read(root/'identity.json'));tx=read(parent/'transaction.json',{})
 required={'optimizer.pt','identity.json','transaction.json'}|{f'rank{r}.pt'for r in range(8)}
 if(marker.get('schema')!=2 or marker.get('world')!=8 or marker.get('updates')!=404 or marker.get('next_inner')!=0 or marker.get('phase')!='READY' or marker.get('files_sha256')!=sha(parent/'files.json')):raise ValueError('M2 repair parent is not exact native404/world8/READY')
 if not required<=set(files)or any(Path(n).is_absolute()or not(parent/n).resolve().is_relative_to(parent)or not(parent/n).is_file()or(parent/n).stat().st_size<=0 for n in files):raise ValueError('M2 parent inventory incomplete/escaping')
 if(read(parent/'identity.json')!=old or old.get('purpose')!='formal'or old.get('world')!=8
    or old['recipe']['research']['arm']!=run['id']or old['recipe']['research']['performance']['inner_updates']!=2
    or sha(run['config'])!=run['config_sha256']or yaml.safe_load(Path(run['config']).read_text())!=old['recipe']):raise ValueError('M2 same-run sealed recipe/identity changed')
 for n in('identity.json','transaction.json'):
  if sha(parent/n)!=files[n]:raise ValueError('M2 parent metadata changed')
 rows=tx.get('ranks',[])
 if len(rows)!=8 or [r.get('rank')for r in rows]!=list(range(8))or any(r.get('run')!=old['sha256']or r.get('updates')!=404 or r.get('next_inner')!=0 or r.get('phase')!='READY'for r in rows):raise ValueError('M2 parent eight-rank transaction differs')
 return parent,old


def check_source_and_identity(run,old,new):
 seal(new);source=Path(run['execution_source']).resolve()
 if scientific_identity(old)!=scientific_identity(new):raise ValueError('Journal repair changed scientific identity')
 allowed={'source_files','execution_identity','sha256'}
 if any(old.get(k)!=new.get(k)for k in(set(old)|set(new))-allowed):raise ValueError('Journal repair changed non-source state or cache/algorithm protocol')
 before=old['source_files'];after=new['source_files']
 if set(before)!=set(after)or{k for k in before if before[k]!=after[k]}!={GUARD}or after[GUARD]!=GUARD_SHA:raise ValueError('Journal repair may change only the reviewed exact invocation.py')
 commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
 if commit!=run['execution_source_commit']:raise ValueError('Journal execution source commit changed')
 oldexec=copy.deepcopy(old['execution_identity']['fields']);newexec=new['execution_identity']
 expected=dict(oldexec,module_root=str(source),source_files=after)
 if newexec.get('schema')!=old['execution_identity']['schema']or newexec.get('fields')!=expected or newexec.get('sha256')!=fingerprint(expected):raise ValueError('Journal execution identity/DDP/backend changed')
 for rel,digest in after.items():
  path=(source/rel).resolve()
  if Path(rel).is_absolute()or not path.is_relative_to(source)or sha(path)!=digest:raise ValueError('Journal source payload changed: '+rel)
 return source


def check_evidence(receipt,run,parent,old,new):
 matched=False
 for item in receipt['evidence']:
  if sha(item['path'])!=item['sha256']:raise ValueError('Journal repair evidence file changed')
  evidence=read(item['path'])
  if evidence.get('status')!='PASS':raise ValueError('Journal repair evidence is not PASS')
  if evidence.get('schema')!=EVIDENCE:continue
  seal(evidence)
  if evidence.get('source_file_sha256')!=GUARD_SHA or evidence.get('actual_prefix_rows')!=6464:raise ValueError('Journal repair lacks full inspected prefixes/source proof')
  own=next((r for r in evidence.get('actual_runs',[])if r.get('arm')==run['id']),None)
  if not own or Path(own['checkpoint']).resolve()!=parent or own['checkpoint_files_sha256']!=sha(parent/'files.json')or own['identity_sha256']!=old['sha256']or own['inner_updates']!=2:raise ValueError('Journal repair evidence is bound to another run/parent')
  rows=own['rank_prefixes']
  if len(rows)!=8 or [r.get('rank')for r in rows]!=list(range(8))or any(r.get('rows')!=404 or r.get('updates')!=[1,404]or r.get('legacy_M4_reproduces_error')is not True for r in rows):raise ValueError('Journal repair requires all8 real404 journal prefixes')
  delta=evidence['source_inventory_deltas'][run['id']]
  if delta!={GUARD:dict(old=old['source_files'][GUARD],new=new['source_files'][GUARD])}:raise ValueError('Journal source evidence delta differs')
  matched=True
 if not matched:raise ValueError('Missing exact journal-only PASS evidence')


def validate_journal_execution(run):
 if run.get('execution_optimizations')!=PROTOCOL:raise ValueError('Only typed journal guard repair; no cache flag/other optimization')
 path=Path(run['execution_migration_receipt']).resolve()
 if sha(path)!=run['execution_migration_receipt_sha256']:raise ValueError('Journal migration receipt changed')
 receipt=seal(read(path));parent,old=parent_material(run);new=receipt['new_identity']
 if(receipt.get('schema')!=RECEIPT or receipt.get('arm')!=run['id']or receipt.get('run_root')!=str(Path(run['root']).resolve())
    or receipt.get('weights_only')is not False or receipt.get('optimizer_rng_buffers_inherited')is not True
    or receipt.get('old_identity')!=old or receipt.get('parent_files_sha256')!=sha(parent/'files.json')):raise ValueError('Journal migration must retain exact same logical run/native404 state')
 source=check_source_and_identity(run,old,new);check_evidence(receipt,run,parent,old,new)
 verify_migration(path,old,new,parent)
 return dict(kind=SCHEMA,source=str(source),receipt=str(path),old_identity=old,new_identity=new)


def derive_source_only_identity(run,source):
 """Mechanical original Trainer identity suffix; actor closure is inherited.

    Valid only because all actual runtime source bytes except this journal
    guard are equal. This is not a model load, new training or GPU qualification.
    """
 parent,old=parent_material(run);source=Path(source).resolve();new=copy.deepcopy(old);new.pop('sha256')
 new['source_files']={rel:sha(source/rel)for rel in old['source_files']}
 fields=dict(old['execution_identity']['fields'],source_files=new['source_files'],module_root=str(source))
 new['execution_identity']=dict(schema=old['execution_identity']['schema'],fields=fields,sha256=fingerprint(fields))
 new['sha256']=fingerprint(new);check_source_and_identity(dict(run,execution_source=str(source)),old,new)
 return new


def prepare(run,source,evidence,output,*,new_identity=None):
 """Write a new immutable proposal directory only. Never apply to live RUNS."""
 source=Path(source).resolve();output=Path(output).resolve()
 if output.exists():raise FileExistsError('Keep all prior migration evidence; choose a new output')
 if any(k.startswith('execution_')for k in run if k!='execution_output_root'):raise ValueError('Do not combine a prior cache/source migration with this journal-only repair')
 proposed=copy.deepcopy(run);proposed.update(execution_source=str(source),execution_source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),execution_optimizations=copy.deepcopy(PROTOCOL),execution_parent_checkpoint=str(Path(run['root']).resolve()/'checkpoints/update000404_inner0'))
 parent,old=parent_material(proposed);new=read(new_identity)if new_identity else derive_source_only_identity(proposed,source)
 check_source_and_identity(proposed,old,new)
 body=dict(schema=RECEIPT,arm=run['id'],run_root=str(Path(run['root']).resolve()),old_identity=old,new_identity=new,parent_files_sha256=sha(parent/'files.json'),evidence=[dict(path=str(Path(evidence).resolve()),sha256=sha(evidence))],weights_only=False,optimizer_rng_buffers_inherited=True)
 check_evidence(body,proposed,parent,old,new);receipt=dict(body,sha256=fingerprint(body))
 output.mkdir(parents=True);rp=output/'MIGRATION_RECEIPT.json';rp.write_text(json.dumps(receipt,indent=2)+'\n')
 proposed.update(execution_migration_receipt=str(rp),execution_migration_receipt_sha256=sha(rp));validate_journal_execution(proposed)
 fields={k:v for k,v in proposed.items()if k.startswith('execution_')and k!='execution_output_root'}
 result=dict(schema='world_M2_journal_guard_recovery_proposal_v1',status='PREPARED_NOT_APPLIED',arm=run['id'],resume=str(parent),execution_fields=fields,command_suffix=['--resume',str(parent),'--migration-receipt',str(rp)],same_run_root=run['root'],old_identity_sha256=old['sha256'],new_execution_identity_sha256=new['execution_identity']['sha256'],new_initialization=False,cache_enabled=False,failed_attempts_retained=True,metadata_derivation='exact source-only Trainer identity suffix, no model load'if new_identity is None else'provided complete new identity checked against sole source delta')
 (output/'RECOVERY_CONFIGURATION.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
 ap=argparse.ArgumentParser(__doc__);ap.add_argument('action',choices=['prepare']);ap.add_argument('--arm',choices=sorted(ARMS),required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--evidence',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--new-identity',type=Path);a=ap.parse_args()
 run=next(r for r in read(ROOT/'RUNS.json')['runs']if r['id']==a.arm)
 print(json.dumps(prepare(run,a.source,a.evidence,a.output,new_identity=a.new_identity),indent=2))
if __name__=='__main__':main()
