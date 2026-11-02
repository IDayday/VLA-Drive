"""Native journal clock is identity-derived; M2 is not guessed from rows."""
import copy,json
import pytest
from starVLA.rl.simwam_recipe import invocation as I


def row(u,m,run='own'):
 c=(u-1)//m+1
 return dict(updates=u,collection=c,inner_completed=(u-1)%m+1,behavior=dict(update=(c-1)*m,run=run))
def identity(m):return dict(sha256='own',recipe=dict(research=dict(performance=dict(inner_updates=m))))
def state(u,m):return dict(updates=u,collections=(u+m-1)//m,scene_cursor=((u+m-1)//m)*128)


def test_M2_resume4_to6_and_pending5_preserves_strict_row_run_guards(tmp_path,monkeypatch):
 monkeypatch.setattr(I,'gather',lambda x:[x]);root=tmp_path/'own';root.mkdir();journal=root/'metrics_rank0.jsonl'
 journal.write_text(''.join(json.dumps(row(u,2))+'\n'for u in range(1,7)))
 # Native resume at4 preserves an uncommitted5..6 suffix as before; it is not
 # called valid committed history. All actual payloads remain the native CP's.
 I.Invocation(root,0,state(4,2),identity(2))
 assert [json.loads(x)['updates']for x in journal.read_text().splitlines()]==[1,2,3,4]
 journal.write_text(''.join(json.dumps(row(u,2))+'\n'for u in range(1,7)))
 I.Invocation(root,0,state(6,2),identity(2))
 I.validate_rows([row(u,2)for u in range(1,6)],5,inner_updates=2) # pendinginner1
 for mutation in('gap','duplicate','behavior','inner','collection','run'):
  rows=[row(u,2)for u in range(1,7)]
  if mutation=='gap':rows.pop(2)
  elif mutation=='duplicate':rows[2]=copy.deepcopy(rows[1])
  elif mutation=='behavior':rows[2]['behavior']['update']=0
  elif mutation=='inner':rows[2]['inner_completed']=3
  elif mutation=='collection':rows[2]['collection']=1
  else:rows[2]['behavior']['run']='foreign'
  journal.write_text(''.join(json.dumps(x)+'\n'for x in rows))
  with pytest.raises(ValueError):I.Invocation(root,0,state(6,2),identity(2))


def test_legacy_M4_remains_default_and_performance_clock_is_never_guessed():
 assert I.journal_inner_updates(dict(sha256='legacy'))==4
 assert I.journal_inner_updates(identity(2))==2 and I.journal_inner_updates(identity(4))==4
 I.validate_rows([row(u,4)for u in range(1,9)],8)
 with pytest.raises(ValueError,match='ancestry'):I.validate_rows([row(u,2)for u in range(1,7)],6)
 for value in(None,{},dict(inner_updates=3),dict(inner_updates=True),dict(inner_updates='2')):
  bad=dict(sha256='own',recipe=dict(research=dict(performance=value)))
  with pytest.raises(ValueError,match='Sealed performance'):I.journal_inner_updates(bad)
