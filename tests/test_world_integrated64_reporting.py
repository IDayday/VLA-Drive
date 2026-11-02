"""NumPy official-score gates must survive the actual journal JSON boundary."""
import copy,io,json
import numpy as np
import pytest,torch
from test_world_safety_exploration_groups import components,records
from starVLA.rl.qwen_world_safety_exploration.integrated64_runtime import IntegratedRuntime


def test_numpy_history_flags_actual_journal_row_json_without_material_mutation():
 for enabled_count in (0,3,8):
  samples=[]
  for b in range(8):
   recs=records(components(16)[b])
   for rec in recs:rec['metrics']={k:np.float64(v)for k,v in rec['metrics'].items()}
   samples.append(dict(records=recs,advantages=torch.linspace(-1,1,16),adv_mask=torch.ones(16),rewards=torch.tensor([[r['metrics']['score']for r in recs]]),global_std=torch.tensor(.2)))
  events=[dict(enabled=np.bool_(i<enabled_count),history_scored_now=np.int64(i%2),fresh_witness={},teacher={})for i in range(8)]
  window=dict(samples=samples,tokens=[f't{i}'for i in range(8)],logs=['train_log']*8,normalization_std=torch.tensor(.3),integrated_retention=dict(events=events))
  before=copy.deepcopy(events)
  #Exact productionfailure: sum ofNumPy booleans returnsint64 even ifallfalse.
  with pytest.raises(TypeError,match='int64 is not JSON serializable'):json.dumps(dict(history_gates=sum(e['enabled']for e in events)),allow_nan=False)
  summary=IntegratedRuntime().exploration_summary(window)
  assert type(summary['history_gates'])is int and summary['history_gates']==enabled_count
  assert type(summary['history_B1_queries'])is int and summary['history_B1_queries']==4
  #Same journal construction/serialization path astrain.py, including scalar
  #metric conversion, actual nested per-scene exploration and allow_nan=False.
  metrics=dict(retention_enabled=torch.tensor(enabled_count/8.))
  row=dict(updates=257,collection=65,inner_completed=1,metrics={k:float(v)for k,v in metrics.items()},exploration=summary)
  journal=io.StringIO();journal.write(json.dumps(row,allow_nan=False)+'\n');restored=json.loads(journal.getvalue())
  assert restored['exploration']['history_gates']==enabled_count
  assert events==before and all(type(e['enabled'])is np.bool_ and type(e['history_scored_now'])is np.int64 for e in events)
