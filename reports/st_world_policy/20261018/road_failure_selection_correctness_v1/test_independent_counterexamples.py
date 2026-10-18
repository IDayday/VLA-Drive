"""Independent negative checks; all fixtures are explicitly synthetic."""
import copy,json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'tests/st_world')]
from tools.st_world_policy import road_failure_selection as selection
from tools.st_world_policy import run_road_direction as runner
from test_road_failure_selection import bound_fixture,put


def test_frame_both_order_and_population_are_frozen(tmp_path):
 evidence,_,frame=bound_fixture(tmp_path)
 for mutate in ('order','population'):
  changed=copy.deepcopy(evidence)
  if mutate=='order': changed['selected'].reverse()
  else: changed['selected'].pop(); changed['selection_count']=1
  changed['selection_sha256_before_proxy']=selection.canonical_hash(changed['selected'])
  with pytest.raises(ValueError): selection.validate_selection(changed,frame)


def test_sibling_native_t0_and_context_cannot_be_handwaved(tmp_path):
 evidence,_,frame=bound_fixture(tmp_path)
 for mutate in ('t0','reference'):
  changed=copy.deepcopy(evidence)
  if mutate=='t0': changed['selected'][0]['safe_sibling']['native_t0']=True
  else: changed['selected'][0]['safe_sibling']['reference_input_sha256']='0'*64
  changed['selection_sha256_before_proxy']=selection.canonical_hash(changed['selected'])
  with pytest.raises(ValueError): selection.validate_selection(changed,frame)


def test_summary_uses_all_requested_rows_and_reports_zero_when_no_valid_direction(tmp_path):
 evidence,_,_=bound_fixture(tmp_path)
 root=tmp_path/'out'
 for item,status in zip(evidence['selected'],('UNDEFINED_DIRECTION','FAILED')):
  put(root/'scenes'/item['scene_id']/'SCENE_RESULT.json',{'scene_id':item['scene_id'],
      'status':status,'original_metrics':item['failed']['metrics']})
 result=runner.summarize(root,evidence,{})
 assert result['requested_scenes']==result['road_repair_denominator']==2
 assert result['complete_scenes']==0 and result['unresolved_or_undefined_scenes']==2
 for repair in result['DAC_repair'].values():
  assert repair['requested_original_failure_denominator']==2
  assert repair['common_valid_original_failure_denominator']==0
  assert repair['repair_rate_valid_subset'] is None
  assert repair['known_repairs_over_requested_lower_bound']==0.
 assert result['status']=='PARTIAL_OR_UNDEFINED'


def test_uncertain_charged_attempt_cannot_be_replayed(tmp_path):
 root=tmp_path/'out';root.mkdir()
 item={'scene_id':'synthetic','candidate_id':'origin'}
 budget=runner.Budget(root)
 identity={'candidate_id':'origin:road_direction_v1:negative'}
 runner.charge_official_call(root,item,identity,budget,runner.FAILURE_LIMITS,failure_mode=True)
 # No score return exists: the durable attempt still consumes the candidate.
 assert not (root/'scenes').exists()
 with pytest.raises(runner.LimitError,match='NO_RETRY'):
  runner.charge_official_call(root,item,identity,budget,runner.FAILURE_LIMITS,failure_mode=True)
 assert len(list((root/'calls').glob('*.json')))==1


def test_actual_registration_code_constants_remain_exact():
 registration=Path('/mnt/project/st-world-policy-20261016/audits/consequence/road_failure_direction_v1/REGISTRATION.json')
 reg=json.loads(registration.read_bytes())
 assert runner.sha(registration.read_bytes())=='c9c76ffd513fcf3cf59e1753358bd552b0635ae6394399e509d15c09dd9cda0a'
 assert reg['source_hashes']==runner.source_hashes()
 assert reg['limits']==runner.FAILURE_LIMITS
 assert reg['decode_gate']==runner.DECODE_GATE
 assert runner.execution_limits(reg)['max_official_calls']==96
