"""Independent synthetic controls only; never proxy or official scoring."""
import json
import multiprocessing
from pathlib import Path
import time
import pytest
from tools.st_world_policy import run_road_direction as r


def sleeper(*args):
    time.sleep(60)


@pytest.mark.parametrize('mode',['aggregate_rss','output'])
def test_parent_caps_clean_up_owned_worker(tmp_path, monkeypatch, mode):
    context=multiprocessing.get_context('spawn'); owned=[]
    class Context:
        def Process(self,*args,**kwargs):
            child=context.Process(*args,**kwargs);owned.append(child);return child
    evidence={'selected':[{'scene_id':f'fixture{i}'} for i in range(32)],'selection_sha256_before_scores':'fixture'}
    ep=tmp_path/'evidence.json';ep.write_text(json.dumps(evidence));rp=tmp_path/'reg.json';rp.write_text('{}')
    reg={'output_root':str(tmp_path/'out'),'source_hashes':{'fixture':'only'},'evidence':{'path':str(ep),'sha256':r.sha(ep.read_bytes())}}
    monkeypatch.setattr(r,'check_admission',lambda *args:(reg,{'attempt_id':mode}))
    monkeypatch.setattr(r,'_child_entry',sleeper)
    monkeypatch.setattr(multiprocessing,'get_context',lambda kind:Context())
    real_read=Path.read_text
    if mode=='aggregate_rss':
        def read(path,*args,**kwargs):
            if str(path).startswith('/proc/') and path.name=='status':return 'VmRSS:\t9000000 kB\n'
            return real_read(path,*args,**kwargs)
        monkeypatch.setattr(Path,'read_text',read)
    else:monkeypatch.setattr(r,'owned_output_bytes',lambda root:r.LIMITS['output_bytes'])
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):monkeypatch.setenv(key,'1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    assert r.supervise(rp,tmp_path/'unused','fixture',False)==2
    assert owned and not any(child.is_alive() for child in owned)
    result=json.loads((tmp_path/'out'/f'RESULT_{mode}.json').read_text())
    assert result['supervisor']['owned_worker_exit_confirmed']
    assert result['supervisor']['stop_reason']==('AGGREGATE_RSS_CAP' if mode=='aggregate_rss' else 'OUTPUT_CAP')
    assert result['requested_scenes']==32


def test_charged_failed_call_survives_new_budget_no_retry(tmp_path):
    root=tmp_path/'out';root.mkdir();item={'scene_id':'fixture','candidate_id':'original'}
    identity={'candidate_id':'original:road_direction_v1:negative'}
    r.charge_official_call(root,item,identity,r.Budget(root),r.FAILURE_LIMITS,failure_mode=True)
    # A hypothetical backend failure cannot refund this durable receipt.
    assert json.loads((root/'calls'/'000.json').read_text())['charged_before_call']
    with pytest.raises(r.LimitError,match='NO_RETRY'):
        r.charge_official_call(root,item,identity,r.Budget(root),r.FAILURE_LIMITS,failure_mode=True)
    assert len(list((root/'calls').glob('*.json')))==1

