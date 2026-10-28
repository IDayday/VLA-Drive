import ast
import json
from pathlib import Path


def test_remote_launch_explicitly_carries_new_run_namespace(tmp_path,monkeypatch):
    from tools.qwen_world_postcot import campaign as c
    monkeypatch.setattr(c,'ROOT',tmp_path)
    monkeypatch.setenv('WORLD_POSTCOT_ROOT',str(tmp_path))
    monkeypatch.setenv('WORLD_POSTCOT_TASK','optimized-task')
    observed={}
    def remote(host,args):
        tree=ast.parse(args[-1]);assign=next(n for n in ast.walk(tree) if isinstance(n,ast.keyword) and n.arg=='env')
        kwargs=next(k.value for k in assign.value.keywords if k.arg is None)
        observed.update(ast.literal_eval(kwargs));return json.dumps(dict(worker_pid=123,worker_start_ticks=456))
    monkeypatch.setattr(c,'remote',remote)
    c.launch('remote-host','/new/immutable/source',['python','-m','train'],'one',gpus='0')
    assert observed==dict(WORLD_POSTCOT_ROOT=str(tmp_path),WORLD_POSTCOT_TASK='optimized-task',PYTHONPATH='/new/immutable/source')
    assert json.loads((tmp_path/'jobs/one/DISPATCH.json').read_text())['worker_pid']==123


def test_optimized_recipe_only_changes_its_refresh_namespace():
    import yaml
    base=Path(__file__).resolve().parents[1]/'configs/qwen_world_postcot'
    original=yaml.safe_load((base/'il.yaml').read_text());optimized=yaml.safe_load((base/'optimized_labels_il.yaml').read_text())
    assert optimized['candidate_refresh']['registry']!=original['candidate_refresh']['registry']
    optimized['candidate_refresh']['registry']=original['candidate_refresh']['registry']
    assert optimized==original
