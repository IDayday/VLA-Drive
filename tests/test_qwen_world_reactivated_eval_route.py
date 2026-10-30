from tools.qwen_world_research import dispatch


def test_explicitly_reauthorized_tsinghua2_routes_only_when_allowed(monkeypatch,tmp_path):
    root=tmp_path; (root/'resources').mkdir()
    host='training-vlawm-tsinghua-zt2'
    monkeypatch.setattr(dispatch,'allowed_hosts',lambda:[host])
    inventory=dict(gpus=[dict(index=i,uuid=f'gpu{i}',free_MiB=80000,total_MiB=80000,utilization=0)
                        for i in range(8)],owners=[])
    monkeypatch.setattr(dispatch,'probe',lambda h:inventory)
    routes,_=dispatch._eval_routes(root,maximum=8)
    assert len(routes)==8 and all(r['host']==host for r in routes)
    monkeypatch.setattr(dispatch,'allowed_hosts',lambda:[])
    assert dispatch._eval_routes(root,maximum=8)[0]==[]
