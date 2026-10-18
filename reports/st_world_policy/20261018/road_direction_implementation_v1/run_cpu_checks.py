"""CPU fixture harness with a real main path for multiprocessing spawn."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
if __name__=='__main__':
    import json,resource,time,pytest,torch
    started=time.monotonic()
    code=pytest.main(['-q','tests/st_world/test_run_road_direction.py','tests/st_world/test_road_margin_proxy.py'])
    print(json.dumps({'exit_code':code,'wall_seconds':time.monotonic()-started,
        'peak_parent_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'peak_child_rss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        'cuda_initialized':torch.cuda.is_initialized(),'real_official_calls':0,'real_asset_reads':0}))
    raise SystemExit(code)
