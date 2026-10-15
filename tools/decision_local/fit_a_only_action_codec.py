"""Fit the original affine action representation using A training GT only."""
import argparse,json,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import read,atomic_json,identity,digest,check_host,source_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source-audit','index','teacher-data','output'):p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--workers',type=int,default=8);a=p.parse_args();check_host()
    if a.output.exists():raise FileExistsError('Fitted normalization is immutable')
    if not 1<=a.workers<=16:raise ValueError('Bounded CPU threads required')
    torch.set_num_threads(1);start=time.monotonic();cpu=time.process_time()
    audit,index=read(a.source_audit),read(a.index);source=read(a.teacher_data/'identity.json')
    if digest(a.index)!=audit['index_sha256']['A_train']:raise ValueError('A-only fit population changed')
    if any(audit['roles'][r['log'].rsplit('_',2)[0]]!='A_train' for r in index):raise ValueError('Non-A fit scene')
    def one(row):
        item=torch.load(a.teacher_data/'records'/(row['token']+'.pt'),weights_only=True,map_location='cpu')
        if item['identity']!=source['identity'] or item['token']!=row['token']:raise ValueError('Foreign GT record')
        r=item['record'];xy=r['future'][0].double().numpy();mask=r['point_valid'][0].numpy()
        if xy.shape!=(8,2) or not mask.all() or not np.isfinite(xy).all():raise ValueError('Original complete ego GT contract changed')
        return xy.sum(0),(xy*xy).sum(0),len(xy)
    total=np.zeros(2);squares=np.zeros(2);count=0
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for begin in range(0,len(index),256):
            for x,x2,n in pool.map(one,index[begin:begin+256]):total+=x;squares+=x2;count+=n
    mean=total/count;std=np.sqrt(squares/count-mean*mean)
    result=dict(schema='a_only_action_affine_v1',fit_role='A_train',mean_xy_m=mean.tolist(),std_xy_m=std.tolist(),
        samples=count,scenes=len(index),logs=len({r['log'] for r in index}),index_identity=identity(index),index_file_sha256=digest(a.index),
        source_audit_identity=audit['identity'],raw_GT_identity=source['identity'],source_sha=source_sha(ROOT),
        formula='population mean/std over eight physical ego future xy points; unchanged sin/cos heading',
        inherited_statistics_reused=False,CPU_process_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-start)
    result['identity']=identity(result);atomic_json(a.output,result);print(json.dumps(result))


if __name__=='__main__':main()
