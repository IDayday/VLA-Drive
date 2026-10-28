"""TRAIN-only GT horizon dispersion, current-state residuals and noise geometry.

Population dispersion is not conditional uncertainty given the full observation.
No loss weight is fitted here and no actor/evaluator/optimizer is executed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(16*1024**2),b''):h.update(b)
    return h.hexdigest()


def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def summarize(values,valid):
    if values.ndim!=3 or values.shape[1:]!=(8,3) or valid.shape!=values.shape[:2]:
        raise ValueError('Expected aligned N,8,3 and N,8 validity')
    result=[]
    for k in range(8):
        selected=values[valid[:,k],k];count=len(selected)
        if not count:result.append(dict(t_seconds=(k+1)*.5,valid_count=0));continue
        if not np.isfinite(selected).all():raise ValueError('Nonfinite valid GT')
        median=np.median(selected,axis=0);circular=np.mean(np.exp(1j*selected[:,2]));concentration=abs(circular)
        result.append(dict(t_seconds=(k+1)*.5,valid_count=count,
            mean=selected.mean(0).tolist(),std_population=selected.std(0,ddof=0).tolist(),median=median.tolist(),
            MAD=(np.median(np.abs(selected-median),axis=0)).tolist(),
            p05=np.quantile(selected,.05,axis=0).tolist(),p95=np.quantile(selected,.95,axis=0).tolist(),
            xy_covariance=np.cov(selected[:,:2],rowvar=False,ddof=0).tolist(),
            circular_heading_mean_rad=float(np.angle(circular)),
            circular_heading_std_rad=None if concentration<=0 else float(np.sqrt(max(0,-2*np.log(min(1.,concentration)))))))
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--data',default='/mnt/project/qwen-world-postcot-20261025/data/TRAIN');p.add_argument('--output',required=True)
    a=p.parse_args();started=time.monotonic();root=Path(a.data);output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((root/'manifest.json').read_text());seal=json.loads((root/'COMPLETE').read_text())
    if manifest['cohort']!='TRAIN' or sha(root/'manifest.json')!=seal['manifest_sha256'] or sha(root/'records.pkl')!=manifest['records_sha256']:
        raise ValueError('TRAIN-only sealed data required')
    partition=json.loads((root.parent/'DATA_PARTITION.json').read_text())
    tokens=manifest['tokens'];records=pickle.load(open(root/'records.pkl','rb'))
    if set(records)!=set(tokens) or set(tokens)&set(partition['dev_tokens']+partition['navtest_tokens']):
        raise ValueError('Split membership mismatch/leakage')
    # The preparation script embargoes complete physical DEV/Test logs.
    logs={records[t]['log'].rsplit('_',2)[0] for t in tokens}
    if logs!=set(partition['train_physical_logs']) or logs&set(partition['dev_physical_logs']+partition['navtest_physical_logs']):
        raise ValueError('Physical log leakage')
    gt=np.asarray([records[t]['requested_GT'] for t in tokens],dtype=np.float64)
    ego=np.asarray([records[t]['proprio'] for t in tokens],dtype=np.float64)
    dt=np.asarray([records[t]['raw_future_dt_seconds'] for t in tokens],dtype=np.float64)
    if gt.shape!=(len(tokens),8,3) or ego.shape!=(len(tokens),8):raise ValueError('Wrong native GT/ego shape')
    future_time=np.arange(1,9,dtype=np.float64)*.5
    valid=np.isfinite(gt).all(-1)&(np.abs(np.cumsum(dt,axis=1)-future_time)<.06)
    cv=np.zeros_like(gt);cv[:,:,:2]=ego[:,None,:2]*future_time[None,:,None]
    ca=cv.copy();ca[:,:,:2]+=.5*ego[:,None,2:4]*future_time[None,:,None]**2
    # Headings are relative to current ego, so the naive constant-heading
    # baseline is zero. No future heading or yaw rate is fed to either baseline.
    speed=np.linalg.norm(ego[:,:2],axis=1)
    navigation=np.array([records[t]['navigation'] for t in tokens])
    groups={'all':np.ones(len(tokens),dtype=bool),'current_speed_lt0.5':speed<.5,
        'current_speed_0.5_to5':(speed>=.5)&(speed<5),
        'current_speed_5_to10':(speed>=5)&(speed<10),'current_speed_ge10':speed>=10}
    for name in sorted(set(navigation)):groups['navigation_'+str(name)]=navigation==name
    result=dict(schema='world_TRAIN_GT_horizon_distribution_v1',scenes=len(tokens),physical_logs=len(logs),
        cohort='TRAIN',units=['m','m','rad'],validity='Original training per-point timestamp mask and finite native 8x3 GT, no label clipping.',
        data_manifest_sha256=sha(root/'manifest.json'),records_sha256=manifest['records_sha256'],
        partition_sha256=sha(root.parent/'DATA_PARTITION.json'),physical_log_disjoint=True,
        code_sha256=sha(__file__),groups={},
        warning='Marginal dispersion and coarse strata/naive CV/CA residuals are not irreducible conditional uncertainty. No weights fitted; existing coordinate-only experiment and time weights unchanged.',
        current_state_contract='vx,vy m/s; ax,ay m/s2; four-way navigation. CV/CA use fixed current ego axes, not full vehicle dynamics.',
        model_forwards=0,scorer_calls=0,optimizer_updates=0)
    for name,mask in groups.items():
        result['groups'][name]=dict(scenes=int(mask.sum()),GT=summarize(gt[mask],valid[mask]),
            residual_from_current_CV=summarize((gt-cv)[mask],valid[mask]),
            residual_from_current_CA=summarize((gt-ca)[mask],valid[mask]))
    all_valid=valid.all(1);normalized=gt/np.array([33.37,21.,1.765])
    result['joint_complete_time_rows']=int(all_valid.sum())
    result['cross_time_coordinate_correlation']=np.corrcoef(normalized[all_valid].reshape(-1,24),rowvar=False).tolist()
    result['correlation_order']=[f't{t:.1f}_{axis}' for t in future_time for axis in ['x','y','heading_raw']]
    variance=np.asarray([v['std_population'] for v in result['groups']['all']['GT']])**2/np.array([33.37,21.,1.765])**2
    result['centered_marginal_variance_SNR_diagnostic']={str(s):(((1-s)/s)**2*variance).tolist() for s in [.2,.5,.8]}
    result['SNR_warning']='Centered population target variance, not full conditional signal/information. Low GT variance can mean easier mean prediction. Flow sigma is distinct from future physical time.'
    result['wall_seconds']=time.monotonic()-started
    result['sha256']=fingerprint(result);(output/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (output/'all_horizons.csv').open('w') as f:
        keys=['time_s','valid_count','GT_std_x_m','GT_std_y_m','GT_std_heading_deg','CV_residual_std_x_m','CV_residual_std_y_m','CA_residual_std_x_m','CA_residual_std_y_m']
        writer=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');writer.writeheader()
        values=result['groups']['all']
        for g,c,v in zip(values['GT'],values['residual_from_current_CV'],values['residual_from_current_CA']):
            writer.writerow(dict(zip(keys,[g['t_seconds'],g['valid_count'],*g['std_population'][:2],np.rad2deg(g['std_population'][2]),*c['std_population'][:2],*v['std_population'][:2]])))
    (output/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',analysis_sha256=sha(output/'analysis.json')))+'\n')
    print(json.dumps(dict(status='COMPLETE',scenes=len(tokens),groups={k:v['scenes'] for k,v in result['groups'].items()},wall_seconds=result['wall_seconds'])))


if __name__=='__main__':main()
