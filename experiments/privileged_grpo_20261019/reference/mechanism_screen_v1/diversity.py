import numpy as np


def summarize(trajectories, metrics):
    a=np.asarray(trajectories,dtype=np.float64)
    assert a.shape==(8,8,3) and np.isfinite(a).all()
    i,j=np.triu_indices(8,1)
    xy=np.linalg.norm(a[i,:,:2]-a[j,:,:2],axis=-1)
    yaw=a[i,-1,2]-a[j,-1,2];yaw=np.abs(np.arctan2(np.sin(yaw),np.cos(yaw)))*180/np.pi
    reward=np.asarray([m['PDMS'] for m in metrics])
    passed=[m['PDMS']>=.8 and all(m[k]==1 for k in ['NC','DAC','TTC']) for m in metrics]
    return dict(pair_count=len(i),pairwise_ADE_m=float(xy.mean()),pairwise_FDE_m=float(xy[:,-1].mean()),
                endpoint_std_x_m=float(a[:,-1,0].std()),endpoint_std_y_m=float(a[:,-1,1].std()),
                pairwise_final_yaw_deg=float(yaw.mean()),pdms_mean=float(reward.mean()),pdms_std=float(reward.std()),
                pdms_min=float(reward.min()),pdms_max=float(reward.max()),pdms_first=float(reward[0]),
                rule_pass_fraction=float(np.mean(passed)),zero_reward_variance=bool(reward.std()<1e-6),
                all_zero=bool(np.all(reward==0)),all_one=bool(np.all(reward==1)),
                mean_metrics={k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]})
