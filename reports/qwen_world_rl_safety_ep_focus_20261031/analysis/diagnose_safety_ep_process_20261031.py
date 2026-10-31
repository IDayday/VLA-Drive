"""Bounded, cache-first read-only SafetyEP process diagnosis; no new scores."""
from __future__ import annotations
import hashlib,json,math,os,re,sys,uuid
from collections import Counter
from pathlib import Path
from datetime import datetime,timezone
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import torch
A=Path(__file__).resolve().parent;sys.path.insert(0,str(A))
from safety_evidence_reviewer import complete_input,load_metrics,atomic
from rescue_pair_analyzer import contrast
R=A.parent;ARM='PPO_SAFETY_EP';OUT=A/'SAFETY_EP_PROCESS_DIAGNOSIS_20261031.json';CACHE=A/'SAFETY_EP_PROCESS_CACHE';CACHE.mkdir(exist_ok=True)
KEYS=('score','no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress','comfort','driving_direction_compliance')
EPS=1e-12
now=lambda:datetime.now(timezone.utc).isoformat()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
physical=lambda s:re.sub(r'_\d+_\d+$','',s)


def longitudinal(values,updates):
    scores=np.stack([v['values'][:,0] for v in values]);z=scores<=EPS;ever=z.any(0);persistent=z.all(0);relapse=np.zeros(z.shape[1],bool);recovered=np.zeros_like(relapse)
    for i in range(1,len(z)):
        relapse|=recovered & (~z[i-1]) & z[i];recovered|=z[i-1]&~z[i]
    result=dict(observed_updates=updates,scene_count=z.shape[1],zero_epsilon=EPS,all_observations_persistent=int(persistent.sum()),
        ever_zero=int(ever.sum()),intermittent=int((ever&~persistent).sum()),never_zero=int((~ever).sum()),
        recovered_at_least_once=int(recovered.sum()),relapsed_scenes=int(relapse.sum()),latest_zero=int(z[-1].sum()),
        new_zero_relative_first=int((~z[0]&z[-1]).sum()),recovered_relative_first=int((z[0]&~z[-1]).sum()),
        transitions=[dict(previous_updates=updates[i-1],updates=updates[i],previous_zero=int(z[i-1].sum()),zero=int(z[i].sum()),
            persistent_from_previous=int((z[i-1]&z[i]).sum()),new_zero=int((~z[i-1]&z[i]).sum()),repaired_zero=int((z[i-1]&~z[i]).sum())) for i in range(1,len(z))],
        interpretation='persistent only over explicitly observed deploymentODE checkpoints; missing evaluations are not zeros; unequal sequence lengths are not comparable')
    if 10000 in updates:
        start=updates.index(10000);result['post10k']=dict(observed_updates=updates[start:],persistent=int(z[start:].all(0).sum()),ever=int(z[start:].any(0).sum()),
            latest_new_zero_relative10k=int((~z[start]&z[-1]).sum()),latest_repaired_relative10k=int((z[start]&~z[-1]).sum()))
    logs=values[0]['logs'];result['by_physical_log']=[dict(log=str(l),scenes=int((logs==l).sum()),persistent=int((persistent&(logs==l)).sum()),ever=int((ever&(logs==l)).sum()),latest_zero=int((z[-1]&(logs==l)).sum())) for l in sorted(set(logs))]
    return result


def causes(v):
    m=v['values'];z=m[:,0]<=EPS;n=m[:,1]<=EPS;d=m[:,2]<=EPS;utility=(5*m[:,4]+5*m[:,3]+2*m[:,5])/12
    return dict(zeros=int(z.sum()),zero_NC_only=int((z&n&~d).sum()),zero_DAC_only=int((z&~n&d).sum()),zero_both_NC_DAC=int((z&n&d).sum()),
        zero_with_positive_NC_DAC=int((z&~n&~d).sum()),zero_utility0=int((z&(utility<=EPS)).sum()),TTC_fail_among_zero=int((z&(m[:,3]<1)).sum()),
        low_positive_le05=int(((m[:,0]>EPS)&(m[:,0]<.5)).sum()),nonzero_NCpartial=int(((m[:,0]>EPS)&(m[:,1]>0)&(m[:,1]<1)).sum()))


def shapley(v,base,mask=None):
    # Exact Shapley over canonical NC*DAC*(5EP+5TTC+2C)/12;
    # descriptive accounting of score change,not causal feature/gradient effect.
    a=base['values'];b=v['values'];n0,d0=a[:,1],a[:,2];dn,dd=b[:,1]-n0,b[:,2]-d0
    out=np.zeros((len(a),5))
    for j,col,w in [(2,3,5/12),(3,4,5/12),(4,5,2/12)]:
        c0=a[:,col];dc=b[:,col]-c0
        out[:,0]+=w*dn*(d0*c0+.5*(dd*c0+d0*dc)+dd*dc/3)
        out[:,1]+=w*dd*(n0*c0+.5*(dn*c0+n0*dc)+dn*dc/3)
        out[:,j]+=w*dc*(n0*d0+.5*(dn*d0+n0*dd)+dn*dd/3)
    assert np.max(np.abs(out.sum(1)-(b[:,0]-a[:,0])))<1e-12
    mask=np.ones(len(a),bool) if mask is None else mask
    return dict(method='exact5-component Shapley for canonical product/additive formula',units='PDMS points',
        contributions=dict(zip(('NC','DAC','TTC','EP','Comfort'),(100*out[mask].mean(0)).tolist())),total_delta=float(100*(b[mask,0]-a[mask,0]).mean()),
        note='Additive accounting of paired metric changes including interactions,not causal gradient attribution')


def journal_window(path,target):
    """Seek monotonic native JSONL using binary byte probes,then read64 rows."""
    start=target-63
    with path.open('rb') as f:
        lo=0;hi=path.stat().st_size;probes=0
        while lo<hi:
            mid=(lo+hi)//2;f.seek(mid)
            if mid:f.readline()
            pos=f.tell();line=f.readline();probes+=1
            if not line:
                hi=mid;continue
            row=json.loads(line)
            if row['updates']>=start:hi=mid
            else:lo=f.tell()
        # lo is advanced only to the end of a complete earlier row,so it
        # already points to a line boundary;do not discard the first target.
        f.seek(lo)
        rows=[]
        while True:
            line=f.readline()
            if not line:break
            row=json.loads(line)
            if row['updates']>target:break
            if row['updates']>=start:rows.append(row)
    assert [r['updates'] for r in rows]==list(range(start,target+1)),(target,[r['updates'] for r in rows[:2]],len(rows))
    metrics={}
    for k in ('policy_loss','kl_loss','kl_x','kl_x_norm','kl_v','clipfrac','ratio_mean','adv_kept_frac','pg_logprob_weight_abs_mean','positive_ppo_blocked_fraction','negative_ppo_blocked_fraction'):
        vals=[r['metrics'][k] for r in rows if k in r.get('metrics',{})]
        if vals:metrics[k]=dict(mean=float(np.mean(vals)),minimum=float(min(vals)),maximum=float(max(vals)))
    return dict(first_update=start,last_update=target,updates=len(rows),binary_seek_probes=probes,
        lr_used_action=sorted({r['lr'] for r in rows}),optimizer_group_lrs_next=rows[-1].get('optimizer_group_lrs'),
        preclip_gradient_norm_mean=float(np.mean([r['gradient_norm'] for r in rows])),preclip_above1_fraction=float(np.mean([r['gradient_norm']>1 for r in rows])),
        metrics=metrics,interpretation='Rank0 scalar losses/norms only; no PG-vs-reference gradient decomposition can be inferred')


def load_window(root,identity,target):
    cp=CACHE/f'u{target:06}_rollouts.npz';meta=CACHE/f'u{target:06}_rollouts.json'
    if cp.exists() and meta.exists():
        m=read(meta);assert m['identity']==identity['sha256'] and m['target']==target
        with np.load(cp,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
        return data,dict(m,cache_reused=True)
    rows=[];groups=[]
    first=(target-64)//4+1;last=target//4
    for c in range(first,last+1):
        for rank in range(8):
            p=root/'behavior'/f'collection{c:06}'/f'rank{rank}.pt';v=torch.load(p,map_location='cpu',mmap=True,weights_only=False)
            assert v['behavior_version']['run']==identity['sha256'] and v['behavior_version']['update']==(c-1)*4
            s=v['structured_advantage'];m=np.array([[r['metrics'][k] for k in KEYS] for r in v['records']])
            components=s['components'].double().numpy().reshape(8,4)
            assert np.max(abs(components-m[:,[1,2,3,4]]))<1e-12
            groups.append(dict(token=v['token'],log=physical(v['log']),collection=c,rank=rank,behavior_update=v['behavior_version']['update'],source=str(p),source_bytes=p.stat().st_size))
            def vector(key):return s[key].double().numpy().reshape(8)
            ndb=np.array([np.asarray(r.get('non_drivable_body_by_time',[]),bool).mean() if len(r.get('non_drivable_body_by_time',[])) else np.nan for r in v['records']])
            rows.append(dict(metrics=m,A_raw=vector('A_raw'),A_safe=vector('A_safe'),A_EP=vector('A_EP'),
                advantages=v['advantages'].double().numpy().reshape(8),mask=v['adv_mask'].double().numpy().reshape(8),normalization_std=float(v['normalization_std']),
                raw_progress=np.array([r['raw_progress_meters'] for r in v['records']]),ND_time_fraction=ndb,
                collision_seconds=np.array([np.nan if r.get('official_first_at_fault_collision_seconds') is None else r['official_first_at_fault_collision_seconds'] for r in v['records']]),
                TTC_seconds=np.array([np.nan if r.get('official_first_ttc_infraction_seconds') is None else r['official_first_ttc_infraction_seconds'] for r in v['records']])))
            del v
    data={k:np.stack([x[k] for x in rows]) for k in rows[0]}
    data['tokens']=np.array([x['token'] for x in groups]);data['logs']=np.array([x['log'] for x in groups]);data['behavior_update']=np.array([x['behavior_update'] for x in groups])
    norm_errors=[]
    for i in range(0,len(rows),8):
        std=np.std(data['A_raw'][i:i+8]);norm_errors.append(float(np.max(abs(data['advantages'][i:i+8]-data['A_raw'][i:i+8]/(std+1e-8)))))
        assert np.max(abs(data['normalization_std'][i:i+8]-std))<1e-12
    assert max(norm_errors)<2e-5
    tmp=cp.with_suffix('.tmp.npz');np.savez_compressed(tmp,**data);tmp.replace(cp)
    m=dict(schema='bounded_SafetyEP_G8_rollout_window_v1',target=target,identity=identity['sha256'],groups=len(groups),collections=16,rank_count=8,optimizer_updates=64,
        first_collection=first,last_collection=last,groups_provenance=groups,maximum_normalized_A_error=max(norm_errors),cache=str(cp),cache_sha256=sha(cp),
        source_payloads_read=len(groups),source_payload_bytes=sum(x['source_bytes'] for x in groups),method='only16sealed-behavior collections×8ranks perpoint,mmapCPU,no rawpayload rehash/model/scoring')
    atomic(meta,m);return data,dict(m,cache_reused=False)


def summarize_pg(d):
    m=d['metrics'];f=(m[:,:,[1,2,3]]==1).all(2);count=f.sum(1);raw=d['A_raw'];adv=d['advantages'];cost=(1-m[:,:,[1,2,3]]).mean(2)
    tied=np.max(abs(raw),axis=1)<=EPS;ep=m[:,:,4];saturated=(count==8)&(np.min(ep,axis=1)>=1-EPS)
    allfail=count==0;mixed=(count>0)&(count<8);progress=(count==8)&~tied
    classes={'all_feasible_EP1_saturated':saturated,'all_feasible_equal_EP_below1_tie':(count==8)&tied&~saturated,
        'all_fail_cost_tie':allfail&tied,'all_fail_cost_varied':allfail&~tied,'mixed_feasible_failed':mixed,'all_feasible_progress_only':progress}
    assert sum(int(v.sum()) for v in classes.values())==len(m)
    pos=raw>EPS;unsafe_pos=~f&pos;neg_safe=f&(raw<-EPS)
    assert not np.any(unsafe_pos&mixed[:,None])
    perclass={k:dict(groups=int(z.sum()),fraction=float(z.mean()),mean_abs_raw_A=float(np.mean(abs(raw[z]))) if z.any() else None,
        mean_abs_normalized_A=float(np.mean(abs(adv[z]))) if z.any() else None,strict_feasible_candidate_fraction=float(f[z].mean()) if z.any() else None) for k,z in classes.items()}
    global_scale=d['normalization_std'];groups_negative_safe=neg_safe.any(1)
    partial=unsafe_pos&(m[:,:,1]==.5)
    zero=m[:,:,0]<=EPS
    return dict(groups=len(m),candidates=int(m.shape[0]*8),unique_TRAIN_tokens=len(set(d['tokens'].tolist())),categories=perclass,
        feasible_groups=int((count>0).sum()),all_fail_groups=int(allfail.sum()),canonical_all8zero_groups=int(zero.all(1).sum()),
        PG_zero_groups=int(tied.sum()),successful_credit_saturation_groups=int(saturated.sum()),PG_active_groups=int((~tied).sum()),
        positive_unsafe=dict(candidates=int(unsafe_pos.sum()),groups=int(unsafe_pos.any(1).sum()),mixed_group_candidates=int((unsafe_pos&mixed[:,None]).sum()),all_fail_group_candidates=int((unsafe_pos&allfail[:,None]).sum()),
            NC05_candidates=int(partial.sum()),NC0_candidates=int((unsafe_pos&(m[:,:,1]==0)).sum()),DAC0_candidates=int((unsafe_pos&(m[:,:,2]==0)).sum()),TTC0_candidates=int((unsafe_pos&(m[:,:,3]==0)).sum())),
        negative_feasible=dict(candidates=int(neg_safe.sum()),groups=int(groups_negative_safe.sum()),mixed_groups=int((mixed&groups_negative_safe).sum()),all_feasible_progress_groups=int((progress&groups_negative_safe).sum()),
            mixed_group_by_feasible_count={str(i):dict(groups=int((count==i).sum()),any_negative_feasible=int(((count==i)&groups_negative_safe).sum())) for i in range(1,8)}),
        normalized_A_clip5_candidates=int((abs(adv)>5).sum()),normalized_std=dict(minimum=float(global_scale.min()),median=float(np.median(global_scale)),maximum=float(global_scale.max())),
        allfail_tie_extra_information=dict(groups=int((allfail&tied).sum()),raw_progress_varied=int(((allfail&tied)&(np.ptp(d['raw_progress'],axis=1)>1e-6)).sum()),
            ND_time_fraction_varied=int(((allfail&tied)&(np.nanmax(d['ND_time_fraction'],axis=1)-np.nanmin(d['ND_time_fraction'],axis=1)>1e-9)).sum())),
        sampler_vs_deployment=dict(exact_theta_ODE_pair_in_this_native_window=False,status='UNKNOWN_WITHOUT_EXACT_THETA_ODE',reason='behavior update precedes milestone and most scenes are not evaluated at same theta; do not equate G8allfail with deployment0'))


def train_panels():
    spec0=read(R/'IL_BASELINE_EVALUATION.json');allv=[];us=[];points=[]
    for u in (0,1000,5000,10000,20000):
        sp=R/'evaluation_specs'/f'{ARM}_u{u:06}_TRAIN1024.json';spec=spec0 if u==0 else read(sp) if sp.exists() else None
        if not spec:continue
        bank=Path(spec['training_bank']);summary=Path(spec['output'])/'SUMMARY.json'
        if not bank.exists() or not summary.exists() or read(summary).get('status')!='COMPLETE':continue
        b=read(bank);ts=sorted(b['records']);v=dict(values=np.array([[b['records'][t]['metrics'][k] for k in KEYS] for t in ts]),tokens=ts,logs=np.array([physical(b['records'][t]['log']) for t in ts]),label=f'TRAIN_{u}')
        assert len(ts)==1024
        if allv:assert ts==allv[0]['tokens']
        allv.append(v);us.append(u);points.append(dict(updates=u,PDMS=float(100*v['values'][:,0].mean()),zero=int((v['values'][:,0]<=EPS).sum()),bank=str(bank)))
    return dict(status='TRAIN_DIAGNOSTIC_NOT_HELDOUT',points=points,longitudinal=longitudinal(allv,us),
        caveat='current/oldDEV TRAIN panel can be exposed by uniformfull training; no Navtest tokens enter training via this diagnosis')


def main():
    torch.set_num_threads(1);row=next(r for r in read(R/'RUNS.json')['runs'] if r['id']==ARM);root=Path(row['root']);identity=read(root/'identity.json');progress=read(root/'progress.json')
    selection=read(R/'PARENT_SELECTION.json');il=R/'IL_PARENT';seal=read(il/'COMPLETE')
    assert seal['updates']==selection['selected']['updates']==70000 and seal['files_sha256']==selection['selected']['files_sha256'] and sha(il/'files.json')==seal['files_sha256'] and hashlib.sha256(json.dumps(read(il/'identity.json'),sort_keys=True,separators=(',',':')).encode()).hexdigest()==seal['identity_sha256']
    assert identity['IL_parent']['model_sha256']==selection['selected']['model_sha256']=='7dda5b14b1ea67fc631b06c7fa336950fa07a0b84c969f4e714bdeb5b5436d40'
    points=[];values=[];pending=[]
    for sp in sorted((R/'evaluation_specs').glob(f'{ARM}_u*_NAVTEST.json')):
        u=int(re.search(r'_u(\d+)_',sp.name).group(1));info=complete_input(R,ARM,u)
        if info is None:pending.append(dict(updates=u,spec=str(sp),status='PENDING_NOT_A_SCORE'));continue
        v=load_metrics(R,info);v['label']=info['label'];v['protocol']=info['identity']['protocol']['sha256'];values.append(v)
        points.append(dict(updates=u,PDMS=info['summary']['PDMS'],means=info['summary']['means'],zeros=causes(v),provenance=v['provenance']))
    us=[x['updates'] for x in points];base=values[us.index(10000)];high=base['values'][:,0]>=.9
    longitudinal_out=longitudinal(values,us);retention=[];attr=[]
    for u,v in zip(us,values):
        pair=contrast(v,base);h=contrast(v,base,high)
        retention.append(dict(updates=u,threshold=.9,cohort_definition='fixed Safety10k score>=.9',cohort_size=int(high.sum()),target_PDMS=h['target_PDMS'],delta_PDMS=h['delta_PDMS'],CI95_PDMS=h['CI95_PDMS'],
            high_to_zero=h['zeros']['high_to_zero'],target_zero=h['zeros']['target'],drops_ge10pp=int(((base['values'][:,0]-v['values'][:,0]>=.1)&high).sum()),gains_ge10pp=int(((v['values'][:,0]-base['values'][:,0]>=.1)&high).sum()),components=h['components']))
        attr.append(dict(updates=u,vs10k_PDMS_delta=pair['delta_PDMS'],vs10k_CI95=pair['CI95_PDMS'],component_metric_deltas=pair['components'],score_accounting=shapley(v,base),high_score_accounting=shapley(v,base,high)))
    cohort_accounting=[]
    for u,v in zip(us,values):
        buckets={}
        for name,mask in [('fixed10k_zero',base['values'][:,0]<=EPS),('fixed10k_positive_below09',(base['values'][:,0]>EPS)&~high),('fixed10k_high_ge09',high)]:
            score_delta=v['values'][mask,0]-base['values'][mask,0]
            buckets[name]=dict(scenes=int(mask.sum()),delta_PDMS_within_cohort=float(100*score_delta.mean()),
                contribution_to_global_PDMS_delta=float(100*score_delta.sum()/len(high)),
                repaired_zero=int(((base['values'][:,0]<=EPS)&(v['values'][:,0]>EPS)&mask).sum()),new_zero=int(((base['values'][:,0]>EPS)&(v['values'][:,0]<=EPS)&mask).sum()))
        cohort_accounting.append(dict(updates=u,baseline_updates=10000,cohorts=buckets,total_PDMS_delta=float(100*(v['values'][:,0]-base['values'][:,0]).mean())))
    windows=[];rolls=[];alltokens=[]
    milestones=[u for u in (1000,5000,10000,*range(12000,progress['updates']+1,2000)) if (root/'behavior'/f'collection{u//4:06}'/'rank7.pt').exists()]
    for u in milestones:
        d,m=load_window(root,identity,u);s=summarize_pg(d);windows.append(dict(updates=u,summary=s,journal=journal_window(root/'metrics_rank0.jsonl',u),cache_provenance={k:v for k,v in m.items() if k!='groups_provenance'}));rolls.append(d);alltokens.extend(d['tokens'].tolist())
    combined={k:np.concatenate([d[k] if d[k].ndim else d[k][None] for d in rolls]) for k in rolls[0]}
    probe=read(A/'FIXED_THETA_ZERO_PROBE_DIAGNOSIS_20261031.json')['arms'][ARM]
    result=dict(schema='world_SafetyEP_process_diagnosis_v1',status='COMPLETE',created_at=now(),live_native_progress_snapshot=progress,
        actions=dict(GPU_use=False,new_scoring=False,training_or_owner_source_edit=False,process_signal=False,NAVTEST_training_selection=False),
        parent_binding=dict(parent=str(il),updates=70000,model_sha256=selection['selected']['model_sha256'],seal_files_sha256=seal['files_sha256'],parent_selection_sha256=selection['sha256'],
            still_same_original_IL70=True,changed_by_this_audit=False,full_model_tensor_rehashed=False,method='native/selection metadata and small fileSHA bindings only'),
        evaluation_points=points,pending_evaluations=pending,zero_longitudinal=longitudinal_out,high10k_retention=retention,high10k_threshold=.9,component_attribution=attr,cohort_gain_loss_accounting=cohort_accounting,
        uncertainty=dict(bootstrap=2000,seed=42,physical_log_clusters=44,estimator='paired physical-log cluster resampling,scene-weighted ratio of sums',scope='same training/eval seed,not across-training-seed CI;multiple-point CIs descriptive and unadjusted',high_cohort='chosen conditionally on fixed10k score>=.9;bounded upside/selection conditioning,retention metric not unbiased overall performance estimate'),
        TRAIN_fixed1024=train_panels(),rollout_windows=windows,rollout_aggregate=summarize_pg(combined),
        rollout_sampling_limits=dict(optimizer_updates_perpoint=64,collections_perpoint=16,TRAIN_groups_perpoint=128,ranks=8,candidates_pergroup=8,
            windows=len(windows),total_groups=len(alltokens),unique_sampled_TRAIN_tokens=len(set(alltokens)),
            repeated_TRAIN_group_tokens_in_windows=len(alltokens)-len(set(alltokens)),not_full_training_population_estimator=True,all_ranks_per_collection_preserved=True,
            full_source_sampler='uniformResumableEpochSampler permutation; only current48000-ish scene exposures of103288,not full traversal; counters not multiplied again by4inner'),
        same_theta_sampler_evidence=dict(source=str(A/'FIXED_THETA_ZERO_PROBE_DIAGNOSIS_20261031.json'),status='REUSED_COMPLETE_SELECTED_TRAIN15',updates=10000,scenes=15,
            canonical_B1_zero=probe['counts']['canonical_B1_zero'],canonical_B1_zero_SDE8_has_feasible=probe['counts']['B1_zero_but_SDE8_has_feasible'],
            canonical_B1_zero_ODE8_has_feasible=probe['counts']['B1_zero_but_ODE8_has_feasible'],matched_roots_ODE_feasible_SDE_failed=probe['counts']['ODE_feasible_SDE_not'],matched_roots_SDE_feasible_ODE_failed=probe['counts']['SDE_feasible_ODE_not'],
            ODE8=probe['ODE8'],SDE8=probe['SDE8'],scenes_detail=[dict(token=s['token'],flags=s['flags'],canonical_single_ODE_PDMS=s['canonical_single_ODE_PDMS'],ODE8_feasible=s['ODE8']['strictly_feasible'],SDE8_feasible=s['SDE8']['strictly_feasible']) for s in probe['scenes']],
            interpretation='same theta+same roots selected TRAIN15 diagnostic; proves deploymentzero may have training feasible signal,also noise can destroy safe candidates; not population prevalence or training PG gradient'),
        credit_mechanism=dict(protocol=identity['structured_advantage_protocol'],mixed_group_unsafe_positive_A='impossible: unsafe A_safe=-m/8<0,A_EP=0; globalpositive std preserves sign',
            allfail_partialNC_positive_A='possible: mean(c)-c for c=mean(1-NC,1-DAC,1-TTC),so NC.5 candidate can have positive A when its totaldeficit is below groupmean; still not strict safety',
            conservative_feasible_negative_A='mixed group feasible: A_raw=1-m/8+EP_i-mean_EP_feasible; negative iff EP deficit exceeds1-m/8. allfeasible m8:EP belowmean negative by design; no extra safety margin ranking among metric-perfect candidates',
            global_std='one populationstd of64rawA from8freshgroups,zero groups included; changes magnitude not sign; source cap5 thereafter',
            mean_reference='beta50 loss is to original frozen IL70 transitionmeans,not KL to oldtheta; scalar kl_loss/PG magnitudes do not identify parameter-gradient attribution;group-centered signed PGmean can be near0 despite nonzero gradient,so larger mean kl_loss is not dominance proof',
            runtime_learning_scope='Qwen feature extractor frozen; current action attentionLoRA plus scene/predictive memory trainable,reference IL70 all frozen'),
        prioritized_hypotheses=[])
    ag=result['rollout_aggregate'];latest=points[-1];after10=[p for p in points if p['updates']>10000]
    result['findings_zh']=[
        f'完整NAVTEST已观测{us}；最新{latest["updates"]}步PDMS={latest["PDMS"]:.6f}，10k为91.349892。零分须看新增/修复与固定高分队列，不能只看总分或一个失败名单。',
        f'零分全观测持续{longitudinal_out["all_observations_persistent"]}、曾出现{longitudinal_out["ever_zero"]}、间歇{longitudinal_out["intermittent"]}、复发{longitudinal_out["relapsed_scenes"]}；“persistent”仅指当前{len(us)}个离散观测点。',
        f'有界TRAIN采样共{ag["groups"]}组，真正成功EP1饱和{ag["successful_credit_saturation_groups"]}组、全fail成本tie{ag["categories"]["all_fail_cost_tie"]["groups"]}组，不能把PG0统一当没学会。',
        f'正A不安全候选共{ag["positive_unsafe"]["candidates"]}条，mixed组中为0；只来自全fail相对成本排序。NC0.5正A{ag["positive_unsafe"]["NC05_candidates"]}条，不能视为已安全。',
        f'可行候选获负A的mixed组{ag["negative_feasible"]["mixed_groups"]}个；这是可行集合内部EP与安全margin相加的已知作用，不意味着这些候选客观最安全。',
        '同θTRAIN15已证明：6个部署B1零分场景中3个有SDE可行候选；同时120matched roots有41个ODE可行却被SDE破坏，5个被SDE修复。不能由跨θ历史宣称纯sampler因果。',
        '原IL70起点与冻结reference未换；full103k合法并不代表已遍历。均值reference的拉回方向要PG/anchor梯度分量实测，不能由scalarloss下结论。']
    result['prioritized_hypotheses']=[
        dict(priority=1,name='先保护固定高分并分解零分轮换',evidence='high10k_retention + zero_longitudinal adjacent transitions + canonical component score accounting',testable='fresh同IL70只改信用/约束一项，固定44logpaired zero修复/新增zero和high retention；不从NAVTEST tokens构建训练名单'),
        dict(priority=2,name='可行候选的EP压力与全fail成本排序分开处理',evidence=dict(mixed_groups=ag['categories']['mixed_feasible_failed']['groups'],mixed_negative_safe_groups=ag['negative_feasible']['mixed_groups'],allfail_positive_unsafe=ag['positive_unsafe']),testable='优先按实际出现频次选择小单变量：mixed降低EP权重，或allfail层次约束/真实TRAIN成功witness；不重复已有withinfeasibleEP，也不一刀切禁least-cost fallback'),
        dict(priority=3,name='候选信号转成部署ODE修复',evidence='same_theta_sampler_evidence selected15',testable='从fresh同IL70单独评估采样/ratio或受限verified-positive指导；不能把G8中奖当部署分，也不能混PG优势与历史teacher'),
        dict(priority=4,name='anchor/表征信息是否限制稳定安全',evidence='bounded native scalar windows show activity only; no separated gradients or current-risk identifiability evidence',testable='先小CPU/受限诊断区分PG与anchor梯度方向及current观测可辨识性；当前数据不足以证明应该减beta或解冻Qwen')]
    atomic(OUT,result);atomic(R/'reporting/refresh_request.json',dict(id='SAFETY_EP_PROCESS_'+uuid.uuid4().hex,scientific_result=False,evidence_kind='completed_supplemental_analysis',reason='SafetyEP原生过程/零分轮换/信用信号诊断完成',evidence=str(OUT)))
    print(json.dumps(dict(output=str(OUT),evaluation_updates=us,rollout_groups=ag['groups'],zero_longitudinal={k:longitudinal_out[k] for k in ('all_observations_persistent','ever_zero','intermittent','relapsed_scenes','latest_zero')},PG_categories={k:v['groups'] for k,v in ag['categories'].items()},positive_unsafe=ag['positive_unsafe'],mixed_negative_safe_groups=ag['negative_feasible']['mixed_groups']),ensure_ascii=False))
if __name__=='__main__':main()
