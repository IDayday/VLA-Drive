"""CPU finite-K opportunities from immutable matched-root SDE/ODE audits.

Candidate maxima are discovery diagnostics, never deployed policy scores.
All scene denominators include scenes with no feasible candidate.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from starVLA.rl.qwen_flow_exploration.protocol import read_protocol
from starVLA.rl.qwen_flow_exploration.common_analysis import checked_seal,bootstrap,STRATA
from starVLA.rl.qwen_flow_exploration.preservation.sampler_audit import load_group,reference_rng_book
from starVLA.rl.qwen_sourceflow.contracts import fingerprint,file_sha
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest
from starVLA.rl.flow_grpo.transactions import atomic_json


SAFETY=('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound')


def arrays(records):
    if not records or any(len(group)!=16 for group in records):raise ValueError('Complete fixed K16 groups required')
    scores=np.asarray([[row['raw_reward'] for row in group] for group in records],dtype=np.float64)
    safe=np.asarray([[all(row['native_subscores'][key]==1 for key in SAFETY) for row in group] for group in records],dtype=bool)
    if not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)):raise ValueError('Finite original0-1 scores required')
    return derived(scores,safe)


def derived(scores,safe):
    if scores.ndim!=2 or scores.shape!=safe.shape or scores.shape[1]!=16 or safe.dtype!=bool:
        raise ValueError('Do not mix scenes/candidates or change K')
    feasible=safe.any(1)
    safe_max=np.where(safe,scores,-np.inf).max(1)
    return dict(scores=scores,safe=safe,scene_mean=scores.mean(1),overall_max=scores.max(1),
                feasible=feasible,safe_max=safe_max,safe_max_zero_proxy=np.where(feasible,safe_max,0.))


def summary(values,baselines,mask,threshold):
    n=int(mask.sum())
    if not n:return dict(status='NO_SCENES',scenes=0)
    s={key:value[mask] for key,value in values.items()};f=s['feasible'];safe_max=s['safe_max']
    result=dict(status='COMPLETE',scenes=n,candidates=n*16,feasible_scene_count=int(f.sum()),feasible_at16=float(f.mean()),
        safe_max_missing_scene_count=int((~f).sum()),mean_single_candidate_PDMS=100*float(s['scene_mean'].mean()),
        mean_overall_candidate_max_PDMS=100*float(s['overall_max'].mean()),
        mean_safe_max_zero_proxy_PDMS=100*float(s['safe_max_zero_proxy'].mean()),
        safe_max_available_only_mean_PDMS=100*float(safe_max[f].mean()) if f.any() else None,
        safe_max_available_only_denominator=int(f.sum()),fixed_denominator=n,
        safe_candidate_fraction=float(s['safe'].mean()),zero_score_candidate_fraction=float((s['scores']==0).mean()),
        full_score_candidate_fraction=float((s['scores']==1).mean()),
        feasible_scenes_safe_max_equals_overall_max=int((f&(np.abs(s['safe_max_zero_proxy']-s['overall_max'])<=1e-12)).sum()),
        overall_max_exceeds_safe_max_proxy_scene_count=int((s['overall_max']>s['safe_max_zero_proxy']+1e-12).sum()),
        scene_mean_PDMS_quantiles={str(q):100*float(np.quantile(s['scene_mean'],q)) for q in (.05,.1,.5,.9)},
        candidate_PDMS_quantiles={str(q):100*float(np.quantile(s['scores'],q)) for q in (.05,.1,.5,.9)},
        opportunities={})
    for name,baseline in baselines.items():
        b=baseline[mask];gains=safe_max-b
        counts={}
        for label,cutoff in [('strict_positive',0.),('ge_0.001',.001),('ge_useful',threshold)]:
            improve=(gains>0) if cutoff==0 else (gains>=cutoff)
            accepted=f&improve
            candidate_gain=s['scores']-b[:,None]
            candidate_good=s['safe']&((candidate_gain>0) if cutoff==0 else (candidate_gain>=cutoff))
            counts[label]=dict(raw_score_threshold=cutoff,scene_count=int(accepted.sum()),scene_fraction=float(accepted.mean()),
                safe_improving_candidate_fraction=float(candidate_good.mean()))
        result['opportunities'][name]=dict(baseline_mean_PDMS=100*float(b.mean()),thresholds=counts,
            mean_safe_gain_zero_proxy_PDMS_points=100*float((s['safe_max_zero_proxy']-b).mean()),
            mean_safe_gain_available_only_PDMS_points=100*float(gains[f].mean()) if f.any() else None,
            missing_safe_gain_scenes=int((~f).sum()),
            semantics='Raw safe gain is undefined without a feasible candidate. The fixed-population proxy sets only safe-max to0; it is not a scored trajectory or a successful repair.')
    return result


def load_audit(path,protocol,pool,book):
    path=Path(path);identity=checked_seal(json.loads((path/'identity.json').read_text()),'audit identity')
    result=checked_seal(json.loads((path/'result.json').read_text()),'audit result');seal=json.loads((path/'COMPLETE').read_text())
    tokens=protocol['splits']['TRAIN'][:128]
    if (identity['protocol_sha256']!=protocol['sha256'] or identity['base_model_sha256']!=protocol['base_model_sha256'] or
            identity['TRAIN_tokens_sha256']!=fingerprint(tokens) or identity['reference_rng_book_sha256']!=tensor_digest(book) or
            identity['G']!=16 or identity['scenes']!=128 or identity['K']!=10 or identity['chunk']!=4 or
            result['status']!='COMPLETE' or result['identity_sha256']!=identity['sha256'] or
            seal['identity_sha256']!=identity['sha256'] or seal['result_file_sha256']!=file_sha(path/'result.json')):
        raise ValueError('Wrong/incomplete same-root audit population or identity')
    if {p.name for p in (path/'scenes').iterdir() if p.is_dir() and '.incomplete.' not in p.name}!=set(tokens):
        raise ValueError('Missing/extra committed audit scenes')
    saved=[load_group(path/'scenes',identity,i,token,book[i],protocol) for i,token in enumerate(tokens)]
    sde=arrays([value['records'] for value in saved]);ode=arrays([value['ODE_records'] for value in saved])
    for name,values in [('SDE_current',sde),('ODE_current',ode)]:
        if abs(100*values['scene_mean'].mean()-result[name]['mean_PDMS'])>1e-10:
            raise ValueError('Published mean disagrees with complete actual candidate records')
    return dict(identity=identity,result=result,SDE=sde,ODE=ode,
                logs=[value['records'][0]['log_id'] for value in saved],tokens=tokens)


def paired(sde,ode,mask,logs,*,repeats):
    if not mask.any():return dict(status='NO_SCENES',scenes=0)
    current=sde['feasible'][mask];control=ode['feasible'][mask];diff=sde['safe_max_zero_proxy']-ode['safe_max_zero_proxy']
    return dict(status='COMPLETE',scenes=int(mask.sum()),both_feasible=int((current&control).sum()),
        SDE_only_feasible=int((current&~control).sum()),ODE_only_feasible=int((~current&control).sum()),neither_feasible=int((~current&~control).sum()),
        mean_SDE_minus_ODE_PDMS_points=100*float((sde['scene_mean']-ode['scene_mean'])[mask].mean()),
        mean_overall_max_SDE_minus_ODE_PDMS_points=100*float((sde['overall_max']-ode['overall_max'])[mask].mean()),
        mean_safe_max_zero_proxy_SDE_minus_ODE_PDMS_points=100*float(diff[mask].mean()),
        safe_max_SDE_better_ge005_scene_count=int((diff[mask]>=.005).sum()),
        safe_max_ODE_better_ge005_scene_count=int((diff[mask]<=-.005).sum()),
        safe_max_zero_proxy_paired_log_bootstrap=bootstrap(diff[mask],np.asarray(logs)[mask],repeats=repeats,seed=42,kind='score'),
        uncertainty='Finite fixedK16 maxima, not theoretical oracle. Scene-level log resampling only; no new training/root/innovation sampling uncertainty.')


def self_test():
    scores=np.full((2,16),.7);scores[1]=.8;safe=np.zeros((2,16),dtype=bool);safe[1,3]=True
    v=derived(scores,safe);r=summary(v,{'reference':np.asarray([.1,.5])},np.ones(2,dtype=bool),.005)
    assert r['fixed_denominator']==2 and r['feasible_scene_count']==1 and r['safe_max_missing_scene_count']==1
    assert abs(r['mean_safe_max_zero_proxy_PDMS']-40)<1e-12 and abs(r['safe_max_available_only_mean_PDMS']-80)<1e-12
    assert r['opportunities']['reference']['thresholds']['ge_useful']['scene_count']==1
    all_missing=summary(derived(scores,np.zeros_like(safe)),{'reference':np.asarray([0.,.5])},np.ones(2,dtype=bool),.005)
    assert all_missing['safe_max_available_only_mean_PDMS'] is None and all_missing['mean_safe_max_zero_proxy_PDMS']==0
    assert all_missing['opportunities']['reference']['thresholds']['ge_useful']['scene_count']==0
    assert summary(v,{},np.zeros(2,dtype=bool),.005)['status']=='NO_SCENES'
    row={'raw_reward':.9,'native_subscores':dict(no_at_fault_collisions=.5,drivable_area_compliance=1.,time_to_collision_within_bound=1.)}
    assert not arrays([[row]*16])['feasible'][0]
    return dict(status='PASS',qualification_only=True,no_GPU_or_new_scoring=True,
        checks=['Missing safe-max stays undefined;0 proxy and conditional denominator distinct','No feasible candidate never becomes a success','Empty stratum remains explicit','NC0.5 is not a complete safety pass'])


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--protocol');p.add_argument('--audit');p.add_argument('--base-audit')
    p.add_argument('--reference-pool');p.add_argument('--reference-strata');p.add_argument('--output',required=True)
    p.add_argument('--scene-output',help='Optional privateCSV; do not publish scene tokens or trajectories in aggregate reports')
    p.add_argument('--bootstrap-repeats',type=int,default=2000);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError('Keep earlier derived reports; no immutable result files are changed')
    out.parent.mkdir(parents=True,exist_ok=True)
    if a.self_test:atomic_json(out,self_test());return
    if not all([a.protocol,a.audit,a.reference_pool,a.reference_strata]):p.error('Provide protocol/audit/reference-pool/reference-strata')
    torch.set_num_threads(1);protocol=read_protocol(a.protocol)
    pool=torch.load(a.reference_pool,map_location='cpu',weights_only=False);book=reference_rng_book(pool)
    current=load_audit(a.audit,protocol,pool,book)
    if current['identity']['reference_pool_sha256']!=file_sha(a.reference_pool):raise ValueError('Wrong original reference pool')
    if current['identity']['arm']!='BASE' and not a.base_audit:p.error('Learned arms require --base-audit for the frozen initial ODE mean')
    base=current if current['identity']['arm']=='BASE' and not a.base_audit else load_audit(a.base_audit,protocol,pool,book)
    if (base['identity']['arm']!='BASE' or base['identity']['budget_sha256']!=current['identity']['budget_sha256'] or
            base['tokens']!=current['tokens'] or base['logs']!=current['logs']):raise ValueError('Common initial ODE baseline must be fixed/matched')
    strata=checked_seal(json.loads(Path(a.reference_strata).read_text()),'fixed independent reference strata')
    if strata['protocol_sha256']!=protocol['sha256']:raise ValueError('Wrong reference stratification')
    fixed=[strata['by_split']['TRAIN'][token] for token in current['tokens']]
    if any(row['stratum'] not in STRATA for row in fixed):raise ValueError('Do not redefine strata after sampling')
    independent=np.asarray([row['reference_mean'] for row in fixed],dtype=np.float64)
    if not np.isfinite(independent).all() or np.any((independent<0)|(independent>1)):raise ValueError('Invalid independent5 reference means')
    baselines=dict(frozen_BASE_matched_root_ODE_mean_K16=base['ODE']['scene_mean'],independent5_reference_mean=independent)
    masks={'ALL':np.ones(128,dtype=bool),**{label:np.asarray([row['stratum']==label for row in fixed]) for label in STRATA}}
    reports={label:dict(SDE=summary(current['SDE'],baselines,mask,protocol['practical']['mean_gain_raw']),
        ODE=summary(current['ODE'],baselines,mask,protocol['practical']['mean_gain_raw']),
        paired_SDE_minus_ODE=paired(current['SDE'],current['ODE'],mask,current['logs'],repeats=a.bootstrap_repeats)) for label,mask in masks.items()}
    report=dict(schema='finiteK16_matched_root_safe_opportunities_v1',status='CANDIDATE_OPPORTUNITY_ONLY',
        arm=current['identity']['arm'],protocol_sha256=protocol['sha256'],audit_identity=current['identity']['sha256'],base_audit_identity=base['identity']['sha256'],
        audit_result_sha256=file_sha(Path(a.audit)/'result.json'),reference_pool_sha256=file_sha(a.reference_pool),strata_sha256=strata['sha256'],
        analysis_source_sha256=file_sha(__file__),scenes=128,logs=len(set(current['logs'])),K=16,
        criterion='Official NC==1 and DAC==1 and TTC==1; NC0.5 is not safe. Uses true canonical raw PDMS.',
        denominator='All128 scenes retained. Missing safe-max is undefined;0 proxy is explicitly separate from a scored trajectory. Available-only mean is supplementary with its count.',
        baseline_semantics=dict(frozen_BASE_matched_root_ODE_mean_K16='Frozen BASE empirical mean over16 identical stored initial roots; also own candidate pool for BASEODE, so not independent of its max.',
                                independent5_reference_mean='Frozen original common-bank five independent reference seeds11001..11005; also defines strata, never used to select new outputs.'),
        no_new_model_forward_or_score=True,reporting_scale='All PDMS fields are raw0-1 multiplied100; gains .005 raw equal0.5PDMS points.',
        no_deployment_or_oracle_guarantee=True,by_stratum=reports)
    report['sha256']=fingerprint(report);atomic_json(out,report)
    if a.scene_output:
        destination=Path(a.scene_output)
        if destination.exists():raise FileExistsError(destination)
        destination.parent.mkdir(parents=True,exist_ok=True)
        rows=[]
        for i,token in enumerate(current['tokens']):
            row=dict(scene_token=token,log_id=current['logs'][i],stratum=fixed[i]['stratum'],frozen_BASE_ODE_mean=float(base['ODE']['scene_mean'][i]),independent5_reference_mean=float(independent[i]))
            for mode in ('SDE','ODE'):
                v=current[mode];f=bool(v['feasible'][i]);row.update({mode+'_mean':float(v['scene_mean'][i]),mode+'_overall_max':float(v['overall_max'][i]),
                    mode+'_feasible':f,mode+'_safe_max':float(v['safe_max'][i]) if f else None,mode+'_safe_max_zero_proxy':float(v['safe_max_zero_proxy'][i])})
            rows.append(row)
        with destination.open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(status=report['status'],arm=report['arm'],ALL=reports['ALL']),indent=2))


if __name__=='__main__':main()
