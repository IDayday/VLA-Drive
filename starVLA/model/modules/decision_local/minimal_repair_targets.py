"""Bounded offline local repair; independent polygon revalidation is mandatory."""
from dataclasses import dataclass,asdict
import time
import numpy as np
from scipy.optimize import minimize
from .minimal_repair_contract import CaseStatus
from .recovery_plans import PlanCurve


@dataclass(frozen=True)
class SolverConfig:
    starts: int = 3
    max_iterations: int = 100
    max_seconds: float = 3.
    ftol: float = 1e-6
    finite_difference_step: float = 1e-4
    lateral_bound_m: float = 1.5
    temporal_bound_s: float = .6
    endpoint_delay_s: float = .8
    ambiguity_relative_cost: float = .05
    ambiguity_distance_m: float = .5

    def to_dict(self): return asdict(self)


def solve_minimal_repair(plan,velocity,checker,config=SolverConfig()):
    begin=time.monotonic();cpu_begin=time.process_time()
    plan=np.asarray(plan,np.float32).copy();initial=checker.check(plan,velocity)
    record={'input_check':initial,'starts':[],'target':None,'target_valid':False}
    def finish(status,reason,target=None):
        record.update(status=int(status),status_name=status.name,reason=reason,
                      target=target,target_valid=target is not None,
                      wall_seconds=time.monotonic()-begin,cpu_seconds=time.process_time()-cpu_begin)
        return record
    if initial['out_of_range']: return finish(CaseStatus.OUT_OF_SCOPE,'full_body_out_of_fixed_map')
    if not initial['complete_annotations']:return finish(CaseStatus.UNCERTAIN,'map_height_object_or_future_support_incomplete')
    if initial['feasible']:
        if np.linalg.norm(plan[-1,:2])<1.:
            return finish(CaseStatus.TASK_UNCERTAIN,'geometry_feasible_but_progress_intent_unknown')
        return finish(CaseStatus.KEEP,'verified_annotated_local_constraints',plan.copy())
    if not initial['known_violation']:return finish(CaseStatus.UNCERTAIN,'boundary_or_clearance_ambiguity')
    cfg=checker.config
    curve=PlanCurve(plan,velocity,velocity_boundary=cfg.velocity_boundary);ts=curve.times[1:]
    starts=[np.zeros(4),np.array([-.75,0.,0.,0.]),np.array([.75,0.,0.,0.])][:config.starts]
    bounds=[(-config.lateral_bound_m,config.lateral_bound_m),(-.75,.75),
            (-config.temporal_bound_s,config.temporal_bound_s),(0,config.endpoint_delay_s)]
    candidates=[]
    def deadline():
        if time.monotonic()-begin>config.max_seconds:raise TimeoutError('per-input fixed solver wall budget')
    def generate(coefficients):
        deadline();return curve.poses(ts,coefficients)
    def objective(coefficients):
        target=generate(coefficients)
        xy=np.mean(np.sum((target[:,:2]-plan[:,:2])**2,-1))/.5**2
        yaw=np.arctan2(np.sin(target[:,2]-plan[:,2]),np.cos(target[:,2]-plan[:,2]))
        phi,_=curve.warp(ts,coefficients)
        return float(xy+np.mean(yaw*yaw)/.15**2+np.mean((phi-ts)**2)/.25**2)
    dense=np.linspace(0,curve.horizon,int(round(curve.horizon/cfg.sample_dt))+1)
    def constraints(coefficients):
        # Optimization uses 0.1s polygons; acceptance reconstructs the returned
        # eight-point trajectory independently at 0.05s, including its dynamics.
        target=generate(coefficients);reconstructed=PlanCurve(target,velocity,velocity_boundary=cfg.velocity_boundary)
        poses=reconstructed.poses(dense)
        value,valid,support=checker.relations(poses,dense)
        v=reconstructed.spline(dense,1);a=reconstructed.spline(dense,2)
        yawrate=np.abs(np.gradient(np.unwrap(poses[:,2]),dense,edge_order=2))
        _,dphi=curve.warp(dense,coefficients)
        endnorm=np.linalg.norm(plan[-1,:2]);direction=plan[-1,:2]/max(endnorm,1e-6)
        progress=np.dot(target[-1,:2],direction)-cfg.min_progress_fraction*endnorm if endnorm>.5 else 1.
        return np.r_[np.where(valid[:,0],value[:,0]-cfg.road_margin_m,-1.),
                     np.where(support['constraint_object_support'],value[:,1]-cfg.obstacle_clearance_m,-1.),
                     cfg.max_speed_mps-.01-np.linalg.norm(v,axis=-1),
                     cfg.max_acceleration_mps2-.02-np.linalg.norm(a,axis=-1),
                     cfg.max_yaw_rate_rps-.01-yawrate,dphi-.05,progress,
                     cfg.max_plan_change_m-np.linalg.norm(target[:,:2]-plan[:,:2],axis=-1),
                     cfg.max_heading_change_rad-np.abs(np.arctan2(np.sin(target[:,2]-plan[:,2]),np.cos(target[:,2]-plan[:,2])))]
    for start in starts:
        try:
            solution=minimize(objective,start,method='SLSQP',bounds=bounds,
                              constraints={'type':'ineq','fun':constraints},
                              options={'maxiter':config.max_iterations,'ftol':config.ftol,'eps':config.finite_difference_step,'disp':False})
            target=curve.poses(ts,solution.x).astype(np.float32)
            check=checker.check(target,velocity,reference=plan)
            cost=objective(solution.x)
            record['starts'].append({'success':bool(solution.success),'status':int(solution.status),
                'message':str(solution.message),'iterations':int(solution.nit),'coefficients':solution.x.tolist(),
                'cost':cost,'independent_check':check})
            if check['feasible']:candidates.append((cost,target,check))
        except TimeoutError:
            record['starts'].append({'timeout':True});break
        except ValueError as e:
            record['starts'].append({'invalid_parameterization':str(e)})
    if not candidates:return finish(CaseStatus.UNSOLVED,'no_independently_validated_solution_within_budget')
    candidates.sort(key=lambda x:x[0]);best=candidates[0]
    if any(abs(x[0]-best[0])<=max(1e-6,config.ambiguity_relative_cost*best[0]) and
           np.linalg.norm(x[1][:,:2]-best[1][:,:2],axis=-1).max()>config.ambiguity_distance_m for x in candidates[1:]):
        return finish(CaseStatus.AMBIGUOUS,'multiple_materially_distinct_near_equal_local_solutions')
    record['target_check']=best[2];record['selected_cost']=best[0]
    return finish(CaseStatus.REPAIR,'least_change_among_independently_validated_fixed_budget_solutions',best[1])
