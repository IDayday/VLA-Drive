#!/usr/bin/env python3
"""Small log-disjoint train-only diagnostic sample, never a Navtest regret list."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from shapely.geometry import Point
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from omegaconf import OmegaConf
    from navsim.planning.training.dataset import load_feature_target_from_pickle
    from navsim.agents.EpisodeDrive.layers.world_model.physical_label_sidecar import load_metric,original_road_geometry
    import os
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cache-root','metric-root','train-config','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--per-category',type=int,default=4)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    cfg=OmegaConf.load(args.train_config)
    if set(cfg.train_logs)&set(cfg.val_logs):raise ValueError('Train/val logs overlap')
    groups={name:[] for name in ('turn','stop','crowded','boundary')}
    for log in sorted(cfg.train_logs):
        directory=args.cache_root/log
        if not directory.is_dir():continue
        paths=sorted(directory.glob('*/planreg_input_only.gz'))
        for path in paths[:6]:
            record=load_feature_target_from_pickle(path)
            targets,features=record['targets'],record['features']
            if targets.get('task_future_input_schema') is None:continue
            token=targets['token']
            metric_path=args.metric_root/log/'unknown'/token/'metric_cache.pkl'
            if not metric_path.is_file():continue
            metric,_=load_metric(str(metric_path))
            status=np.asarray(features['status_feature'])
            speed=float(np.linalg.norm(status[4:6]))
            heading=float(abs(targets['trajectory'][-1,2]))
            actors=len(metric.observation[0].tokens)
            union,coverage,_=original_road_geometry(str(metric_path),targets['physical_map_name'],os.environ['NUPLAN_MAPS_ROOT'])
            corners=[Point(xy) for xy in metric.ego_state.car_footprint.geometry.exterior.coords[:-1]]
            margin=None if union is None else min(p.distance(union.boundary)*(1 if union.covers(p) else -1) for p in corners)
            flags=dict(turn=heading>.3,stop=speed<.5,crowded=actors>=30,boundary=margin is not None and margin<.5)
            available=[name for name in groups if flags[name] and len(groups[name])<args.per_category]
            if not available:continue
            category=min(available,key=lambda name:len(groups[name]))
            groups[category].append(dict(log=log,token=token,category=category,speed=speed,
                endpoint_heading_abs=heading,actor_count=actors,current_road_margin=margin,
                record=str(path),metric_cache=str(metric_path)))
            print({name:len(rows) for name,rows in groups.items()},flush=True)
            break  # one scene per log; split cannot share a log
        if all(len(rows)>=args.per_category for rows in groups.values()):break
    if not all(len(rows)>=args.per_category for rows in groups.values()):
        raise RuntimeError(f'Insufficient distinct category logs: {[(k,len(v)) for k,v in groups.items()]}')
    rows=[]
    for category,items in groups.items():
        for i,row in enumerate(items):
            row['partition']='train' if i<args.per_category//2 else 'development'
            rows.append(row)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(split='trainval_train_logs_only',rows=rows,
        log_disjoint=True,selection='current state, logged actor count, raw map and GT turn; no evaluator scores',
        vlm_prior_exposure='Public Base pretraining log exposure unknown; no old trained planner used'),indent=2)+'\n')


if __name__=='__main__':main()
