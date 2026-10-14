"""Measure raster/body sampling approximation against mature polygon distances."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from shapely import contains_xy,distance,points
from shapely.geometry import Polygon,box,MultiPoint
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,digest,source_sha
from starVLA.dataloader.structured_world.labels import road_distance
from starVLA.model.modules.structured_world.grid import GridSpec
from starVLA.model.modules.structured_world.queries import body_points
from starVLA.model.modules.decision_local.consequence_geometry import read_consequences


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);grid=GridSpec();body={'length':5.176,'width':2.297,'rear_axle_to_center':1.461}
    polygons={'straight':box(-15,-5,70,5),'corner':Polygon([(-15,-5),(20,-5),(20,20),(10,20),(10,5),(-15,5)]),
        'hole':Polygon([(-15,-20),(70,-20),(70,20),(-15,20)],holes=[[(12,-2),(18,-2),(18,2),(12,2)]])}
    poses=np.array([[x,y,yaw] for x in (0.,9.,12.,15.,18.,21.) for y in (-4.,0.,4.) for yaw in (0.,np.pi/6,np.pi/2)],np.float32)
    ones=np.ones((1,grid.height,grid.width),bool);occ=np.zeros_like(ones,dtype=np.uint8);rows=[]
    dense=body_points(torch.from_numpy(poses),body,spacing_m=.05).numpy()
    for name,polygon in polygons.items():
        field=road_distance(polygon,grid)
        approximate,valid,_=read_consequences(field,ones[0],occ,ones,poses[:,None],body,grid=grid)
        for i,pose in enumerate(poses):
            cloud=dense[i];inside=contains_xy(polygon,cloud[:,0],cloud[:,1]);unsigned=distance(points(cloud),polygon.boundary)
            exact=np.clip(np.where(inside,unsigned,-unsigned),-10.,10.).min()
            value=float(approximate[i,0,0]);rows.append({'shape':name,'pose':pose.tolist(),'raster_body_margin_m':value,
                'polygon_point_reference_margin_m':float(exact),'absolute_error_m':abs(value-exact),'valid':bool(valid[i,0,0])})
    # For disjoint obstacles, body-to-polygon clearance is exact; overlap depth
    # is intentionally not called exact polygon penetration depth.
    obstacle=box(12,-2,16,2);centers=grid.centers().numpy();oc=contains_xy(obstacle,centers[...,0],centers[...,1]).astype(np.uint8)[None]
    field=road_distance(box(-15,-30,70,30),grid)
    result,support,_=read_consequences(field,ones[0],oc,ones,poses[:,None],body,grid=grid)
    obstacle_rows=[]
    corners=body_points(torch.from_numpy(poses),body).numpy()
    for i,pose in enumerate(poses):
        footprint=MultiPoint(corners[i]).convex_hull;overlap=footprint.intersects(obstacle)
        exact=float(footprint.distance(obstacle));pred=float(result[i,0,1])
        obstacle_rows.append({'pose':pose.tolist(),'exact_polygon_gap_m':min(exact,10.),'raster_clearance_m':pred,
            'exact_overlap':bool(overlap),'raster_overlap':bool(result[i,0,2]),'clearance_valid':bool(support[i,0,1]),
            'gap_error_when_disjoint_m':abs(pred-min(exact,10.)) if not overlap and support[i,0,1] else None})
    errors=[r['absolute_error_m'] for r in rows if r['valid']]
    atomic_json(a.output,{'source_sha':source_sha(ROOT),'tool_sha256':digest(__file__),'road_cases':rows,'obstacle_cases':obstacle_rows,
        'road_error_quantiles_m':np.quantile(errors,[.5,.9,.99,1]).tolist(),'grid_m':.5,'body_sample_m':.5,'polygon_reference_body_sampling_m':.05,
        'reference_limit':'Exact Shapely point distances on0.05m body samples; not an analytic continuous-body minimum. Disjoint polygon gap is analytic.',
        'no_private_images_or_dev_test_scores':True})


if __name__=='__main__':main()
