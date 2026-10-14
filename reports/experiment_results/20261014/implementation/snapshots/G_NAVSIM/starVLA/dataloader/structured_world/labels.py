"""Derived endpoint labels using pinned UniAD geometry and mature map APIs."""
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from shapely import affinity, contains_xy
from shapely.geometry import box
from shapely.ops import unary_union
from .centered_boxes import CenteredOccFlowLabels
from starVLA.model.modules.structured_world.grid import GridSpec
from .cameras import projected_depth


@lru_cache(maxsize=16)
def read_lidar(path):
    if Path(path).suffix == '.pcd':
        from nuplan.database.utils.pointclouds.lidar import LidarPointCloud
        cloud = LidarPointCloud.from_buffer(Path(path).read_bytes(), 'pcd')
    else:
        from nuscenes.utils.data_classes import LidarPointCloud
        cloud = LidarPointCloud.from_file(path)
    return cloud.points[:3].T.astype(np.float32)


def lidar_ego(frame):
    cloud = read_lidar(frame.lidar_path)
    return cloud@frame.lidar2ego[:3, :3].T + frame.lidar2ego[:3, 3]


def annotation_ground_support(points, grid, *, maximum_fill_m=2.):
    """Conservative local road-level evidence, never inferred from no returns.

    Low LiDAR returns define a current-road level near the ego. We retain only
    surfaces within one metre of that level and interpolate at most two metres.
    Large grade changes and missing evidence are unknown. This is a disclosed
    label-validity approximation, not a 3D semantic occupancy reconstruction.
    """
    points = np.asarray(points)
    radius = np.linalg.norm(points[:, :2], axis=-1)
    near = points[(radius >= 2.) & (radius <= 8.) & (points[:, 2] > -4.) & (points[:, 2] < 1.)]
    if len(near) < 20:
        return np.full((grid.height, grid.width), np.nan, dtype=np.float32), np.zeros((grid.height, grid.width), bool), {'ground_level': None, 'reason': 'insufficient_near_ego_returns'}
    ground = float(np.quantile(near[:, 2], .15))
    selected = points[(points[:, 2] >= ground-1.) & (points[:, 2] <= ground+.75)]
    cell = np.floor((selected[:, :2]-[grid.xmin, grid.ymin])/[grid.dx, grid.dy]).astype(np.int64)
    inside = (cell[:, 0] >= 0) & (cell[:, 0] < grid.width) & (cell[:, 1] >= 0) & (cell[:, 1] < grid.height)
    cell, selected = cell[inside], selected[inside]
    surface = np.full(grid.height*grid.width, np.inf, dtype=np.float32)
    np.minimum.at(surface, cell[:, 1]*grid.width+cell[:, 0], selected[:, 2])
    surface = surface.reshape(grid.height, grid.width)
    observed = np.isfinite(surface)
    if not observed.any():
        return np.full_like(surface, np.nan), observed, {'ground_level': ground, 'reason': 'no_registered_grid_ground_returns'}
    distance, nearest = distance_transform_edt(~observed, sampling=(grid.dy, grid.dx), return_indices=True)
    filled = surface[tuple(nearest)]
    valid = (distance <= maximum_fill_m) & (np.abs(filled-ground) <= 1.)
    filled[~valid] = np.nan
    return filled, valid, {'ground_level': ground, 'ground_fill_m': maximum_fill_m, 'valid_fraction': float(valid.mean())}


class RoadMap:
    def __init__(self, dataset, root, *, version='nuplan-maps-v1.0'):
        self.dataset, self.root, self.version = dataset, root, version
        self.maps = {}

    def polygon(self, scene, grid):
        pose = scene.inverse_transform
        yaw = np.arctan2(pose[1, 0], pose[0, 0])
        if self.dataset == 'navsim':
            from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
            from nuplan.common.maps.maps_datatypes import SemanticMapLayer
            if scene.map_name not in self.maps:
                self.maps[scene.map_name] = get_maps_api(self.root, self.version, scene.map_name)
            # nuPlan's vector drivable layer is the established union of road
            # segments, intersections, generic drivable areas and car parks.
            layer = self.maps[scene.map_name]._get_vector_map_layer(SemanticMapLayer.DRIVABLE_AREA)
            patch = box(pose[0, 3]-150., pose[1, 3]-150., pose[0, 3]+150., pose[1, 3]+150.)
            selected = layer.geometry[layer.geometry.intersects(patch)]
            polygon = unary_union(selected.tolist())
            polygon = affinity.translate(polygon, -pose[0, 3], -pose[1, 3])
            return affinity.rotate(polygon, -yaw, origin=(0., 0.), use_radians=True)
        from third_party.vad.ported.polygon_map import VADPolygonMap
        if scene.map_name not in self.maps:
            self.maps[scene.map_name] = VADPolygonMap(self.root, scene.map_name)
        # The mature VAD extraction includes holes and all drivable_area pieces.
        polygons = self.maps[scene.map_name].get_contour_line(
            (pose[0, 3], pose[1, 3], 300., 300.), np.degrees(yaw), 'drivable_area', scene.map_name)
        return unary_union(polygons)


def road_distance(polygon, grid, *, clip_m=10.):
    """Signed EDT on an expanded grid; subcell approximation is measured."""
    margin_x, margin_y = int(np.ceil((clip_m+1)/grid.dx)), int(np.ceil((clip_m+1)/grid.dy))
    xs = grid.xmin + (np.arange(-margin_x, grid.width+margin_x)+.5)*grid.dx
    ys = grid.ymin + (np.arange(-margin_y, grid.height+margin_y)+.5)*grid.dy
    xx, yy = np.meshgrid(xs, ys)
    inside = contains_xy(polygon, xx, yy)
    # Half-cell adjustment locates an axis-aligned edge between pixel centers.
    din = distance_transform_edt(inside, sampling=(grid.dy, grid.dx))
    dout = distance_transform_edt(~inside, sampling=(grid.dy, grid.dx))
    distance = np.where(inside, np.maximum(0., din-.5*min(grid.dx, grid.dy)),
                        -np.maximum(0., dout-.5*min(grid.dx, grid.dy)))
    return np.clip(distance[margin_y:margin_y+grid.height, margin_x:margin_x+grid.width],
                   -clip_m, clip_m).astype(np.float32)


def occupancy(scene, grid):
    generator = CenteredOccFlowLabels(dict(xbound=[grid.xmin, grid.xmax, grid.dx],
        ybound=[grid.ymin, grid.ymax, grid.dy], zbound=[-4., 4., 8.]),
        only_vehicle=False, filter_invisible=False, ignore_index=255, compute_flow=False)
    generator.filter_cls_ids = np.arange(10)  # Include pedestrians, cones and barriers.
    tracks = sorted({token for frame in scene.frames if frame is not None for token in frame.track_tokens})
    ids = {token: i+1 for i, token in enumerate(tracks)}
    identity = np.eye(3, dtype=np.float32)
    reference = scene.inverse_transform
    results = {'future_gt_bboxes_3d': [], 'future_gt_labels_3d': [], 'future_gt_inds': [],
               'future_gt_vis_tokens': [], 'occ_l2e_r_mats': [], 'occ_l2e_t_vecs': [],
               'occ_e2g_r_mats': [], 'occ_e2g_t_vecs': [], 'occ_has_invalid_frame': False,
               'occ_img_is_valid': np.array([f is not None for f in scene.frames])}
    for frame in scene.frames:
        results['future_gt_bboxes_3d'].append(frame.boxes if frame else None)
        results['future_gt_labels_3d'].append(frame.labels if frame else None)
        results['future_gt_inds'].append(np.array([ids[t] for t in frame.track_tokens]) if frame else None)
        results['future_gt_vis_tokens'].append(frame.visibility if frame else None)
        # Every adapter already converted sensor boxes into its current ego.
        results['occ_l2e_r_mats'].append(identity)
        results['occ_l2e_t_vecs'].append(np.zeros(3, dtype=np.float32))
        # Subtract global coordinates in FP64 before invoking the mature box
        # transformations. NAVSIM map coordinates can be hundreds of km.
        relative_pose = np.linalg.inv(reference)@(frame.ego2global if frame else reference)
        results['occ_e2g_r_mats'].append(relative_pose[:3, :3].astype(np.float32))
        results['occ_e2g_t_vecs'].append(relative_pose[:3, 3].astype(np.float32))
    generated = generator(results)
    occupied = generated['gt_segmentation'].numpy().astype(np.uint8)
    validity, height_records, current_points = [], [], None
    box_index = 0
    for t, frame in enumerate(scene.frames):
        reframed = generated['gt_future_boxes'][box_index] if frame is not None else None
        box_index += int(frame is not None)
        if frame is None or not frame.annotations_complete:
            occupied[t] = 255
            validity.append(np.zeros((grid.height, grid.width), bool))
            height_records.append({'valid': False, 'reason': 'missing_annotation_frame'})
            continue
        try:
            points = lidar_ego(frame)
        except (RuntimeError, EOFError, FileNotFoundError) as error:
            # Missing/truncated label-only LiDAR never turns into free space.
            # The scene remains in the original FM/trajectory population.
            occupied[t] = 255
            validity.append(np.zeros((grid.height, grid.width), bool))
            height_records.append({'valid': False, 'reason': 'invalid_label_only_lidar',
                'sensor_path': frame.lidar_path, 'error_type': type(error).__name__, 'error': str(error)})
            if t == 0: current_points = np.empty((0, 3), dtype=np.float32)
            continue
        if t == 0:
            current_points = points
        relative = np.linalg.inv(reference)@frame.ego2global
        points = points@relative[:3, :3].T + relative[:3, 3]
        surface, supported, quality = annotation_ground_support(points, grid)
        # A high/unknown-level object is never projected as a ground obstacle,
        # nor is its footprint silently replaced with known free space.
        if len(reframed):
            corners = reframed.corners[:, [0, 3, 7, 4], :2].numpy()
            cells = np.round((corners-[grid.xmin, grid.ymin])/[grid.dx, grid.dy]).astype(np.int32)
            for object_index, polygon in enumerate(cells):
                body = np.zeros((grid.height, grid.width), np.uint8)
                cv2.fillPoly(body, [polygon], 1)
                footprint = body.astype(bool)
                known = footprint & supported
                bottom = float(reframed.tensor[object_index, 2])
                reliable = known.any() and np.median(np.abs(surface[known]-bottom)) <= 1.
                if not reliable:
                    supported[footprint] = False
                else:
                    supported[footprint] = True  # Trusted annotation preserves occluded objects.
        occupied[t, ~supported] = 255
        validity.append(supported)
        height_records.append(quality)
    return occupied, np.stack(validity), generated['gt_instance'].numpy(), current_points, height_records


def build_labels(scene, road_map, grid=None):
    grid = grid or GridSpec()
    labels, valid, instances, points, height_records = occupancy(scene, grid)
    polygon = road_map.polygon(scene, grid)
    distance = road_distance(polygon, grid)
    road_valid = valid[0] & (not polygon.is_empty)
    depth = projected_depth(points, scene.observations['calibration'], scene.observations['geometry_pixel_valid'],
                            image_size=scene.observations['geometry_images'].shape[-2:])
    return {'road_distance': distance, 'road_valid': road_valid, 'occupancy': labels,
            'occupancy_valid': valid, 'instances': instances, 'depth': depth.numpy(),
            'ego_physical': scene.ego_physical, 'future_valid': scene.future_valid,
            'height_quality': height_records, 'inverse_transform': scene.inverse_transform,
            'ego_body': scene.ego_body, 'protocol': scene.protocol,
            'label_origin': 'derived offline map/box/LiDAR annotations; not native risk GT',
            'occupancy_event_semantics': 'endpoint state change in fixed ego(t0), not cumulative first entry'}
