"""Front-camera features and independently rendered semantic BEV targets."""

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np
import torch
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer
from PIL import Image
from shapely import affinity
from shapely.geometry import LineString, Polygon

from navsim.common.dataclasses import AgentInput, Annotations, Scene
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

MAP_CLASS_NAMES = ("background", "road", "walkway", "centerline")
AGENT_CLASS_NAMES = ("background", "vehicle", "pedestrian")
VEHICLE_ONLY_AGENT_CLASS_NAMES = ("background", "vehicle")


def get_agent_class_names(include_pedestrians: bool = True) -> tuple[str, ...]:
    return (
        AGENT_CLASS_NAMES
        if include_pedestrians
        else VEHICLE_ONLY_AGENT_CLASS_NAMES
    )


@dataclass
class RetrieveTargetConfig:
    bev_pixel_height: int = 128
    bev_pixel_width: int = 256
    bev_pixel_size: float = 0.25
    bev_radius: float = 32.0
    centerline_thickness: int = 2
    include_pedestrians: bool = True


class RetrieveFeatureBuilder(AbstractFeatureBuilder):
    """Build normalized camera tensors while retaining a camera dimension."""

    CAMERA_IDS = {
        "cam_f0": 0,
        "cam_l0": 1,
        "cam_r0": 2,
        "cam_b0": 3,
    }

    def __init__(
        self,
        image_size: Sequence[int] = (1148, 672),
        camera_names: Sequence[str] = ("cam_f0",),
    ) -> None:
        self.image_size = tuple(int(value) for value in image_size)
        self.camera_names = tuple(camera_names)
        unknown = set(self.camera_names) - set(self.CAMERA_IDS)
        if unknown:
            raise ValueError(f"Unsupported camera names: {sorted(unknown)}")
        if not self.camera_names:
            raise ValueError("At least one camera is required")

    def get_unique_name(self) -> str:
        return "retrieve_v1_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        cameras = agent_input.cameras[-1]
        images = []
        camera_ids = []
        camera_intrinsics = []
        lidar_to_camera = []

        for camera_name in self.camera_names:
            camera = getattr(cameras, camera_name)
            if camera.image is None:
                raise RuntimeError(f"Required camera {camera_name} is missing")

            image = Image.fromarray(camera.image).resize(self.image_size)
            image_array = np.asarray(image, dtype=np.float32) / 255.0
            image_array = (image_array - IMAGENET_MEAN) / IMAGENET_STD
            images.append(torch.from_numpy(image_array).permute(2, 0, 1))
            camera_ids.append(self.CAMERA_IDS[camera_name])

            intrinsics = np.asarray(camera.intrinsics, dtype=np.float32).copy()
            original_height, original_width = camera.image.shape[:2]
            intrinsics[0, :] *= self.image_size[0] / original_width
            intrinsics[1, :] *= self.image_size[1] / original_height
            camera_intrinsics.append(torch.from_numpy(intrinsics))

            sensor_to_lidar = np.eye(4, dtype=np.float32)
            sensor_to_lidar[:3, :3] = np.asarray(
                camera.sensor2lidar_rotation,
                dtype=np.float32,
            )
            sensor_to_lidar[:3, 3] = np.asarray(
                camera.sensor2lidar_translation,
                dtype=np.float32,
            )
            lidar_to_camera.append(torch.from_numpy(np.linalg.inv(sensor_to_lidar)))

        return {
            "image": torch.stack(images),
            "camera_ids": torch.tensor(camera_ids, dtype=torch.long),
            "cam_K": torch.stack(camera_intrinsics),
            "world_2_cam": torch.stack(lidar_to_camera),
        }


class RetrieveTargetBuilder(AbstractTargetBuilder):
    """Render independent map and agent semantic labels.

    Map labels follow the TransFuser ordering for classes 1-3. Agent labels
    retain vehicles and optionally pedestrians, with branch-local labels.
    The two label maps are rendered separately, so map and agent semantics may
    occupy the same BEV pixel.
    """

    def __init__(
        self,
        bev_pixel_height: int = 128,
        bev_pixel_width: int = 256,
        bev_pixel_size: float = 0.25,
        bev_radius: float = 32.0,
        centerline_thickness: int = 2,
        include_pedestrians: bool = True,
    ) -> None:
        self.config = RetrieveTargetConfig(
            bev_pixel_height=int(bev_pixel_height),
            bev_pixel_width=int(bev_pixel_width),
            bev_pixel_size=float(bev_pixel_size),
            bev_radius=float(bev_radius),
            centerline_thickness=int(centerline_thickness),
            include_pedestrians=bool(include_pedestrians),
        )

    def get_unique_name(self) -> str:
        if self.config.include_pedestrians:
            return "retrieve_v1_target"
        return "retrieve_vehicle_only_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        frame_idx = scene.scene_metadata.num_history_frames - 1
        frame = scene.frames[frame_idx]
        annotations = frame.annotations
        ego_pose = StateSE2(*frame.ego_status.ego_pose)

        map_target = np.zeros(self._bev_shape, dtype=np.int64)
        road_mask = self._compute_map_polygon_mask(
            scene.map_api,
            ego_pose,
            [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION],
        )
        walkway_mask = self._compute_map_polygon_mask(
            scene.map_api,
            ego_pose,
            [SemanticMapLayer.WALKWAYS],
        )
        centerline_mask = self._compute_map_linestring_mask(
            scene.map_api,
            ego_pose,
            [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
        )
        map_target[road_mask] = 1
        map_target[walkway_mask] = 2
        map_target[centerline_mask] = 3

        agent_target = np.zeros(self._bev_shape, dtype=np.int64)
        vehicle_mask = self._compute_box_mask(
            annotations,
            [TrackedObjectType.VEHICLE],
        )
        agent_target[vehicle_mask] = 1
        if self.config.include_pedestrians:
            pedestrian_mask = self._compute_box_mask(
                annotations,
                [TrackedObjectType.PEDESTRIAN],
            )
            agent_target[pedestrian_mask] = 2

        if map_target.shape != self._bev_shape or agent_target.shape != self._bev_shape:
            raise RuntimeError(
                f"Target shape mismatch: map={map_target.shape}, "
                f"agent={agent_target.shape}, expected={self._bev_shape}"
            )
        return {
            "map_target": torch.from_numpy(map_target),
            "agent_target": torch.from_numpy(agent_target),
        }

    @property
    def _bev_shape(self):
        return (self.config.bev_pixel_height, self.config.bev_pixel_width)

    def _compute_map_polygon_mask(
        self,
        map_api: AbstractMap,
        ego_pose: StateSE2,
        layers: List[SemanticMapLayer],
    ) -> np.ndarray:
        objects = map_api.get_proximal_map_objects(
            point=ego_pose.point,
            radius=self.config.bev_radius,
            layers=layers,
        )
        mask = np.zeros(self._bev_shape[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in objects[layer]:
                polygon: Polygon = self._geometry_local_coords(
                    map_object.polygon,
                    ego_pose,
                )
                exterior = np.asarray(polygon.exterior.coords).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [self._coords_to_pixel(exterior)], color=255)
        return self._opencv_to_bev(mask)

    def _compute_map_linestring_mask(
        self,
        map_api: AbstractMap,
        ego_pose: StateSE2,
        layers: List[SemanticMapLayer],
    ) -> np.ndarray:
        objects = map_api.get_proximal_map_objects(
            point=ego_pose.point,
            radius=self.config.bev_radius,
            layers=layers,
        )
        mask = np.zeros(self._bev_shape[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in objects[layer]:
                linestring: LineString = self._geometry_local_coords(
                    map_object.baseline_path.linestring,
                    ego_pose,
                )
                points = np.asarray(linestring.coords).reshape((-1, 1, 2))
                cv2.polylines(
                    mask,
                    [self._coords_to_pixel(points)],
                    isClosed=False,
                    color=255,
                    thickness=self.config.centerline_thickness,
                )
        return self._opencv_to_bev(mask)

    def _compute_box_mask(
        self,
        annotations: Annotations,
        included_types: List[TrackedObjectType],
    ) -> np.ndarray:
        mask = np.zeros(self._bev_shape[::-1], dtype=np.uint8)
        for name, box in zip(annotations.names, annotations.boxes):
            if name not in tracked_object_types:
                continue
            if tracked_object_types[name] not in included_types:
                continue

            x, y, heading = box[0], box[1], box[-1]
            length, width, height = box[3], box[4], box[5]
            oriented_box = OrientedBox(
                StateSE2(x, y, heading),
                length,
                width,
                height,
            )
            exterior = np.asarray(oriented_box.geometry.exterior.coords).reshape(
                (-1, 1, 2)
            )
            cv2.fillPoly(mask, [self._coords_to_pixel(exterior)], color=255)
        return self._opencv_to_bev(mask)

    @staticmethod
    def _geometry_local_coords(geometry: Any, origin: StateSE2) -> Any:
        cosine = np.cos(origin.heading)
        sine = np.sin(origin.heading)
        translated = affinity.affine_transform(
            geometry,
            [1, 0, 0, 1, -origin.x, -origin.y],
        )
        return affinity.affine_transform(
            translated,
            [cosine, sine, -sine, cosine, 0, 0],
        )

    def _coords_to_pixel(self, coords: np.ndarray) -> np.ndarray:
        pixel_center = np.asarray([[0, self.config.bev_pixel_width / 2.0]])
        return ((coords / self.config.bev_pixel_size) + pixel_center).astype(np.int32)

    @staticmethod
    def _opencv_to_bev(mask: np.ndarray) -> np.ndarray:
        return np.rot90(mask)[::-1] > 0
