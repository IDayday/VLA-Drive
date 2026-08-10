"""Field2Plan MVP modules.

The package has no import-time model loading and no dependency on external
teachers.  All modules are device agnostic and inherit device/dtype from their
inputs.
"""

from .trajectory_codec import TrajectoryCodec
from .geometry_teachers import (
    DA3LegacyDepthAdapter,
    GeometryTeacherAdapter,
    GeometryTeacherSample,
    OfficialVGGTMetricDepthAdapter,
    VGGTAdapter,
    estimate_metric_scale_from_depth_reference,
    estimate_metric_scale_from_camera_rig,
)
from .geometry_supervision import (
    GeometryPrediction,
    GeometrySupervisionHead,
    GeometryTargets,
    build_geometry_targets,
    geometry_supervision_losses,
)
from .controls import (
    ControlledDynamicsTeacher,
    ControlledGeometryTeacher,
    GTMLPFieldControl,
    apply_dynamics_teacher_controls,
    apply_geometry_teacher_controls,
)
from .dynamics_field_writer import ActionFreeDynamicsFieldWriter
from .dynamics_supervision import (
    DynamicsPrediction,
    DynamicsSupervisionHead,
    DynamicsTargets,
    build_dynamics_targets,
    dynamics_supervision_losses,
)
from .dynamics_teachers import (
    DynamicsTeacherAdapter,
    DynamicsTeacherSample,
    OfficialVJEPA2Adapter,
)
from .temporal_alignment import (
    build_temporal_alignment,
    interpolate_temporal_features,
    se2_poses_to_transforms,
)
from .trajectory_refiner import TrajectoryRefiner
from .trajectory_tube_reader import TrajectoryTubeReader

__all__ = [
    "TrajectoryCodec",
    "TrajectoryRefiner",
    "TrajectoryTubeReader",
    "GeometryTeacherAdapter",
    "GeometryTeacherSample",
    "DA3LegacyDepthAdapter",
    "OfficialVGGTMetricDepthAdapter",
    "VGGTAdapter",
    "estimate_metric_scale_from_depth_reference",
    "estimate_metric_scale_from_camera_rig",
    "GeometryPrediction",
    "GeometryTargets",
    "GeometrySupervisionHead",
    "build_geometry_targets",
    "geometry_supervision_losses",
    "ControlledGeometryTeacher",
    "ControlledDynamicsTeacher",
    "GTMLPFieldControl",
    "apply_geometry_teacher_controls",
    "apply_dynamics_teacher_controls",
    "ActionFreeDynamicsFieldWriter",
    "DynamicsPrediction",
    "DynamicsTargets",
    "DynamicsSupervisionHead",
    "build_dynamics_targets",
    "dynamics_supervision_losses",
    "DynamicsTeacherAdapter",
    "DynamicsTeacherSample",
    "OfficialVJEPA2Adapter",
    "se2_poses_to_transforms",
    "build_temporal_alignment",
    "interpolate_temporal_features",
]
