"""Center semantics around mature bottom-origin, upright LiDAR boxes.

The official rotate/translate and UniAD reframe/raster/track operators remain
the implementation. This wrapper corrects only the vertical half-height shift
when a full rigid rotation includes sensor pitch/roll. Boxes remain upright;
their unrepresented roll/pitch is a disclosed planar-footprint approximation.
"""
import numpy as np
import torch
from third_party.uniad.ported.occflow_label import GenerateOccFlowLabels


def correct_center_after_rotation(boxes, rotation):
    matrix = boxes.tensor.new_tensor(np.asarray(rotation))
    half_height = boxes.tensor[:, 5:6]/2.
    displacement = half_height*(matrix[:, 2]-matrix.new_tensor([0., 0., 1.]))
    boxes.translate(displacement)
    return boxes


class CenteredOccFlowLabels(GenerateOccFlowLabels):
    def reframe_boxes(self, boxes, t_init, t_curr):
        # The base caller already clones; preserve that isolation explicitly.
        boxes = boxes.clone()
        before = boxes.gravity_center.clone()
        result = super().reframe_boxes(boxes, t_init, t_curr)
        def rigid(data, prefix):
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, :3] = data[prefix+'_r']
            matrix[:3, 3] = data[prefix+'_t']
            return matrix
        # Same transforms as the original UniAD method, collapsed only for
        # its center correction. No replacement corner/raster/track operator.
        transform = np.linalg.inv(rigid(t_init, 'l2e'))@np.linalg.inv(rigid(t_init, 'e2g'))@rigid(t_curr, 'e2g')@rigid(t_curr, 'l2e')
        matrix = before.new_tensor(transform)
        desired = before@matrix[:3, :3].T+matrix[:3, 3]
        result.translate(desired-result.gravity_center)
        return result
