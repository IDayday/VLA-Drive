"""Calibrated resize/crop and fixed current-camera input contracts."""
import cv2
import numpy as np
import torch
from PIL import Image

NAVSIM_CAMERAS = ('CAM_F0', 'CAM_L0', 'CAM_R0')
NUSCENES_CAMERAS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')


def crop_16_9(width, height):
    if width/height > 16/9:
        cropped_width = int(height*16/9)
        return ((width-cropped_width)//2, 0, (width+cropped_width)//2, height)
    if width/height < 16/9:
        cropped_height = int(width*9/16)
        return (0, (height-cropped_height)//2, width, (height+cropped_height)//2)
    return (0, 0, width, height)


def camera_inputs(paths, sensor2ego, intrinsics, distortion=None, *, image_size=(256, 448)):
    """Geometry is rectified with the original K; Qwen retains its locked ROI.

    Rectification never invents rays outside the source sensor: the validity
    mask is the remapped all-one original image and excludes padded borders.
    LiDAR supervision uses this same rectified pixel/calibration contract.
    """
    if len(paths) not in (3, 6):
        raise ValueError('Registered three/six current views required')
    distortion = [None]*len(paths) if distortion is None else distortion
    geometry, qwen, valid, post_rots, post_trans = [], [], [], [], []
    height, width = image_size
    for path, K, coefficients in zip(paths, intrinsics, distortion):
        with Image.open(path) as source:
            image = source.convert('RGB')
            left, top, right, bottom = crop_16_9(*image.size)
            qwen.append(image.crop((left, top, right, bottom)).resize((1024, 576), Image.Resampling.LANCZOS))
            pixels = np.asarray(image)
        K = np.asarray(K, dtype=np.float64)
        supported = np.ones(pixels.shape[:2], dtype=np.uint8)
        if coefficients is not None and np.any(coefficients):
            mx, my = cv2.initUndistortRectifyMap(K, np.asarray(coefficients), None, K,
                                                (pixels.shape[1], pixels.shape[0]), cv2.CV_32FC1)
            pixels = cv2.remap(pixels, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            supported = cv2.remap(supported, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        pixels = cv2.resize(pixels[top:bottom, left:right], (width, height), interpolation=cv2.INTER_LINEAR)
        supported = cv2.resize(supported[top:bottom, left:right], (width, height), interpolation=cv2.INTER_NEAREST)
        # Standard ImageNet normalization, same for both datasets.
        tensor = torch.from_numpy(np.moveaxis(pixels.copy(), -1, 0)).float()/255.
        tensor = (tensor-tensor.new_tensor([.485, .456, .406])[:, None, None])/tensor.new_tensor([.229, .224, .225])[:, None, None]
        geometry.append(tensor)
        valid.append(torch.from_numpy(supported.astype(bool)))
        sx, sy = width/(right-left), height/(bottom-top)
        post_rots.append(np.diag([sx, sy, 1.]))
        post_trans.append([-sx*left + .5*sx-.5, -sy*top + .5*sy-.5, 0.])
    calibration = {'sensor2ego': torch.as_tensor(np.asarray(sensor2ego), dtype=torch.float32),
                   'intrinsics': torch.as_tensor(np.asarray(intrinsics), dtype=torch.float32),
                   'post_rots': torch.as_tensor(np.asarray(post_rots), dtype=torch.float32),
                   'post_trans': torch.as_tensor(np.asarray(post_trans), dtype=torch.float32)}
    return {'image': qwen, 'geometry_images': torch.stack(geometry),
            'calibration': calibration, 'geometry_pixel_valid': torch.stack(valid)}


def projected_depth(points_ego, calibration, pixel_valid, *, image_size=(256, 448)):
    """Nearest positive LiDAR depth per rectified pixel; zero means unlabeled."""
    points = np.asarray(points_ego, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('Expected current LiDAR points in ego(t0)')
    height, width = image_size
    labels = []
    for i in range(len(calibration['intrinsics'])):
        sensor = calibration['sensor2ego'][i].numpy().astype(np.float64)
        camera = (points-sensor[:3, 3]) @ sensor[:3, :3]
        projected = camera @ calibration['intrinsics'][i].numpy().T
        front = np.isfinite(projected).all(-1) & (camera[:, 2] > 0.)
        projected = projected[front]
        depth = camera[front, 2]
        projected[:, :2] /= projected[:, 2:3]
        projected = projected @ calibration['post_rots'][i].numpy().T + calibration['post_trans'][i].numpy()
        pixel = np.rint(projected[:, :2]).astype(np.int64)
        inside = ((pixel[:, 0] >= 0) & (pixel[:, 0] < width) & (pixel[:, 1] >= 0) & (pixel[:, 1] < height))
        pixel, depth = pixel[inside], depth[inside]
        buffer = np.full(height*width, np.inf, dtype=np.float32)
        np.minimum.at(buffer, pixel[:, 1]*width+pixel[:, 0], depth)
        buffer = buffer.reshape(height, width)
        buffer[~np.isfinite(buffer) | ~pixel_valid[i].numpy()] = 0.
        labels.append(torch.from_numpy(buffer))
    return torch.stack(labels)
