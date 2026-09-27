"""Current predicted-box evidence proxies; projection support is NOT visibility."""
from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass(frozen=True)
class CurrentObservation:
    token: str
    intrinsics: np.ndarray          # V,3,3 AFTER declared crop/resize
    camera_to_ego: np.ndarray       # V,4,4
    image_transforms: np.ndarray    # V,3,3 raw -> actual input pixels
    distortion: np.ndarray         # V,5 Brown-Conrady
    camera_names: tuple
    image_size: tuple               # width,height
    timestamp: int
    camera_timestamps: tuple
    ego_speed_mps: float
    navigation: str
    input_identity: str

    def validate(self):
        if self.camera_names != ('CAM_F0', 'CAM_L0', 'CAM_R0'):
            raise ValueError('Only the three allowed current front views are accepted')
        for value, shape in ((self.intrinsics, (3, 3, 3)), (self.camera_to_ego, (3, 4, 4)),
                             (self.image_transforms, (3, 3, 3)), (self.distortion, (3, 5))):
            if np.shape(value) != shape or not np.isfinite(value).all():
                raise ValueError('Invalid current camera calibration')
        if len(self.camera_timestamps) != 3 or any(t > self.timestamp for t in self.camera_timestamps):
            raise ValueError('Future camera observations forbidden')
        if self.image_size != (1024, 576) or self.navigation not in ('left', 'straight', 'right', 'unknown'):
            raise ValueError('Unverified image transform/navigation contract')
        if not np.isfinite(self.ego_speed_mps) or self.ego_speed_mps < 0:
            raise ValueError('Invalid current ego speed')

    def fingerprint(self):
        self.validate()
        h = hashlib.sha256()
        for a in (self.intrinsics, self.camera_to_ego, self.image_transforms, self.distortion):
            h.update(np.asarray(a, dtype=np.float64).tobytes())
        h.update(repr((self.token, self.image_size, self.timestamp, self.camera_timestamps,
                       self.ego_speed_mps, self.navigation, self.input_identity)).encode())
        return h.hexdigest()


def box_corners(boxes):
    boxes = np.asarray(boxes, dtype=np.float64)
    signs = np.array([[-1,-1,-1],[-1,1,-1],[1,1,-1],[1,-1,-1],
                      [-1,-1,1],[-1,1,1],[1,1,1],[1,-1,1]])
    points = signs[None] * boxes[:, None, 3:6] / 2
    yaw = np.arctan2(boxes[:, 6], boxes[:, 7]); c, s = np.cos(yaw), np.sin(yaw)
    x, y = points[..., 0].copy(), points[..., 1].copy()
    points[..., 0] = c[:, None] * x - s[:, None] * y
    points[..., 1] = s[:, None] * x + c[:, None] * y
    return points + boxes[:, None, :3]


def project_points(points, intrinsic, camera_to_ego, distortion):
    inv = np.linalg.inv(camera_to_ego)
    camera = points @ inv[:3, :3].T + inv[:3, 3]
    z = camera[..., 2]; xy = camera[..., :2] / np.maximum(z[..., None], .1)
    x, y = xy[..., 0], xy[..., 1]; r2 = x*x + y*y
    # Invalid wide/behind-camera coordinates are excluded BEFORE high-order distortion.
    legal = (z > .1) & (r2 < 100) & np.isfinite(xy).all(-1)
    x = np.where(legal, x, 0); y = np.where(legal, y, 0); r2 = np.where(legal, r2, 0)
    k1, k2, p1, p2, k3 = distortion
    radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
    derivative = 1 + 3*k1*r2 + 5*k2*r2**2 + 7*k3*r2**3
    distorted = np.stack([x*radial + 2*p1*x*y + p2*(r2 + 2*x*x),
                          y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y, np.ones_like(x)], -1)
    uv = (distorted @ intrinsic.T)[..., :2]
    return uv, legal & (radial > 0) & (derivative > 0) & np.isfinite(uv).all(-1)


def measure_observability(boxes, logits, observation, config, evidence_2d=None):
    """No targets, crops from GT, object velocities, or future data are accepted.

    evidence_2d is optional CURRENT detector pixel boxes [N,V,4], already associated
    on the input side. Absent evidence remains missing, never fabricated as agreement.
    """
    observation.validate()
    boxes, logits = np.asarray(boxes, dtype=np.float64), np.asarray(logits, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 8 or logits.shape != (len(boxes), 8):
        raise ValueError('Expected current xyz/lwh/sincos boxes and seven-class logits')
    finite = np.isfinite(boxes).all(-1) & np.isfinite(logits).all(-1)
    safe = np.where(finite[:, None], boxes, np.array([0,0,0,1,1,1,0,1]))
    legal_size = (safe[:,3:6] >= [.15,.10,.15]).all(-1) & (safe[:,3:6] <= [25,8,8]).all(-1)
    plausible = finite & legal_size & (np.abs(safe[:,2]) < 5) & (np.linalg.norm(safe[:,6:8],axis=-1) > .5)
    safe_logits = np.where(finite[:, None], logits, 0)
    probabilities = np.exp(safe_logits - safe_logits.max(-1, keepdims=True))
    probabilities /= probabilities.sum(-1, keepdims=True)
    classes = probabilities[:, :7].argmax(-1)
    # Conservative physical limits; not a trained object-validity classifier.
    for ci, maximum in ((1,[3,3,3.5]), (2,[5,3,3.5]), (3,[3,3,3])):
        plausible &= (classes != ci) | (safe[:,3:6] <= maximum).all(-1)
    points = box_corners(safe); width, height = observation.image_size
    areas, truncations, heights, projected, supported, legal_views = [], [], [], [], [], []
    consistency = []
    if evidence_2d is not None and np.shape(evidence_2d) != (len(boxes), 3, 4):
        raise ValueError('Invalid current 2D evidence shape')
    for v in range(3):
        uv, valid = project_points(points, observation.intrinsics[v], observation.camera_to_ego[v], observation.distortion[v])
        valid &= plausible[:, None]
        lo = np.where(valid[..., None], uv, np.inf).min(1)
        hi = np.where(valid[..., None], uv, -np.inf).max(1)
        enough = valid.sum(-1) >= 4
        lo = np.where(enough[:, None], lo, 0); hi = np.where(enough[:, None], hi, 0)
        full = np.maximum(hi-lo,0).prod(-1)
        clo = np.maximum(lo, [0,0]); chi = np.minimum(hi, [width,height])
        extent = np.maximum(chi-clo,0); area = extent.prod(-1)
        trunc = 1-area/np.maximum(full,1e-9)
        # Any corner behind camera is severe truncation, never high reliability.
        trunc = np.where(valid.all(-1), trunc, 1.)
        overlap = enough & (area > 0)
        areas.append(area); heights.append(extent[:,1]); truncations.append(np.clip(trunc,0,1))
        projected.append(np.concatenate([lo, hi],-1)); supported.append(overlap); legal_views.append(valid.all(-1))
        if evidence_2d is not None:
            e = np.asarray(evidence_2d)[:,v]; known = np.isfinite(e).all(-1)
            e = np.where(known[:,None],e,0)
            intersection = np.maximum(np.minimum(chi,e[:,2:])-np.maximum(clo,e[:,:2]),0).prod(-1)
            union = area + np.maximum(e[:,2:]-e[:,:2],0).prod(-1)-intersection
            consistency.append(np.where(known,intersection/np.maximum(union,1e-9),np.nan))
    area, trunc, ph = np.stack(areas,1), np.stack(truncations,1), np.stack(heights,1)
    support = np.stack(supported,1); legal = np.stack(legal_views,1)
    adequate = support & legal & (area >= config['min_pixel_area']) & (ph >= config['min_pixel_height']) & (trunc <= config['max_truncation'])
    if evidence_2d is not None:
        agree = np.stack(consistency,1); adequate &= ~np.isfinite(agree) | (agree >= .25)
    else: agree = np.full(area.shape,np.nan)
    # An ordinal evidence proxy, not probability of visibility, detection, or safety.
    proxy = np.maximum(0,np.minimum(1,np.log1p(area)/np.log1p(4096)))*(1-trunc)
    proxy = np.where(legal & support,proxy,0).max(-1)
    reliable = plausible & (adequate.sum(-1) >= config['min_support_views'])
    return dict(existence_score=1-probabilities[:,-1],object_wins=probabilities.argmax(-1)!=7,
                entity_class=classes,geometry_plausible=plausible,observation_support=support,
                support_views=support.sum(-1),adequate_views=adequate.sum(-1),pixel_area=area,
                pixel_height=ph,truncation=trunc,projection_legal=legal,projected_boxes=np.stack(projected,1),
                visual_reliability=proxy,reliable=reliable,current_2d_iou=agree,
                current_2d_available=np.isfinite(agree))
