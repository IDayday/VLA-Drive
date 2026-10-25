"""Frozen offline target extractors with exact, independently audited weights.

These modules are deliberately absent from the deployment model. There is no
random-projector fallback and no use of a full-clip history slice as observation.
"""
from __future__ import annotations

import contextlib
import importlib
import inspect
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

from .teacher_targets import (CAMERAS, IMAGE_MEAN, IMAGE_STD, DINOTargetBatch,
    WAFutureTargetBatch, file_sha256, identity_hash, validate_clip_timestamps,clip_timing_valid)

DINO_REPOSITORY = 'timm/vit_large_patch16_dinov3.lvd1689m'
DINO_REVISION = '30c1109559f65dea34316b0d4842d35c5771fe11'
DINO_WEIGHT_SHA256 = '45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941'
WA_REPOSITORY = 'AFARI-Research/WA-JEPA'
WA_REVISION = '15c0770ebd233665214590bb2190a90907499f9e'
WA_SOURCE_COMMIT = 'bec29660f5ea46ac73db8e2d0c33c8d1a72c23ad'
WA_WEIGHT_SHA256 = 'e1d3be013de7dbaa5a6c30925a0d2f91e6ce1a6b17a75ca73fe01a1abe6d6408'
WA_SOURCE_FILES = {
    'models/multiview_causal_future_jepa.py': 'f05b1984bfc0d3e291ce06cf277f9317d2c17d0fa1eb70be4a915f1e152f1d1e',
    'models/jepa_common.py': '15c424c5f134377c86353a2aba7546a086ba04ea378978430be2e89b3cf59050',
    'third_party/vjepa2/models/vision_transformer.py': 'f2f77f07b5daf66287f33d0f95c911f071b753d706a5902889e8ef42b8d93f90',
    'third_party/vjepa2/models/utils/modules.py': '4f691b1dfbcce010aad54d9306134b03e7e82c4c1ee47913fd8ebd06e5815439',
    'third_party/vjepa2/models/utils/patch_embed.py': '29e11ab97ab3ccdef107d6a7d0d7b374b58e712076cc3561f07b7e603c9b5165',
    'configs/causal_future_jepa_mv_navsim.yaml': '788c7f405f71f4cca867328a0879c64c25d49287bcde30b930c0b839f18f256f',
    'LICENSE': 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4',
}


class FrozenTargetModule(nn.Module):
    def train(self, mode=True):
        # Outer training mode must not enable teacher dropout/buffer updates.
        return super().train(False)


def _dtype_name(value):
    return str(value).removeprefix('torch.')


def _numeric_identity():
    return dict(torch=torch.__version__, allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        allow_tf32_cudnn=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


class DINOCurrentTeacher(FrozenTargetModule):
    def __init__(self, model_root, *, device='cpu', compute_dtype=torch.float32):
        super().__init__()
        import timm
        from safetensors.torch import load_file
        root = Path(model_root)
        path = root/'model.safetensors'
        if file_sha256(path) != DINO_WEIGHT_SHA256:
            raise ValueError('DINO weight hash differs from the pinned ViT-L/16 artifact')
        with torch.random.fork_rng(devices=[]):
            self.encoder = timm.create_model('vit_large_patch16_dinov3', pretrained=False,
                                            num_classes=0, img_size=(288, 512), dynamic_img_size=True)
        self.encoder.load_state_dict(load_file(str(path)), strict=True)
        self.encoder.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        if tuple(self.encoder.patch_embed.patch_size) != (16, 16) or self.encoder.num_features != 1024:
            raise ValueError('Not the registered DINOv3 ViT-L/16')
        self.prefix_tokens = int(self.encoder.num_prefix_tokens)
        self.compute_dtype = compute_dtype
        if compute_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError('Unregistered DINO compute precision')
        self.teacher_identity = dict(schema='world_postcot_DINO_current_v1', repository=DINO_REPOSITORY,
            revision=DINO_REVISION, checkpoint=str(path.resolve()), weight_sha256=DINO_WEIGHT_SHA256,
            implementation='timm', timm=timm.__version__, input_hw=[288,512],
            native_grid=[18,32], pooled_grid=[9,16], channels=1024, camera_names=list(CAMERAS),
            preprocessing='PIL RGB full-field bicubic resize512x288; ImageNet once; no crop/pad',
            feature='final post-norm dense patches; CLS/register prefix removed',
            prefix_tokens=self.prefix_tokens, extra_token_L2_normalization=False,
            pooling='fixed mean2x2 after native feature extraction',
            parameter_dtype='float32', compute_dtype=_dtype_name(compute_dtype),
            source_sha256=__import__('hashlib').sha256(inspect.getsource(type(self.encoder).forward_features).encode()).hexdigest(),
            numerical_runtime=_numeric_identity(), training_only=True)
        self.teacher_identity['sha256'] = identity_hash(self.teacher_identity)
        self.train(False)

    @torch.no_grad()
    def forward(self, pixels, view_valid, timestamps_us):
        if pixels.ndim != 5 or tuple(pixels.shape[1:]) != (3,3,288,512):
            raise ValueError('DINO input must be ImageNet-normalized[B,3views,3,288,512]')
        if view_valid.shape != pixels.shape[:2] or view_valid.dtype != torch.bool:
            raise ValueError('DINO requires actual camera presence')
        device = next(self.encoder.parameters()).device
        pixels = pixels.to(device=device, dtype=torch.float32)
        view_valid = view_valid.to(device)
        selected = view_valid.flatten().nonzero().flatten()
        result = pixels.new_zeros((len(pixels)*3,9,16,1024))
        if len(selected):
            inputs = pixels.flatten(0,1).index_select(0, selected)
            if not torch.isfinite(inputs).all():
                raise ValueError('Nonfinite valid current image')
            autocast = torch.autocast(device.type, dtype=torch.bfloat16) if self.compute_dtype == torch.bfloat16 else contextlib.nullcontext()
            with autocast:
                native = self.encoder.forward_features(inputs)
            patches = native[:, self.prefix_tokens:]
            if patches.shape[1:] != (18*32,1024):
                raise ValueError('DINO native prefix/grid contract changed')
            grid = patches.reshape(len(selected),18,32,1024)
            # Pool in native returned precision, after postnorm; no extra L2.
            pooled = grid.reshape(len(selected),9,2,16,2,1024).mean((2,4))
            result[selected] = pooled.float()
        features = result.reshape(len(pixels),3,9,16,1024).detach()
        valid = view_valid[:,:,None,None].expand(len(pixels),3,9,16).clone()
        return DINOTargetBatch(features, valid, timestamps_us.to(device), self.teacher_identity).validate()


def _official_wa_modules(source_root):
    root = Path(source_root).resolve()
    required = ['models/multiview_causal_future_jepa.py', 'models/jepa_common.py',
                'third_party/vjepa2/models/vision_transformer.py', 'LICENSE']
    if not all((root/name).is_file() for name in required):
        raise FileNotFoundError('Complete pinned official WA-JEPA source is required')
    for name, expected in WA_SOURCE_FILES.items():
        if file_sha256(root/name) != expected:
            raise ValueError('WA-JEPA source does not match pinned commit: '+name)
    existing = sys.modules.get('models')
    if existing is not None:
        path = getattr(existing, '__file__', None)
        if path and not Path(path).resolve().is_relative_to(root):
            raise ImportError('Refuse another repository\'s generic models package')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module('models.multiview_causal_future_jepa')
    common = importlib.import_module('models.jepa_common')
    encoder = importlib.import_module('third_party.vjepa2.models.vision_transformer')
    for imported in [module, common, encoder]:
        if not Path(imported.__file__).resolve().is_relative_to(root):
            raise ImportError('WA-JEPA import escaped the declared source tree')
    return module, common, encoder


class WAJEPATargetTeacher(FrozenTargetModule):
    def __init__(self, source_root, checkpoint, *, device='cpu', compute_dtype=torch.bfloat16):
        super().__init__()
        checkpoint = Path(checkpoint)
        if file_sha256(checkpoint) != WA_WEIGHT_SHA256:
            raise ValueError('WA-JEPA checkpoint differs from the official pinned release')
        if compute_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError('WA teacher compute precision must be explicit BF16 or FP32')
        official, common, encoder_factory = _official_wa_modules(source_root)
        self._chunk_layer_norm = common._chunk_layer_norm
        with torch.random.fork_rng(devices=[]):
            self.target_encoder = encoder_factory.vit_large(img_size=(256,512), patch_size=16,
                num_frames=12, tubelet_size=2, use_rope=True, use_sdpa=True, use_silu=False,
                wide_silu=True, use_activation_checkpointing=False, interpolate_rope=True,
                modality_embedding=True, img_temporal_dim_size=1, pretrained_img_size=(256,256))
            self.target_scene_projector = official.SceneTokenProjector(target_dim=4096, scene_dim=512)
        state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        if not isinstance(state, dict) or not all(isinstance(k,str) for k in state):
            raise ValueError('Unknown official checkpoint state container')
        if all(k.startswith('module.') for k in state):
            state = {k.removeprefix('module.'): v for k,v in state.items()}
        coverage = {}
        for prefix, target in [('target_encoder.',self.target_encoder),
                               ('target_scene_projector.',self.target_scene_projector)]:
            extracted = {k[len(prefix):]: v for k,v in state.items() if k.startswith(prefix)}
            if not extracted:
                raise ValueError('Official trained teacher submodule is missing: '+prefix)
            target.load_state_dict(extracted, strict=True)
            coverage[prefix[:-1]] = dict(loaded_tensors=len(extracted), expected_tensors=len(target.state_dict()),
                                        missing=[],unexpected=[],strict=True)
        if self.target_encoder.embed_dim != 1024 or len(self.target_encoder.hierarchical_layers) != 4:
            raise ValueError('Official hierarchical feature space is not4096D')
        self.target_encoder.to(device=device,dtype=compute_dtype).eval().requires_grad_(False)
        self.target_scene_projector.to(device=device,dtype=compute_dtype).eval().requires_grad_(False)
        self.register_buffer('image_mean', torch.tensor(IMAGE_MEAN,device=device,dtype=compute_dtype).view(1,3,1,1,1), persistent=False)
        self.register_buffer('image_std', torch.tensor(IMAGE_STD,device=device,dtype=compute_dtype).view(1,3,1,1,1), persistent=False)
        self.teacher_identity = dict(schema='world_postcot_WA_future_v1', repository=WA_REPOSITORY,
            revision=WA_REVISION, checkpoint=str(checkpoint.resolve()), weight_sha256=WA_WEIGHT_SHA256,
            source_commit=WA_SOURCE_COMMIT, source_root=str(Path(source_root).resolve()),
            source_files={name:file_sha256(Path(source_root)/name) for name in
                ['models/multiview_causal_future_jepa.py','models/jepa_common.py',
                 'third_party/vjepa2/models/vision_transformer.py','configs/causal_future_jepa_mv_navsim.yaml','LICENSE']},
            checkpoint_coverage=coverage, input_hw=[256,512], history_frames=4, future_frames=8,
            frame_rate_hz=2., tubelet_size=2, future_tubelets=4, raw_hierarchical_dim=4096,
            projected_dim=512, native_grid=[16,32], pooled_grid=[8,16], camera_names=list(CAMERAS),
            released_predictor_camera_names=['cam_l0','cam_f0','cam_r0','cam_b0'],
            camera_independence='official encoder flattens[B,V] into independent batch; shared projector is tokenwise; predictor not loaded',
            preprocessing='cv2 BGR→RGB INTER_AREA512x256; dataset[-1,1]; teacher ImageNet conversion',
            extraction_order=['target_encoder(training=True API, module.eval)',
                'official hierarchical chunk LayerNorm1024', 'future slice excluding2history tubelets',
                'strictly loaded target_scene_projector4096→512', 'fixed spatial mean2x2'],
            compute_dtype=_dtype_name(compute_dtype), numerical_runtime=_numeric_identity(),
            full_clip_history_tokens_never_returned=True, training_only=True,
            public_checkpoint_training_data='official NAVSIM stage2 uses navtrain with hash-log5percent holdout; overlap with internal navtrain DEV is possible')
        self.teacher_identity['sha256'] = identity_hash(self.teacher_identity)
        self.train(False)

    @torch.no_grad()
    def forward(self, clip_rgb, frame_valid, timestamps_us):
        if clip_rgb.ndim != 6 or tuple(clip_rgb.shape[1:]) != (12,3,3,256,512):
            raise ValueError('WA input requires[B,12frames,3views,RGB,256,512]')
        if frame_valid.shape != clip_rgb.shape[:3] or frame_valid.dtype != torch.bool:
            raise ValueError('Every actual frame/view needs presence metadata')
        tubelets = validate_clip_timestamps(timestamps_us,allow_incomplete=True)
        device = self.image_mean.device
        dtype = self.image_mean.dtype
        valid_views = (frame_valid.all(1) & clip_timing_valid(timestamps_us)[:,None].to(frame_valid.device)).to(device)
        selected = valid_views.flatten().nonzero().flatten()
        result = torch.zeros((len(clip_rgb)*3,4,8,16,512), device=device,dtype=dtype)
        if len(selected):
            valid_pixels = clip_rgb.permute(0,2,3,1,4,5).reshape(len(clip_rgb)*3,3,12,256,512)
            valid_pixels = valid_pixels.index_select(0, selected.to(valid_pixels.device))
            if not torch.isfinite(valid_pixels).all() or (valid_pixels < 0).any() or (valid_pixels > 1).any():
                raise ValueError('WA input must be finite, unnormalized RGB in[0,1]')
            # Reproduce the official dataset→prepare_batch→_make_full_clip
            # numerical order. No teacher history feature reaches the actor.
            minus_one = clip_rgb.to(device=device,dtype=torch.float32).mul(2).sub(1).clamp(-1,1).to(dtype)
            clip = minus_one.permute(0,2,3,1,4,5).reshape(len(clip_rgb)*3,3,12,256,512).index_select(0,selected)
            clip = (clip.add(1).mul(.5).clamp(0,1)-self.image_mean)/self.image_std
            hierarchical = self.target_encoder(clip, training=True)
            normalized = self._chunk_layer_norm(hierarchical,self.target_encoder.embed_dim)
            if normalized.shape[1:] != (6*16*32,4096):
                raise ValueError('Official teacher temporal/spatial flattening changed')
            future = normalized[:,2*16*32:]
            # The official projector's API retains an explicit view axis even
            # though its LayerNorm/Linear are tokenwise. Independent selected
            # views are batch entries, each with one real view; do not bypass
            # or replace the strictly loaded official projector.
            projected = self.target_scene_projector(future[:,None])[:,0].reshape(len(selected),4,16,32,512)
            pooled = projected.reshape(len(selected),4,8,2,16,2,512).mean((3,5))
            result[selected] = pooled
        features = result.reshape(len(clip_rgb),3,4,8,16,512).detach()
        valid = valid_views[:,:,None,None,None].expand(len(clip_rgb),3,4,8,16).clone()
        return WAFutureTargetBatch(features,valid,timestamps_us.to(device),tubelets.to(device),self.teacher_identity).validate()
