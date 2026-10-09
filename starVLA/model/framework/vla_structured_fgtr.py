"""Qwen + original DDP FM + single-frame structured world + one FGTR residual."""
from dataclasses import asdict
import torch
from .DDPForesight import DDPForesight
from starVLA.model.modules.vehicle_joint.initialization import initialization_seed
from starVLA.model.modules.foresight.dino_feature_head import DINOFeatureHead
from starVLA.model.modules.foresight.losses import masked_regression
from starVLA.model.modules.structured_world.grid import GridSpec
from starVLA.model.modules.structured_world.geo_bev import SingleFrameGeoBEV
from starVLA.model.modules.structured_world.future import SingleFrameFuture
from starVLA.model.modules.structured_world.fgtr import DDPProposalFGTR, sample_proposal
from starVLA.model.modules.structured_world.semantics import StructuredSemantics
from starVLA.model.modules.structured_world.queries import sample_queries
from starVLA.model.modules.structured_world.losses import compute_structured_losses, final_trajectory_loss, scene_reduce
from starVLA.dataloader.foresight_dataset import decode_ego


REGISTERED_GROUPS = {
    'G0_NO_FUTURE_LABEL': ('none', 'uniform'),
    'G1_FULL_UNIFORM': ('full', 'uniform'),
    'G2_EVENT_UNIFORM': ('event', 'uniform'),
    'G3_EVENT_LOCAL': ('event', 'local'),
}


class VLAStructuredFGTR(DDPForesight):
    def __init__(self, config, accelerator=None):
        options = dict(config.structured_world)
        group = options['group']
        if group not in REGISTERED_GROUPS:
            raise ValueError('A preregistered first-round group is required')
        self.semantic_mode, self.query_mode = REGISTERED_GROUPS[group]
        cameras = int(config.framework.camera_count)
        steps = int(config.framework.action_model.action_horizon)
        if (cameras, steps) not in ((3, 8), (6, 6)):
            raise ValueError('NAVSIM 3/8 or nuScenes 6/6 contract required')
        if int(config.framework.action_model.repeated_diffusion_steps) != 8:
            raise ValueError('Original FM repeat is fixed to eight')
        if options.get('initialization_kind', 'generic') != 'generic':
            raise ValueError('Formal groups cannot initialize from historical driving weights')
        if config.foresight.enable_future_dino or config.foresight.enable_interaction:
            raise ValueError('Old future DINO and GT-MAE are retained separately, not new run dependencies')
        super().__init__(config, accelerator)
        self.grid = GridSpec(**dict(options.get('grid', {})))
        self.steps, self.cameras = steps, cameras
        self.structured_options = options
        hidden = self.qwen_vl_interface.model.config.hidden_size
        # Independent initialization forks leave the generic VLM/original DDP
        # initialization and all global FM random streams untouched.
        if self.foresight_config.enable_current_dino:
            with initialization_seed(int(config.seed)+2300):
                self.dino_head = DINOFeatureHead(hidden, self.foresight_config.dino_feature_dim,
                    self.foresight_config.readout_dim, self.foresight_config.readout_layers, cameras=cameras)
        with initialization_seed(int(config.seed)+4100):
            self.geometry = SingleFrameGeoBEV(self.grid, tuple(options.get('geometry_image_size', (256, 448))),
                                               imagenet_checkpoint=options['imagenet_checkpoint'])
        with initialization_seed(int(config.seed)+4200):
            self.future_space = SingleFrameFuture(hidden, steps, grid=self.grid)
        with initialization_seed(int(config.seed)+4300):
            self.refiner = DDPProposalFGTR(hidden, steps, grid=self.grid)
        with initialization_seed(int(config.seed)+4400):
            self.scene_semantics = StructuredSemantics(self.semantic_mode)
        self.shared_geometry_identity = None
        self.initialization_contract = {'VLM': 'generic Qwen public pretrained',
            'action_ego_W_and_driving_modules': 'seeded random, original DDP unchanged',
            'geometry_backbone': self.geometry.initialization['backbone'],
            'grid': asdict(self.grid), 'group': group, 'steps': steps, 'cameras': cameras}

    def _build_qwen_batch(self, examples, instructions):
        # The parent extracts frozen vision features before encode_current's
        # language autocast block. FP32 master parameters otherwise make that
        # vision forward silently FP32. Match the registered BF16 Qwen compute
        # contract while retaining FP32 parameters; canonical inference uses
        # amp()'s existing inference_fp32 bypass.
        with self.amp():
            return super()._build_qwen_batch(examples, instructions)

    def load_shared_geometry(self, path, expected_identity, *, allow_debug=False):
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state['identity'] != expected_identity or state['dataset'] != self.structured_options['dataset']:
            raise ValueError('Shared perception checkpoint population/identity mismatch')
        if state['training_task'] != 'current_depth_road_occupancy_no_planner':
            raise ValueError('Geometry preparation must not contain a driving planner')
        if not allow_debug and state['population'] != 'full_train_population':
            raise ValueError('Debug geometry cannot initialize formal groups')
        self.geometry.load_state_dict(state['geometry'], strict=True)
        self.scene_semantics.current.load_state_dict(state['current_head'], strict=True)
        self.shared_geometry_identity = expected_identity
        self.geometry.initialization['geometry_supervised_pretraining'] = expected_identity

    def geometric_current(self, observations):
        device = next(self.geometry.parameters()).device
        images = torch.stack([item['geometry_images'] for item in observations]).to(device=device, dtype=torch.float32)
        calibration = {key: torch.stack([item['calibration'][key] for item in observations]).to(device=device, dtype=torch.float32)
                       for key in ('sensor2ego', 'intrinsics', 'post_rots', 'post_trans')}
        # No map, box, LiDAR/depth label, or global pose can enter this call.
        pixel_valid = torch.stack([item['geometry_pixel_valid'] for item in observations]).to(device)
        return self.geometry(images, calibration, pixel_valid)

    def plan_from_current(self, observations, *, proposal_generator=None, initial_noise=None, encoded=None, H_A=None):
        encoded = self.encode_current(observations) if encoded is None else encoded
        H_A = self.build_planner_condition(encoded) if H_A is None else H_A
        with self.amp():
            q0 = sample_proposal(self.action_model, H_A, generator=proposal_generator, initial_noise=initial_noise)
        q0_physical = decode_ego(q0).detach()
        current = self.geometric_current(observations)
        # These paths are deliberately outside the proposal no_grad block.
        future = self.future_space(current['B0'], H_A, q0_physical)
        corrected = self.refiner(H_A, future, q0, q0_physical)
        return {'encoded': encoded, 'H_A': H_A, 'current': current, 'Bt': future,
                'q0_encoded': q0, 'q0': q0_physical, **corrected,
                'q_final': decode_ego(corrected['q_final_encoded'])}

    def forward(self, observations, targets, *, completed_updates=0, noise_generator=None,
                time_generator=None, proposal_generator=None, query_generator=None, global_counts=None):
        if any(generator is None for generator in (noise_generator, time_generator, proposal_generator, query_generator)):
            raise ValueError('FM noise/time, proposal and auxiliary sampling require independent RNG streams')
        encoded = self.encode_current(observations)  # exactly one current Qwen call
        action = self.build_planner_condition(encoded)
        device = action.device
        targets = {key: value.to(device) if torch.is_tensor(value) else value for key, value in targets.items()}
        ego = targets['ego'].float()
        if ego.shape != (len(observations), self.steps, 4) or not torch.isfinite(ego).all():
            raise ValueError('Invalid original ego target')
        if not targets['future_valid'].all():
            raise ValueError('Formal FM population requires complete future; identical eligibility in all groups')
        repeat = int(self.config.framework.action_model.repeated_diffusion_steps)
        y = ego.repeat(repeat, 1, 1)
        cfg = self.config.framework.action_model
        if cfg.noise_beta_beta != 1.:
            raise ValueError('Original explicit Beta(alpha,1) time contract required')
        noise = torch.randn(y.shape, device=device, dtype=y.dtype, generator=noise_generator)
        u = torch.rand((len(y),), device=device, generator=time_generator)
        times = (cfg.noise_s-u.pow(1./cfg.noise_beta_alpha))/cfg.noise_s
        with self.amp():
            fm = self.action_model(action.repeat(repeat, 1, 1), y, noise=noise, times=times)
        counts = global_counts or {}
        scenes = counts.get('ego_scenes')
        losses = {'original_FM': scene_reduce(fm.expand(len(ego)), global_scenes=scenes)}
        raw = {'original_FM': losses['original_FM'].detach()}
        if hasattr(self, 'dino_head'):
            cfg = self.foresight_config
            values, valid = targets['current_dino'], targets['current_dino_valid']
            expected = (len(ego), self.cameras, cfg.dino_feature_dim, cfg.dino_height, cfg.dino_width)
            if values.shape != expected or valid.shape != (len(ego), self.cameras, cfg.dino_height, cfg.dino_width):
                raise ValueError('Dataset-current DINO cache identity/layout mismatch')
            with self.amp():
                prediction = self.dino_head(encoded['W'], torch.zeros(len(ego), device=device),
                                             (cfg.dino_height, cfg.dino_width))
            dino, count = masked_regression(prediction, values, valid[:, :, None], global_count=counts.get('current_dino'))
            losses['current_DINO'] = dino  # common fixed weight1.0, no extra repeat
            raw['current_DINO'] = dino.detach()
        planned = self.plan_from_current(observations, encoded=encoded, H_A=action, proposal_generator=proposal_generator)
        gradient_norms = {}
        for name, value in (('H_A', planned['H_A']), ('B0', planned['current']['B0']), ('Bt', planned['Bt'])):
            if value.requires_grad:
                def record_gradient(gradient, key=name):
                    gradient_norms[key] = float(torch.linalg.vector_norm(gradient.detach().float()))
                value.register_hook(record_gradient)
        semantic = self.scene_semantics(planned['current']['B0'], planned['Bt'])
        query = sample_queries(planned['q0'], targets, mode=self.query_mode, generator=query_generator,
                                count=int(self.structured_options.get('queries', 1024)), grid=self.grid)
        aux, effective = compute_structured_losses(semantic, targets, query, targets['ego_bodies'],
                            semantic_mode=self.semantic_mode, statistics=self.structured_options['class_statistics'],
                            global_scenes=scenes, grid=self.grid)
        # Depth supervision is loss-only and scene-normalized, with no-return
        # pixels excluded by the pinned RCSample depth-label implementation.
        depth_scenes = []
        depth = planned['current']['depth']
        for row in range(len(ego)):
            selected = [value[row*self.cameras:(row+1)*self.cameras] for value in depth]
            depth_scenes.append(self.geometry.view.get_depth_loss(targets['depth'][row:row+1], selected))
        aux['depth'] = scene_reduce(torch.stack(depth_scenes), global_scenes=scenes)
        refine, effective_refine = final_trajectory_loss(planned['q_final_encoded'], ego, targets['future_valid'], global_scenes=scenes)
        weights = self.structured_options['loss_weights']
        losses['current_geometry'] = float(weights['geometry'])*(aux['depth']+aux['road']+aux['current_occupancy'])
        losses['local_relations'] = float(weights['query'])*aux['query_relations']
        if self.semantic_mode != 'none':
            losses['future_semantics'] = float(weights['future'])*aux['future_semantics']
        else:
            # Keep G0's exact readout structure and graph, without future GT loss.
            # A zero anchor also makes empty-label/DDP behavior explicit.
            losses['future_semantics'] = semantic['future_logits'].sum()*0.
        warmup = min(1., (completed_updates+1)/int(self.structured_options['refine_warmup_updates']))
        losses['refine'] = float(weights['refine'])*warmup*refine
        raw.update({k: value.detach() for k, value in aux.items()}, refine=refine.detach())
        return {'loss': sum(losses.values()), 'losses': losses,
                'metrics': {'raw': raw, 'effective_counts': effective, 'activation_gradient_norms': gradient_norms,
                    'refine_effective_weight': float(weights['refine'])*warmup,
                    'query_strata': query['strata_records'], 'proposal_out_of_range': planned['out_of_range'].detach()},
                'diagnostics': {'q0': planned['q0'].detach(), 'q_final': planned['q_final'].detach()}}

    @torch.no_grad()
    def predict_action(self, observations, *, sampling_seed=42, initial_noise=None, return_diagnostics=False):
        device = next(self.parameters()).device
        generator = torch.Generator(device=device).manual_seed(sampling_seed)
        planned = self.plan_from_current(observations, proposal_generator=generator, initial_noise=initial_noise)
        if return_diagnostics:
            return {key: planned[key] for key in ('q0', 'q_final', 'out_of_range', 'reference_points')}
        return planned['q_final']

    def strip_auxiliary_heads(self):
        if hasattr(self, 'dino_head'):
            del self.dino_head
        del self.scene_semantics  # All field heads are supervision/diagnostic only.
        # Geometry, future-space generation and actual FGTR remain executable.
        return self
