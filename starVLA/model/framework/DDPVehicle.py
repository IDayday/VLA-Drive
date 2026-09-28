"""From-generic DDP camera framework with an optional single joint action head.

The original Qwen/Wan/PPD/action modules are constructed independently per arm.
Only current observations enter encode_current; future arrays enter losses only.
"""
from contextlib import nullcontext
from dataclasses import fields
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from starVLA.model.framework.QwenOFT import Qwenvl_OFT
from starVLA.cache.navsim_feature_cache import append_world_action_tokens
from starVLA.model.modules.structured_world.contracts import WorldTargets
from starVLA.model.modules.structured_world.agent_heads import AgentHeads
from starVLA.model.modules.structured_world.scene_agent_reader import SceneAgentReader
from starVLA.model.modules.structured_world.losses import world_losses
from starVLA.model.modules.structured_world.geometry import geometric_fov
from starVLA.model.modules.vehicle_joint.action_head import VehicleJointActionHead, modeled_mask
from starVLA.model.modules.vehicle_joint.graphs import VehicleGraphConfig, select_vehicles
from starVLA.model.modules.vehicle_joint.initialization import (
    initialization_seed, add_random_driving_tokens, verify_generic_source, file_sha256,
    validate_pinned_sources,
)


class DDPVehicle(Qwenvl_OFT):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise ValueError("Complete driving-checkpoint initialization is forbidden; use identity-checked campaign resume")

    def __init__(self, config, accelerator=None):
        if config.get("from_scratch") is None:
            raise ValueError("This framework requires a verified generic source manifest")
        if config.from_scratch.arm not in ("A", "B", "C"):
            raise ValueError("Expected one of the preregistered A/B/C arms")
        if any(config.get(k, 0) for k in ("doing_s2", "vit_pre", "w_video_latent")):
            raise ValueError("Driving proxies, BEV and future video conditioning are forbidden")
        if config.datasets.reward_data.load_reward_data or config.framework.action_model.mlp_head:
            raise ValueError("Use the original FM action head without reward/scorer")
        sources = json.loads(Path(config.from_scratch.source_manifest).read_text())
        validate_pinned_sources(sources)
        if config.datasets.video_data.load_2d_data:
            verify_generic_source(config.framework.video_model.model_name, sources["wan"])
        if config.w_depth:
            import os
            depth_root = Path(os.environ["DEPTH_MODEL_CKPTS"])
            for name in ("ppd", "depth_v2"):
                for relative, expected in sources[name]["files"].items():
                    if file_sha256(depth_root / relative) != expected:
                        raise ValueError(f"Generic depth checkpoint mismatch: {name}/{relative}")
        # Entire original construction has the SAME RNG stream in A/B/C.
        with initialization_seed(int(config.seed)):
            super().__init__(config, accelerator=accelerator)
        if self.qwen_vl_interface.model.config.hidden_size != config.framework.qwenvl.vl_hidden_dim:
            raise ValueError("Configured action/Reader conditioning does not match the public VLM")
        if self.w_depth:
            allowed = {"dit.qwen_proj.weight", "dit.qwen_proj.bias",
                       "dit.qwen_cross_attn.in_proj_weight", "dit.qwen_cross_attn.in_proj_bias",
                       "dit.qwen_cross_attn.out_proj.weight", "dit.qwen_cross_attn.out_proj.bias"}
            report = self.ppd_generic_load_report
            unaccounted = [k for k in report["missing"] if k not in allowed and not k.startswith("sem_encoder.pretrained.")]
            if unaccounted or report["unexpected"]:
                raise ValueError(f"Unaccounted generic PPD checkpoint keys: {unaccounted}, {report['unexpected']}")
            if set(k for k in report["missing"] if not k.startswith("sem_encoder.")) != allowed:
                raise ValueError("Generic PPD must leave ALL new Qwen driving adapters randomly initialized")
            semantic = self.gs_model.semantic_load_report
            if semantic["missing"] or semantic["unexpected"]:
                raise ValueError(f"Unaccounted generic semantic encoder keys: {semantic}")
        self.arm = config.from_scratch.arm
        self.joint_enabled = self.arm != "A"
        self.sources = sources
        self.graph_config = VehicleGraphConfig(**dict(config.from_scratch.graph))
        self.vehicle_tokens = []
        if self.joint_enabled:
            hidden = self.qwen_vl_interface.model.config.hidden_size
            with initialization_seed(int(config.seed)+2000):
                self.vehicle_reader = SceneAgentReader(hidden, hidden, dim=256, scene_tokens=0,
                                                       agent_tokens=config.from_scratch.vehicle_queries)
                self.vehicle_heads = AgentHeads(hidden, classes=1, steps=8)
                # Positive canonical current yaw at initialization; future yaw
                # remains unmodeled for every vehicle regardless of this head.
                with torch.no_grad(): self.vehicle_heads.box.bias[7] = 1.
            self.action_model = VehicleJointActionHead(config, extension_seed=int(config.seed)+3000,
                                                       base_head=self.action_model)
            self.vehicle_tokens = [f"<vehicle_query_{i}>" for i in range(config.from_scratch.vehicle_queries)]
            self.vehicle_token_initialization = add_random_driving_tokens(
                self.qwen_vl_interface.model, self.qwen_vl_interface.processor.tokenizer,
                self.vehicle_tokens, int(config.seed)+4000)
            self._special_token_ids["vehicle"] = tuple(self.vehicle_token_initialization["tokens"].values())
        # Original recipe freezes generic visual extraction/encoders, not Qwen
        # language layers nor a tied lm_head that would freeze token embeddings.
        self.qwen_vl_interface.model.model.visual.requires_grad_(False)
        if hasattr(self, "rgb_model"):
            for name in ("vae", "clip_image_encoder", "text_encoder"):
                if hasattr(self.rgb_model, name): getattr(self.rgb_model, name).requires_grad_(False)
        if config.from_scratch.gradient_checkpointing:
            self.qwen_vl_interface.model.model.language_model.gradient_checkpointing_enable()
            self.action_model.model.gradient_checkpointing = True
            if hasattr(self, "rgb_model"):
                self.rgb_model.transformer3d.enable_gradient_checkpointing()

    def train(self, mode=True):
        super().train(mode)
        self.qwen_vl_interface.model.model.visual.eval()
        if hasattr(self, "rgb_model"):
            for name in ("vae", "clip_image_encoder", "text_encoder"):
                if hasattr(self.rgb_model, name): getattr(self.rgb_model, name).eval()
        if hasattr(self, "gs_model") and hasattr(self.gs_model, "sem_encoder"):
            self.gs_model.sem_encoder.eval()
        return self

    def amp(self):
        return torch.autocast("cuda" if next(self.parameters()).is_cuda else "cpu", dtype=torch.bfloat16)

    def encode_current(self, examples):
        # Explicit input whitelist prevents every label/cache field from being
        # forwarded even when the training batch also carries supervision.
        current = [{k: e[k] for k in ("image", "lang", "state")} for e in examples]
        instructions = [append_world_action_tokens(e["lang"], self.act_tok, bool(self.w_depth)) for e in current]
        if self.joint_enabled:
            instructions = [s.replace(self.act_query_tokens[0], "".join(self.vehicle_tokens)+self.act_query_tokens[0]) for s in instructions]
        ids, attention, positions, slots, visual, deepstack = self._build_qwen_batch(current, instructions)
        with self.amp():
            embeddings = self.qwen_vl_interface.model.get_input_embeddings()(ids)
            states = torch.as_tensor(np.asarray([e["state"] for e in current]), device=ids.device, dtype=torch.float32)[:, 0]
            state_embeds = self.action_input_model(states)
            batch = torch.arange(len(examples), device=ids.device)
            embeddings[batch, slots["history"][:, 0]] = state_embeds.to(embeddings.dtype)
            if self.config.datasets.video_data.load_2d_data:
                embeddings[batch[:, None], slots["rgb"]] = self.rgb_query.to(embeddings.dtype)[None]
            if self.w_depth:
                embeddings[batch[:, None], slots["gs"]] = self.gs_query.to(embeddings.dtype)[None]
            if self.joint_enabled:
                lengths = ids.eq(self.qwen_vl_interface.model.config.image_token_id).sum(-1).tolist()
                parts = visual.split(lengths)
                padded = nn.utils.rnn.pad_sequence(parts, batch_first=True)
                support = torch.arange(padded.shape[1], device=ids.device)[None] < torch.tensor(lengths, device=ids.device)[:, None]
                read = self.vehicle_reader(padded, support=support).agent_memory
                embeddings[batch[:, None], slots["vehicle"]] = read.to(embeddings.dtype)
            hidden = self._qwen_language_forward(ids, embeddings, attention, positions, visual, deepstack)
            gathered = {key: hidden[batch[:, None], idx] for key, idx in slots.items()}
            if self.joint_enabled:
                gathered["vehicle_prediction"] = self.vehicle_heads(gathered["vehicle"])
        return gathered

    def current_graph(self, encoded, examples):
        prediction = encoded["vehicle_prediction"]
        device = prediction["boxes"].device
        b, k = len(examples), self.graph_config.max_vehicles+1
        boxes = prediction["boxes"].new_zeros(b, k, 8)
        boxes[:, 0] = boxes.new_tensor([0., 0., 0., 4.9, 2., 1.6, 0., 1.])
        queries = encoded["vehicle"].new_zeros(b, k, encoded["vehicle"].shape[-1])
        queries[:, 0] = encoded["action"].mean(1)
        active = torch.zeros(b, k, dtype=torch.bool, device=device); active[:, 0] = True
        source = torch.full((b, k), -1, device=device, dtype=torch.long)
        audits = []
        for i, example in enumerate(examples):
            calibration = example["current_calibration"]
            support = geometric_fov(prediction["boxes"][i, :, :3].detach().float().cpu().numpy(),
                                    calibration["intrinsics"], calibration["extrinsics"], calibration["distortion"])
            selected, context, audit = select_vehicles(prediction["boxes"][i].float(),
                prediction["logits"][i].float().softmax(-1)[:, 0], torch.from_numpy(support).to(device),
                example["ego_speed"], example["navigation"], self.graph_config)
            n = len(selected)
            if n:
                boxes[i, 1:n+1] = prediction["boxes"][i, selected]
                queries[i, 1:n+1] = encoded["vehicle"][i, selected]
                active[i, 1:n+1] = True
                source[i, 1:n+1] = torch.tensor(selected, device=device)
            audit["graph_source"] = "predicted_current_camera"
            # All fixed visual vehicle queries already participate in Qwen's
            # causal world->action path. Context indices are an audit, not GT.
            audits.append(audit)
        return boxes, queries, active, source, audits

    @staticmethod
    def move_targets(examples, device):
        targets = []
        for e in examples:
            t = e["vehicle_targets"]
            if isinstance(t, dict): t = WorldTargets(**t)
            targets.append(WorldTargets(**{f.name: getattr(t, f.name).to(device) if isinstance(getattr(t, f.name), torch.Tensor)
                                          else getattr(t, f.name) for f in fields(t)}))
        return targets

    def auxiliary_world_losses(self, encoded, examples, ego):
        zero = ego.sum()*0.
        losses = {"video": zero, "depth": zero}
        with self.amp():
            if self.config.datasets.video_data.load_2d_data:
                rgb = encoded["rgb"]
                video = [{k: v.to(ego.device) if isinstance(v, torch.Tensor) else v
                          for k,v in e["2d_gen_data"].items()} for e in examples]
                losses["video"], _ = self.rgb_model(video, rgb)
                if self.rgb_query_loss:
                    summary = self.traj_emb(rgb, self.traj_emb_h0[None].expand(1, len(examples), -1).contiguous())[1].squeeze(0)
                    losses["video"] = losses["video"] + F.l1_loss(self.rgb_act_pre(summary).reshape_as(ego).float(), ego)
            if self.w_depth:
                depth = [e["depth_data"] for e in examples]
                values = {key: torch.stack([d[key] for d in depth]).flatten(0, 1).to(ego.device) for key in ("image", "depth", "mask")}
                values["qwen_token"] = encoded["gs"].repeat_interleave(3, 0)
                loss = self.gs_model.forward_train(values)["loss"]
                if self.gs_query_loss:
                    summary = self.gs_traj_emb(encoded["gs"], self.gs_traj_emb_h0[None].expand(1, len(examples), -1).contiguous())[1].squeeze(0)
                    loss = loss + F.l1_loss(self.gs_act_pre(summary).reshape_as(ego).float(), ego)
                losses["depth"] = loss*.1
        return losses

    def forward(self, examples, *, completed_updates=0, role_scheduler=None, noise_generator=None):
        encoded = self.encode_current(examples)
        ego = torch.as_tensor(np.asarray([e["action"] for e in examples]), device=encoded["action"].device, dtype=torch.float32)
        if not torch.isfinite(ego).all(): raise ValueError("Invalid ego labels")
        losses = self.auxiliary_world_losses(encoded, examples, ego)
        repeat = self.config.framework.action_model.repeated_diffusion_steps
        metrics = {}
        if not self.joint_enabled:
            with self.amp():
                losses["main_fm"] = self.action_model(encoded["action"].repeat(repeat, 1, 1), ego.repeat(repeat, 1, 1))
        else:
            targets = self.move_targets(examples, ego.device)
            # Per-scene normalization followed by scene mean makes microbatch,
            # tail and DDP accumulation weighting explicit (no variable-label
            # denominator silently changes when a global batch is partitioned).
            per_scene, matches = [], []
            for i, target in enumerate(targets):
                (sums, counts), match = world_losses(
                    {k:v[i:i+1].float() for k,v in encoded["vehicle_prediction"].items()}, [target], return_sums=True)
                per_scene.append({k:sums[k]/max(1, counts[k]) for k in sums})
                matches.extend(match)
            world = {k:torch.stack([d[k] for d in per_scene]).mean() for k in per_scene[0]}
            losses.update({"vehicle_"+k: v for k,v in world.items()})
            boxes, queries, active, source, audits = self.current_graph(encoded, examples)
            labels = ego.new_zeros(len(ego), active.shape[1], 8, 4)
            valid = torch.zeros_like(labels, dtype=torch.bool)
            labels[:, 0] = ego; valid[:, 0] = True
            for b, (rows, cols) in enumerate(matches):
                assignment = dict(zip(rows.tolist(), cols.tolist()))
                for actor in torch.where(active[b, 1:])[0].tolist():
                    actor += 1; query = int(source[b, actor])
                    if query not in assignment: continue
                    gt = assignment[query]
                    # Matching is never gated by class correctness or 2m error.
                    labels[b, actor, :, :2] = (targets[b].future_xy_in_ego_t0[gt]-boxes[b, actor, :2].detach())/20.
                    valid[b, actor, :, :2] = targets[b].future_valid_mask[gt, :, None]
            rep = lambda t: t.repeat((repeat,)+(1,)*(t.ndim-1))
            conditions = [rep(encoded["action"]), rep(queries), rep(boxes), rep(active)]
            y, v = rep(labels), rep(valid)
            def fm(known=None):
                noise = torch.randn(y.shape, device=y.device, dtype=y.dtype, generator=noise_generator)
                times = self.action_model.sample_time(len(y), y.device, y.dtype)
                with self.amp():
                    return self.action_model.loss(*conditions, y, v, noise, times, known)
            losses["main_fm"], metrics = fm()
            from starVLA.model.modules.vehicle_joint.masks import auxiliary_due
            if auxiliary_due(completed_updates, self.config.from_scratch.all_hidden_start):
                if role_scheduler is None: raise ValueError("Auxiliary updates require a resumable independent role scheduler")
                known, tasks = role_scheduler.known_mask(active, valid)
                if self.arm == "B": known = torch.zeros_like(known)
                extra, aux_metrics = fm(rep(known))
                losses["role_auxiliary"] = .1*extra
                metrics["role_tasks"] = tasks
                metrics["auxiliary_coordinates"] = aux_metrics
            metrics["graphs"] = audits
        return {"loss": sum(losses.values()), "losses": losses, "metrics": metrics}

    @torch.no_grad()
    def predict_action(self, examples, *, initial_noise):
        encoded = self.encode_current(examples)
        with self.amp():
            if self.joint_enabled:
                boxes, queries, active, source, audit = self.current_graph(encoded, examples)
                joint = self.action_model.sample(encoded["action"], queries, boxes, active, initial_noise)
                ego_encoded = self.action_model.executed_ego(joint)
                vehicles_xy = joint[:, 1:, :, :2]*20.+boxes[:, 1:, None, :2]
                vehicles_xy = torch.where(active[:, 1:, None, None], vehicles_xy, 0.)
            else:
                ego_encoded = self.action_model.predict_action(encoded["action"], initial_noise=initial_noise)
                joint = ego_encoded[:, None]; vehicles_xy = None; active = None; audit = []
        # Unique standard DDP ego decoder, applied once to slot0 only.
        from starVLA.dataloader.navsim_dataset import x_mean, x_std, y_mean, y_std
        ego_xy = ego_encoded[..., :2].float()*ego_encoded.new_tensor([x_std, y_std])+ego_encoded.new_tensor([x_mean, y_mean])
        ego_yaw = torch.atan2(ego_encoded[..., 2].float(), ego_encoded[..., 3].float())
        result = torch.cat((ego_xy, ego_yaw[..., None]), -1)
        if not torch.isfinite(result).all(): raise FloatingPointError("Nonfinite executed ego")
        return {"ego": result, "joint_encoded": joint, "vehicle_xy": vehicles_xy,
                "active_actor_mask": active, "graph_audit": audit}
