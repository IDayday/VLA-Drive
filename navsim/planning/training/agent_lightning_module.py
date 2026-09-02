from time import sleep
import time
import os
import json
import logging
import inspect
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch import Tensor
from typing import Dict, Tuple, Any, List
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import Trajectory
from navsim.planning.training.formal_timing import PhaseTimer

logger = logging.getLogger(__name__)


def _timed_allreduce_hook(
    state: Dict[str, Any], bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """DDP-equivalent averaged all-reduce with benchmark-only timing."""
    start = time.perf_counter()
    tensor = bucket.buffer()
    future = dist.all_reduce(tensor, async_op=True).get_future()

    def finish(result):
        reduced = result.value()[0]
        reduced.div_(dist.get_world_size())
        state["seconds"] += time.perf_counter() - start
        state["bucket_count"] += 1
        return reduced

    return future.then(finish)

def _rowwise_isin(tensor_1: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    matches = (tensor_1[:, None] == target_tensor)
    
    return torch.sum(matches, dim=1, dtype=torch.bool)


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(
        self,
        agent: AbstractAgent,
        for_viz=False,
        for_analysis=False,
        diagnostics=None,
    ):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent
        self._formal_timing_enabled = (
            os.getenv("PLANREG_FORMAL_TIMING", "0").lower()
            in {"1", "true", "yes", "on"}
        )
        self._formal_phase_timer = PhaseTimer(self._formal_timing_enabled)
        self._formal_backward_token = None
        self._formal_latest_loss_tensors = {}
        self._formal_latest_gradient_norms = {}
        self._formal_allreduce_state = {"seconds": 0.0, "bucket_count": 0}
        self._formal_comm_hook_registered = False
        self.checkpoint_file=None
        self.for_viz = for_viz
        self.for_analysis=for_analysis
        diagnostics = diagnostics or {}
        self.grad_log_interval = int(
            getattr(diagnostics, "grad_log_interval", diagnostics.get("grad_log_interval", 100))
        )
        self.register_log_interval = int(
            getattr(
                diagnostics,
                "register_log_interval",
                diagnostics.get("register_log_interval", 100),
            )
        )
        self.debug_unused_parameters = bool(
            getattr(
                diagnostics,
                "debug_unused_parameters",
                diagnostics.get("debug_unused_parameters", False),
            )
        )
        self.require_finite_loss_and_gradients = bool(
            getattr(
                diagnostics,
                "require_finite_loss_and_gradients",
                diagnostics.get("require_finite_loss_and_gradients", False),
            )
        )
        if self.grad_log_interval <= 0 or self.register_log_interval <= 0:
            raise ValueError("Diagnostic log intervals must be positive")
        self._gradient_finiteness_checks = []
        self._finite_gradient_hook_handles = []
        if self.require_finite_loss_and_gradients:
            # Register once during construction rather than traversing every
            # parameter after every backward pass.  The hooks only retain the
            # names of offending gradients, not the gradient tensors.
            for name, parameter in self.agent.named_parameters():
                if not parameter.requires_grad:
                    continue

                def _check_finite_gradient(gradient, *, parameter_name=name):
                    if gradient is not None:
                        gradient_values = (
                            gradient.coalesce().values()
                            if gradient.is_sparse
                            else gradient
                        )
                        # Keep the scalar on-device here. on_after_backward
                        # batches the device-to-host check so this does not
                        # synchronize once per parameter.
                        self._gradient_finiteness_checks.append(
                            (
                                f"agent.{parameter_name}",
                                torch.isfinite(gradient_values).all(),
                            )
                        )
                    return gradient

                self._finite_gradient_hook_handles.append(
                    parameter.register_hook(_check_finite_gradient)
                )

    def on_fit_start(self) -> None:
        if hasattr(self.agent, "configure_total_optimizer_steps"):
            self.agent.configure_total_optimizer_steps(
                int(self.trainer.estimated_stepping_batches)
            )
        if self._formal_timing_enabled and dist.is_available() and dist.is_initialized():
            wrapped_model = self.trainer.strategy.model
            if isinstance(wrapped_model, DistributedDataParallel):
                wrapped_model.register_comm_hook(
                    self._formal_allreduce_state, _timed_allreduce_hook
                )
                self._formal_comm_hook_registered = True
        if getattr(self.agent, "_formal_initialization", False):
            self._write_formal_parameter_audit()

    def _write_formal_parameter_audit(self) -> None:
        if not self.trainer.is_global_zero:
            return
        summaries = getattr(self.agent, "_planreg_optimizer_group_summary", None)
        runtime = getattr(self.agent, "_planreg_optimizer_runtime", None)
        if not summaries or not runtime:
            raise RuntimeError(
                "Formal optimizer audit is unavailable after optimizer construction"
            )
        trainable = sum(
            parameter.numel()
            for parameter in self.agent.parameters()
            if parameter.requires_grad
        )
        frozen = sum(
            parameter.numel()
            for parameter in self.agent.parameters()
            if not parameter.requires_grad
        )
        llm_trainable = sum(
            parameter.numel()
            for parameter in self.agent.backbone.model.language_model.parameters()
            if parameter.requires_grad
        )
        if llm_trainable:
            raise RuntimeError("Formal parameter audit found trainable LLM parameters")
        scheduler = self.agent.scheduler_args
        start_ratio = float(scheduler.start_lr_ratio)
        minimum_ratio = float(scheduler.min_lr_ratio)
        logical_peak_lrs = dict(runtime["logical_learning_rates"])
        payload = {
            "schema_version": 1,
            "agent_checkpoint_loaded": bool(
                getattr(self.agent, "_agent_checkpoint_loaded", True)
            ),
            "actual_global_batch": int(runtime["actual_global_batch"]),
            "lr_scale_multiplier": float(runtime["lr_scale"]),
            "optimizer_groups": summaries,
            "logical_peak_learning_rates": logical_peak_lrs,
            "logical_start_learning_rates": {
                name: value * start_ratio
                for name, value in logical_peak_lrs.items()
            },
            "logical_final_learning_rates": {
                name: value * minimum_ratio
                for name, value in logical_peak_lrs.items()
            },
            "scheduler": {
                "type": str(scheduler.type),
                "interval": str(scheduler.interval),
                "warmup_ratio": float(scheduler.warmup_ratio),
                "start_lr_ratio": start_ratio,
                "min_lr_ratio": minimum_ratio,
                "total_optimizer_steps": int(
                    self.trainer.estimated_stepping_batches
                ),
            },
            "ema": {
                "actual_start_momentum": float(
                    self.agent._ema_actual_start_momentum.item()
                ),
                "actual_end_momentum": float(
                    self.agent._ema_actual_end_momentum.item()
                ),
            },
            "trainable_parameter_count": trainable,
            "frozen_parameter_count": frozen,
            "language_model_trainable_parameter_count": llm_trainable,
            "world_model_enabled": bool(self.agent.world_model_enabled),
        }
        if payload["agent_checkpoint_loaded"]:
            raise RuntimeError("Formal run unexpectedly loaded an agent checkpoint")
        metadata_dir = Path(self.trainer.default_root_dir) / "run_metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        output = metadata_dir / "training_parameter_audit.json"
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)

    def on_train_batch_start(self, batch, batch_idx) -> None:
        del batch, batch_idx
        if self._formal_timing_enabled:
            self._formal_allreduce_state["seconds"] = 0.0
            self._formal_allreduce_state["bucket_count"] = 0

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        token = self._formal_phase_timer.start("host_to_device_time")
        transferred = super().transfer_batch_to_device(
            batch, device, dataloader_idx
        )
        self._formal_phase_timer.stop(token)
        return transferred

    def on_predict_start(self) -> None:
        if hasattr(self.agent, "remove_training_only_world_model"):
            self.agent.remove_training_only_world_model()

    def on_before_backward(self, loss: Tensor) -> None:
        del loss
        self._formal_backward_token = self._formal_phase_timer.start(
            "backward_time"
        )
        if self.require_finite_loss_and_gradients:
            self._gradient_finiteness_checks.clear()

    def on_after_backward(self) -> None:
        self._formal_phase_timer.stop(self._formal_backward_token)
        self._formal_backward_token = None
        if self._formal_timing_enabled:
            self._formal_phase_timer.add_seconds(
                "all_reduce_time", self._formal_allreduce_state["seconds"]
            )
            if hasattr(self.agent, "get_planreg_gradient_norms"):
                self._formal_latest_gradient_norms = {
                    name: value.detach()
                    for name, value in self.agent.get_planreg_gradient_norms().items()
                }
        optimizer_step = int(self.global_step)
        log_gradients = optimizer_step % self.grad_log_interval == 0
        log_registers = optimizer_step % self.register_log_interval == 0

        if log_gradients and hasattr(self.agent, "get_planreg_gradient_norms"):
            for name, value in self.agent.get_planreg_gradient_norms().items():
                self.log(
                    f"train/{name}",
                    value,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    sync_dist=True,
                )

        if log_registers and hasattr(
            self.agent, "get_planreg_register_diagnostics"
        ):
            for name, value in self.agent.get_planreg_register_diagnostics().items():
                self.log(
                    f"train/{name}",
                    value,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    sync_dist=True,
                )

        # Full parameter traversal is deliberately opt-in and interval-gated.
        if self.debug_unused_parameters and log_gradients:
            unused = [
                name
                for name, parameter in self.named_parameters()
                if parameter.requires_grad and parameter.grad is None
            ]
            if unused and self.trainer.is_global_zero:
                logger.warning(
                    "Unused trainable parameters at optimizer step %d (%d total):\n%s",
                    optimizer_step,
                    len(unused),
                    "\n".join(f"  {name}" for name in unused),
                )
        if self.require_finite_loss_and_gradients:
            checks_by_device = {}
            for name, finite in self._gradient_finiteness_checks:
                checks_by_device.setdefault(finite.device, []).append(
                    (name, finite)
                )
            nonfinite = []
            for checks in checks_by_device.values():
                finite_flags = torch.stack(
                    [finite for _, finite in checks]
                ).cpu().tolist()
                nonfinite.extend(
                    name
                    for (name, _), is_finite in zip(checks, finite_flags)
                    if not is_finite
                )
            nonfinite.sort()
            if nonfinite:
                raise FloatingPointError(
                    f"Non-finite gradients detected: {nonfinite[:16]}"
                )

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch

        if self._formal_timing_enabled:
            self._formal_latest_input_timing = {
                name: features[name].detach()
                for name in ("input_decode_time", "input_transform_time")
                if name in features and isinstance(features[name], torch.Tensor)
            }

        if hasattr(self.agent, "set_optimizer_step"):
            # Lightning restores global_step on resume, so rho/EMA schedules
            # continue from the exact completed optimizer step.
            self.agent.set_optimizer_step(self.global_step)

        if not 'mem' in self.agent.name().lower() or self.agent._config.memory_mode=='base':
            prediction = self.agent.forward(features)
            loss_dict = self.agent.compute_loss(features, targets, prediction)
        elif self.agent._config.memory_mode=='waver':
            prediction = self.agent.forward(features)
            trajectory,gate=self.agent.waver_forward(prediction)
            prediction['proposal_list']=[trajectory.unsqueeze(1)]
            loss_dict=self.agent.compute_waver_loss(features,targets,prediction)

        elif self.agent._config.memory_mode=='adapter':
            prediction=self.agent.adapter_forward(features)
            loss_dict=self.agent.compute_adapter_loss(targets,prediction)

        if type(loss_dict) is dict:
            if self._formal_timing_enabled:
                self._formal_latest_loss_tensors = {
                    name: value.detach()
                    for name, value in loss_dict.items()
                    if isinstance(value, torch.Tensor) and value.numel() == 1
                }
            if self.require_finite_loss_and_gradients:
                nonfinite_losses = [
                    key
                    for key, value in loss_dict.items()
                    if isinstance(value, torch.Tensor)
                    and not bool(torch.isfinite(value).all())
                ]
                if nonfinite_losses:
                    raise FloatingPointError(
                        f"Non-finite losses detected: {nonfinite_losses}"
                    )
            sync_metrics = logging_prefix != "train" or os.getenv(
                "DRIVEVLA_SYNC_TRAIN_METRICS", "0"
            ).lower() in {"1", "true", "yes", "on"}
            train_log_interval = int(
                os.getenv("DRIVEVLA_TRAIN_LOG_INTERVAL", "1")
            )
            if train_log_interval <= 0:
                raise ValueError("DRIVEVLA_TRAIN_LOG_INTERVAL must be positive")
            should_log = (
                logging_prefix != "train"
                or self.global_step % train_log_interval == 0
            )
            if should_log:
                for key,value in loss_dict.items():
                    self.log(
                        f"{logging_prefix}/" + key,
                        value,
                        on_step=True,
                        on_epoch=False,
                        prog_bar=True,
                        sync_dist=sync_metrics,
                    )
            return loss_dict["loss"]
        else:
            return loss_dict

    def consume_formal_step_timings(self) -> Dict[str, float]:
        if not self._formal_timing_enabled:
            return {}
        result = self._formal_phase_timer.consume()
        if hasattr(self.agent, "consume_formal_timings"):
            for name, seconds in self.agent.consume_formal_timings().items():
                result[name] = result.get(name, 0.0) + float(seconds)
        for name, value in getattr(self, "_formal_latest_input_timing", {}).items():
            result[name] = float(value.detach().float().mean().cpu().item())
        for name, value in self._formal_latest_gradient_norms.items():
            result[f"gradient/{name}"] = float(value.float().cpu().item())
        for name, value in self._formal_latest_loss_tensors.items():
            result[f"loss/{name}"] = float(value.float().cpu().item())
        result["all_reduce_bucket_count"] = float(
            self._formal_allreduce_state["bucket_count"]
        )
        self._formal_latest_input_timing = {}
        self._formal_latest_gradient_norms = {}
        self._formal_latest_loss_tensors = {}
        return result
            

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        if ('drivor' in self.agent.name().lower() or 'episode' in self.agent.name().lower()) and 'mem' not in self.agent.name().lower():
            features, targets = batch
            # score,best_score=self.agent.inference(features, targets)
            predictions = self.agent.forward(features)
            all_chosen_trajectories = predictions["trajectory"][:,None]
            all_proposed_trajectories = predictions["proposals"]
            if os.getenv("DRIVEVLA_FUSE_VALIDATION_SCORING", "0").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                combined_trajectories = torch.cat(
                    [all_chosen_trajectories, all_proposed_trajectories], dim=1
                )
                _, _, combined_scores, l2, trajectoy_scores = self.agent.compute_score(
                    targets, combined_trajectories
                )
                proposal_scores = combined_scores[:, :1]
                all_proposal_scores = combined_scores[:, 1:]
                final_score = proposal_scores.mean()
                best_score = all_proposal_scores.amax(dim=-1).mean()
            else:
                final_score, _, proposal_scores, l2, trajectoy_scores = self.agent.compute_score(
                    targets, all_chosen_trajectories
                )
                _, best_score, all_proposal_scores, _, _ = self.agent.compute_score(
                    targets, all_proposed_trajectories
                )
            mean_score=proposal_scores.mean()

            logging_prefix="val"
            if "pdm_score" in predictions:
                pdm_score = predictions["pdm_score"]
                pred_pdms = predictions.get("pred_pdms")
                if pred_pdms is None:
                    pred_pdms = torch.exp(pdm_score) / float(
                        self.agent.action_head_config.ttc
                        + self.agent.action_head_config.ep
                        + self.agent.action_head_config.comfort
                    )
                best_pred_indices = torch.argmax(pdm_score, dim=1)
                batch_indices = torch.arange(
                    len(pdm_score), device=pdm_score.device
                )
                best_pred_score_values = pred_pdms[
                    batch_indices, best_pred_indices
                ]
                selected_target_pdms = proposal_scores.reshape(
                    len(proposal_scores), -1
                )[:, 0]
                score_error = torch.abs(
                    best_pred_score_values - selected_target_pdms
                ).mean()
                self.log(f"{logging_prefix}/score_error", score_error, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                
                best_pred_score_index = best_pred_indices
                best_real_score_index = torch.argmax(all_proposal_scores, dim=1)
                score_hit_rate = torch.mean(best_pred_score_index == best_real_score_index, dtype=torch.float32)

                best_possible_scores = all_proposal_scores[torch.arange(len(all_proposal_scores)), best_real_score_index]
                best_actual_scores = all_proposal_scores[torch.arange(len(all_proposal_scores)), best_pred_score_index]
                lost_score = torch.mean(best_possible_scores - best_actual_scores)
                self.log(f"{logging_prefix}/score_hit_rate", score_hit_rate, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log(f"{logging_prefix}/lost_score", lost_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

                top_5_indices_real = torch.topk(all_proposal_scores, k=5, dim=1).indices
                top_5_score_hit_rate = _rowwise_isin(best_pred_score_index, top_5_indices_real).mean(dtype=torch.float32)
                self.log(f"{logging_prefix}/top_5_score_hit_rate", top_5_score_hit_rate, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            
            # Checkpoint selection only consumes the global epoch score.  Give
            # it the same key directly instead of also reducing and rendering
            # an unused per-batch ``score_step`` value.
            self.log(f"{logging_prefix}/score_epoch", final_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/best_score", best_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/mean_score", mean_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/l2", l2, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            collision=trajectoy_scores[:,0].mean()
            self.log(f"{logging_prefix}/collision", collision, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            drivable_area_compliance=trajectoy_scores[:,1].mean()
            self.log(f"{logging_prefix}/dac", drivable_area_compliance, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            ego_progress=trajectoy_scores[:,2].mean()
            self.log(f"{logging_prefix}/progress", ego_progress, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            time_to_collision_within_bound=trajectoy_scores[:,3].mean()
            self.log(f"{logging_prefix}/ttc", time_to_collision_within_bound, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            comfort=trajectoy_scores[:,4].mean()
            self.log(f"{logging_prefix}/comfort", comfort, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            return final_score
        elif 'mem' in self.agent.name().lower() and self.agent._config.memory_mode=='adapter':
            logging_prefix='val'
            features, targets = batch
            prediction=self.agent.adapter_forward(features)
            loss_dict=self.agent.compute_adapter_loss(targets,prediction)
            for key,value in loss_dict.items():
                self.log(f"{logging_prefix}/"+key, value, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            return loss_dict["loss"]
        else:
            return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        optimizer_factory = self.agent.get_optimizers
        if "total_optimizer_steps" in inspect.signature(
            optimizer_factory
        ).parameters:
            total_steps = int(self.trainer.estimated_stepping_batches)
            if total_steps <= 0:
                raise RuntimeError(
                    "trainer.estimated_stepping_batches must be positive"
                )
            return optimizer_factory(total_optimizer_steps=total_steps)
        return optimizer_factory()
    
    def predict_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Used during the multi-gpu proccessing to parallelize the prediction of trajectories.
        NOTE: requires append_token_to_batch=True in the dataset used to instantiate the trainer.
        """
        return self.predict_step_drivor(batch, batch_idx)

    def predict_step_drivor(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor], List[str]], batch_idx: int):
        features, targets, tokens = batch
        self.agent.eval()
        with torch.no_grad():
            if 'mem' in self.agent.name().lower() and self.agent._config.memory_type=="error" and self.agent._config.memory_mode=="infer":
                predictions=self.agent.traj_forward(features)
                predictions,retrieved_trajs=self.agent.test_forward(predictions)
                predictions=self.agent.score_forward(features,predictions,retrieved_trajs)
                final_trajectories=predictions['proposals']
                _, _, final_scores, _, _ = self.agent.compute_score(targets, final_trajectories)
            elif 'mem' in self.agent.name().lower() and self.agent._config.memory_type=='error' and self.agent._config.memory_mode=='adapter_infer':
                predictions=self.agent.adapter_forward(features)
            else:
                predictions = self.agent.forward(features)
                if 'mem' in self.agent.name().lower() and self.agent._config.memory_type=="error" and self.agent._config.memory_mode=="pre":
                    pdm_scores,_,target_scores,_,_,_=self.agent.compute_score(targets,predictions['trajectory'].unsqueeze(1),test=False)
                    rewards={'pdm_scores':pdm_scores,'target_scores':target_scores}
                    self.agent.add_error_episode(predictions,targets,rewards)
                # if 'mem' in self.agent.name().lower() and self.agent._config.memory_type=="error" and self.agent._config.memory_mode=="infer":
                #     predictions=self.agent.test_forward(predictions)
            poses = predictions["trajectory"]
            if self.for_viz:
                all_proposed_trajectories = predictions["proposal_list"]
                final_trajectories = predictions["proposals"]
                _, _, final_scores, _, _ = self.agent.compute_score(targets, final_trajectories)
                ego_status = features["ego_status"]
        result = {}
        if 'mem' in self.agent.name().lower() and self.agent._config.memory_mode=='infer' and self.for_analysis:
            targets_poses=targets['trajectory'].cpu().numpy()
            accept=predictions['accept'].cpu().numpy()
            retrieved=predictions['retrieved'].cpu().numpy()
            pdm_scores=predictions['pdm_scores'].cpu().numpy()
            hit=predictions['hit'].cpu().numpy()

            for index, (pose,t_pose, a,r,h,pdm_score,token) in enumerate(zip(poses.cpu().numpy(),targets_poses,accept,retrieved, hit,pdm_scores, tokens)):
                proposal = Trajectory(pose)
                proposals = predictions['proposals'][index]
                result[token] = {
                    'trajectory': proposal, 
                    'all_proposals': proposals, 
                    'pred_pdm': pdm_score,
                    'target_pdm': final_scores[index],
                    'pred_nc': predictions['pred_logit']['no_at_fault_collisions'].sigmoid()[index],
                    'pred_dac': predictions['pred_logit']['drivable_area_compliance'].sigmoid()[index],
                    'high_level_command': features['status_feature'][index],
                    'target_pose': t_pose,
                    'current_pose': pose,
                    'accept': a,
                    'retrieved': r,
                    'hit': h,
                }
        elif 'mem' in self.agent.name().lower() and self.agent._config.memory_mode=='adapter_infer' and self.for_analysis:
            targets_poses=targets['trajectory'].cpu().numpy()
            # mem_attn=predictions['mem_attn'].cpu().numpy()
            # for index, (pose,t_pose, attn,token) in enumerate(zip(poses.cpu().numpy(),targets_poses,mem_attn, tokens)):
            #     proposal = Trajectory(pose)
            #     result[token] = {
            #         'trajectory': proposal, 
            #         'target_pose': t_pose,
            #         'mem_attn': attn,
            #     }
            for index, (pose,t_pose,token) in enumerate(zip(poses.cpu().numpy(),targets_poses, tokens)):
                proposal = Trajectory(pose)
                result[token] = {
                    'trajectory': proposal, 
                    'target_pose': t_pose,
                }

        else:
            for index, (pose, token) in enumerate(zip(poses.cpu().numpy(), tokens)):
                proposal = Trajectory(pose)
                if self.for_viz:
                    proposal_list = [proposal_list[index].cpu().numpy() for proposal_list in all_proposed_trajectories]
                    result[token] = {
                        'trajectory': proposal, 
                        'all_proposals': proposal_list, 
                        'all_proposal_scores': final_scores[index],
                        'high_level_command': ego_status[index]
                    }
                else:
                    result[token] = {'trajectory': proposal}
        return result

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if 'mem' in self.agent.name().lower():
            checkpoint['memory'] = self.agent.memory.banks
