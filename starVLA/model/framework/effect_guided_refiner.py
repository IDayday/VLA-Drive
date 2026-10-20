"""Thin EGR deployment/training boundary around one shared frozen Base call."""
from __future__ import annotations

import torch
from torch import nn

from starVLA.model.modules.action_effect.effect_losses import refinement_loss
from starVLA.model.modules.action_effect.refinement_policy import CoefficientRefiner
from starVLA.model.modules.action_effect.trajectory_constraints import ConstraintSpec, geometric_screen


class EffectGuidedRefiner(nn.Module):
    """A0/A1 differ only in training guidance, never in deploy inputs.

    Cached head training calls forward(context, base). Online integration binds
    a Base adapter with encode_and_plan(observations, initial_noise=...) returning
    (the same-call legal context, physical Base trajectory). The adapter is the
    only VLM owner. No predicted or real feedback is required at deployment.
    The optional teacher is part of training state and excluded by deploy_state.
    """
    def __init__(self, refiner: CoefficientRefiner, *, effect_teacher=None,
                 base_policy=None, constraint_spec: ConstraintSpec = ConstraintSpec(),
                 teacher_event_heads_trained=False):
        super().__init__()
        self.refiner = refiner
        self.effect_teacher = effect_teacher
        self.base_policy = base_policy
        self.constraint_spec = constraint_spec.validate()
        if type(teacher_event_heads_trained) is not bool:
            raise ValueError("Explicit trained-event-head identity must be boolean")
        self.teacher_event_heads_trained = teacher_event_heads_trained
        for frozen in (effect_teacher, base_policy):
            if frozen is not None:
                frozen.requires_grad_(False); frozen.eval()

    def train(self, mode=True):
        super().train(mode)
        for frozen in (self.effect_teacher, self.base_policy):
            if frozen is not None:
                frozen.eval()
        return self

    def forward(self, context, base_physical):
        return self.refiner(context, base_physical)

    def training_loss(self, context, base_physical, *, use_effect_guidance=False, **supervision):
        prediction = self(context, base_physical)
        teacher_prediction = None
        if use_effect_guidance:
            if self.effect_teacher is None:
                raise ValueError("A1 requires a verified frozen effect-teacher checkpoint")
            if self.effect_teacher.training or any(p.requires_grad for p in self.effect_teacher.parameters()):
                raise ValueError("Effect teacher must remain frozen and eval")
            teacher_mode = getattr(getattr(self.effect_teacher, "config", None), "mode", None)
            if teacher_mode == "terminal_pdms":
                raise ValueError("A1 safety-effect guidance needs trained component heads; PDMS-only is a separate control")
            if teacher_mode == "absolute_score" and supervision.get("effect_valid") is not None:
                trained = torch.tensor([u == "0..1" for u in self.effect_teacher.label_spec.units],
                                       device=supervision["effect_valid"].device)
                supervision["effect_valid"] = supervision["effect_valid"] & trained
            # Deliberately retain gradient w.r.t. refiner trajectory, not teacher params.
            teacher_prediction = self.effect_teacher(context, base_physical.detach(), prediction["trajectory"])
        result = refinement_loss(prediction, base=base_physical.detach(),
                                 coefficient_scales=self.refiner.config.coefficient_scales,
                                 teacher_prediction=teacher_prediction,
                                 use_event_guidance=self.teacher_event_heads_trained, **supervision)
        return result

    @torch.no_grad()
    def deploy(self, context, base_physical, *, predict_effect=False):
        """No real feedback/simulator access; optional learned prediction is explicit."""
        if self.training:
            raise ValueError("Deployment requires eval mode")
        prediction = self(context, base_physical)
        # ego4 is normalized HISTORY delta-pose (dx,dy,sin dyaw,cos dyaw),
        # not vx/vy/ax/ay. Never reinterpret it as an observed initial velocity.
        # This preliminary screen omits that boundary; official offline replay
        # separately records native initial velocity and controller tracking.
        outcomes = [geometric_screen(p.detach().cpu().double().numpy(), self.refiner.config.basis,
                                      self.constraint_spec) for p in prediction["trajectory"]]
        accepted = torch.tensor([o["geometrically_admissible"] for o in outcomes],
                                device=base_physical.device, dtype=torch.bool)
        trajectory = torch.where(accepted[:, None, None], prediction["trajectory"], base_physical)
        effect = None
        if predict_effect:
            if self.effect_teacher is None:
                raise ValueError("Predicted-effect deployment requires the exported learned effect module")
            effect = self.effect_teacher(context, base_physical, trajectory)
        return {**prediction, "trajectory": trajectory,
                "geometry_accepted": accepted, "geometry_screen": outcomes,
                "real_feedback_used": False, "teacher_forward_calls": int(predict_effect),
                "predicted_effect": effect, "probability_calibration": "NOT_CALIBRATED"}

    @torch.no_grad()
    def generate(self, observations, *, initial_noise=None):
        if self.base_policy is None or not hasattr(self.base_policy, "encode_and_plan"):
            raise ValueError("Bind the audited one-call Base encode_and_plan adapter for online use")
        context, base = self.base_policy.encode_and_plan(observations, initial_noise=initial_noise)
        return self.deploy(context, base)

    def deploy_state(self, *, include_effect_teacher=False):
        """Portable small-head state; optional teacher export is an explicit extra cost."""
        if include_effect_teacher and self.effect_teacher is None:
            raise ValueError("No learned effect teacher is bound for optional deployment")
        return {"schema": "effect_refiner_deploy_v1", "refiner": self.refiner.state_dict(),
                "config": self.refiner.config, "constraint_spec": self.constraint_spec,
                "inference_requires": ["current_legal_context", "base_physical_trajectory"],
                "effect_teacher_required": include_effect_teacher,
                "effect_teacher": self.effect_teacher.state_dict() if include_effect_teacher else None,
                "effect_teacher_config": getattr(self.effect_teacher, "config", None) if include_effect_teacher else None,
                "effect_label_spec": getattr(self.effect_teacher, "label_spec", None) if include_effect_teacher else None}
