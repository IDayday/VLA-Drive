"""Online M0 adapter for scorer-private current-observation rankers.

Two strictly M0-owned observation paths are supported:

* spatial four-view tokens extracted by the frozen M0 vision tower; and
* the 16 current-scene tokens already produced by the frozen EpisodeDrive
  forward pass.

The latter is the lightweight No-VQA scorer-refiner path and must not trigger a
second vision-tower forward.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from local_stage2.export_multiview_dino_observation_replay import CAMERA_NAMES
from local_stage2.export_multiview_m0_observation_replay import (
    pool_m0_visual_tokens,
)
from local_stage2.export_private_visual_replay import _resolve_visual_model
from navsim.agents.EpisodeDrive.drivevla_features import DriveVLAFeatureBuilder
from navsim.agents.EpisodeDrive.episodedrive_agent import EpisodeDriveAgent
from navsim.agents.EpisodeDrive.score_module.independent_ranker import (
    FACTOR_KEYS,
    IndependentProposalRanker,
    IndependentRankerConfig,
    pdms_factor_log_utility,
)
from navsim.agents.EpisodeDrive.score_module.m0_private_residual_ranker import (
    M0PrivateResidualConfig,
    M0PrivateResidualRanker,
)
from navsim.agents.EpisodeDrive.utils.internvl_preprocess import load_image


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


class M0NativePrivateFeatureBuilder(DriveVLAFeatureBuilder):
    """Keep Base front-view inputs and add four current image paths/status."""

    def __init__(self) -> None:
        super().__init__(cache_hidden_state=False, cache_mode=False)

    @staticmethod
    def _path_tensor(path: Path) -> torch.Tensor:
        if not path.is_file():
            raise FileNotFoundError(path)
        return torch.tensor([ord(char) for char in str(path)], dtype=torch.long)

    def compute_features(self, agent_input) -> Dict[str, torch.Tensor]:
        result = super().compute_features(agent_input)
        cameras = agent_input.cameras[-1]
        for camera_name in CAMERA_NAMES:
            path = getattr(cameras, camera_name).image
            if not isinstance(path, Path):
                raise RuntimeError(f"{camera_name} image path is unavailable")
            result[f"m0_private_{camera_name}_path_tensor"] = self._path_tensor(path)
        current = agent_input.ego_statuses[-1]
        result["m0_private_status_feature"] = torch.cat(
            (
                torch.as_tensor(current.ego_pose, dtype=torch.float32),
                torch.as_tensor(current.ego_velocity, dtype=torch.float32),
                torch.as_tensor(current.ego_acceleration, dtype=torch.float32),
                torch.as_tensor(current.driving_command, dtype=torch.float32),
            )
        )
        return result


class M0NativePrivateScorerAgent(EpisodeDriveAgent):
    """Released proposal generator plus an M0-owned four-view scorer."""

    ARTIFACT_TYPE = "episode_drive_m0_native_private_scorer_v1"

    def __init__(self, *args, **kwargs) -> None:
        action_config = kwargs.get("action_head_config")
        if action_config is None:
            raise ValueError("action_head_config is required")
        try:
            from omegaconf import OmegaConf

            OmegaConf.update(
                action_config, "return_memory_fields", True, force_add=True
            )
            OmegaConf.update(
                action_config, "return_scorer_features", True, force_add=True
            )
        except Exception:
            setattr(action_config, "return_memory_fields", True)
            setattr(action_config, "return_scorer_features", True)
        super().__init__(*args, **kwargs)
        self.private_scorer: Optional[torch.nn.Module] = None
        self._private_artifact: Optional[Dict[str, object]] = None
        self._private_visual_model: Optional[torch.nn.Module] = None

    def _peek_observation_source(self) -> str:
        if self._private_artifact is not None:
            return str(
                self._private_artifact.get(
                    "private_observation_source",
                    "private_current_visual_token_cache",
                )
            )
        if not self.checkpoint_path:
            raise RuntimeError("M0-native private scorer artifact is required")
        requested = Path(self.checkpoint_path)
        payload = torch.load(requested, map_location="cpu", weights_only=False)
        if payload.get("artifact_type") != self.ARTIFACT_TYPE:
            raise RuntimeError(
                f"Expected {self.ARTIFACT_TYPE}, got {payload.get('artifact_type')!r}"
            )
        return str(
            payload.get(
                "private_observation_source",
                "private_current_visual_token_cache",
            )
        )

    def get_feature_builders(self):
        if bool(self.vlm_config.cache_hidden_state) or bool(
            self.vlm_config.cache_mode
        ):
            raise RuntimeError(
                "M0-native private scorer requires current online image paths"
            )
        if str(self.vlm_config.vlm_type).lower() != "internvl":
            raise RuntimeError("M0-native private scorer currently requires InternVL")
        if self._peek_observation_source() == "source_checkpoint_current_scene_tokens":
            return [DriveVLAFeatureBuilder(cache_hidden_state=False, cache_mode=False)]
        return [M0NativePrivateFeatureBuilder()]

    def initialize(self) -> None:
        if self._initialized and self.private_scorer is not None:
            return
        if not self.checkpoint_path:
            raise RuntimeError("M0-native private scorer artifact is required")
        requested = Path(self.checkpoint_path)
        payload = torch.load(requested, map_location="cpu", weights_only=False)
        if payload.get("artifact_type") != self.ARTIFACT_TYPE:
            raise RuntimeError(
                f"Expected {self.ARTIFACT_TYPE}, got {payload.get('artifact_type')!r}"
            )
        base_checkpoint = Path(str(payload["base_checkpoint_path"]))
        if not base_checkpoint.is_file():
            raise FileNotFoundError(base_checkpoint)
        if _sha256(base_checkpoint) != payload["base_checkpoint_sha256"]:
            raise RuntimeError("released M0 Base checkpoint SHA256 mismatch")

        self.checkpoint_path = str(base_checkpoint)
        try:
            super().initialize()
        finally:
            self.checkpoint_path = str(requested)

        architecture = str(payload["scorer_architecture"])
        if architecture == "IndependentProposalRanker":
            scorer = IndependentProposalRanker(
                IndependentRankerConfig(**dict(payload["private_config"]))
            )
        elif architecture == "M0PrivateResidualRanker":
            scorer = M0PrivateResidualRanker(
                IndependentRankerConfig(**dict(payload["private_config"])),
                M0PrivateResidualConfig(**dict(payload["residual_config"])),
            )
        else:
            raise RuntimeError(f"unsupported private scorer architecture: {architecture}")
        scorer.load_state_dict(payload["scorer_state_dict"], strict=True)
        scorer.eval()
        self.private_scorer = scorer
        self._private_artifact = dict(payload)
        observation_source = str(
            payload.get(
                "private_observation_source",
                "private_current_visual_token_cache",
            )
        )
        if observation_source == "private_current_visual_token_cache":
            if self.backbone is None:
                raise RuntimeError("released M0 visual backbone was not initialized")
            visual_model, wrapper_chain = _resolve_visual_model(self.backbone)
            declared_chain = payload["private_vision_config"].get(
                "visual_model_wrapper_chain"
            )
            if declared_chain is not None and list(declared_chain) != wrapper_chain:
                raise RuntimeError("online M0 visual wrapper chain differs from cache")
            self._private_visual_model = visual_model
        elif observation_source == "source_checkpoint_current_scene_tokens":
            if payload.get("private_vision_config") is not None:
                raise RuntimeError(
                    "scene-token scorer artifact must not declare a private vision cache"
                )
            self._private_visual_model = None
        else:
            raise RuntimeError(
                f"unsupported private observation source: {observation_source}"
            )
        print(f"✅ M0-native private scorer loaded: {requested}")

    @staticmethod
    def _decode_camera_paths(
        features: Dict[str, torch.Tensor],
    ) -> List[List[str]]:
        by_camera: List[List[str]] = []
        for camera_name in CAMERA_NAMES:
            key = f"m0_private_{camera_name}_path_tensor"
            value = features.pop(key, None)
            if value is None:
                raise RuntimeError(f"missing current private camera input: {key}")
            if value.is_cuda:
                value = value.detach().cpu()
            if value.ndim == 1:
                value = value.unsqueeze(0)
            by_camera.append(EpisodeDriveAgent._decode_paths_from_tensor(value))
        batch_sizes = {len(paths) for paths in by_camera}
        if len(batch_sizes) != 1:
            raise RuntimeError("private camera batches have different sizes")
        return [
            [by_camera[camera][scene] for camera in range(len(CAMERA_NAMES))]
            for scene in range(next(iter(batch_sizes)))
        ]

    @torch.inference_mode()
    def _encode_private_observation(
        self,
        scene_paths: Sequence[Sequence[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._private_artifact is None or self._private_visual_model is None:
            raise RuntimeError("private scorer is not initialized")
        config = self._private_artifact["private_vision_config"]
        max_dynamic_tiles = int(config["max_dynamic_tiles"])
        groups = [
            load_image(path, max_num=max_dynamic_tiles)
            for paths in scene_paths
            for path in paths
        ]
        crop_counts = torch.as_tensor(
            [group.shape[0] for group in groups], dtype=torch.int16
        ).reshape(len(scene_paths), len(CAMERA_NAMES))
        device = next(self._private_visual_model.parameters()).device
        raw = self._private_visual_model.extract_feature(
            torch.cat(groups).to(device=device, non_blocking=True).bfloat16()
        )
        pooled, valid = pool_m0_visual_tokens(
            raw,
            crop_counts,
            tuple(config["pool_grid"]),
            int(config["max_crops_per_camera"]),
        )
        return pooled.to(device=device).float(), valid.to(device=device)

    def forward(self, features, targets=None, tokens_list=None):
        if self.private_scorer is None or self._private_artifact is None:
            raise RuntimeError("M0-native private scorer was not initialized")
        observation_source = str(
            self._private_artifact.get(
                "private_observation_source",
                "private_current_visual_token_cache",
            )
        )
        scene_paths = None
        private_status = None
        if observation_source == "private_current_visual_token_cache":
            scene_paths = self._decode_camera_paths(features)
            private_status = features.pop("m0_private_status_feature", None)
            if private_status is None:
                raise RuntimeError("missing current private status feature")
            if private_status.ndim == 1:
                private_status = private_status.unsqueeze(0)
        prediction = super().forward(features, targets, tokens_list)
        proposals = prediction["proposals"]
        base_scores = prediction["pdm_score"]
        if observation_source == "private_current_visual_token_cache":
            if scene_paths is None or private_status is None:
                raise RuntimeError("private visual inputs were not prepared")
            observation, observation_mask = self._encode_private_observation(
                scene_paths
            )
            status = private_status.to(
                device=proposals.device, non_blocking=True
            ).float()
        elif observation_source == "source_checkpoint_current_scene_tokens":
            if "language_feature" not in prediction or "ego_feature" not in prediction:
                raise RuntimeError(
                    "No-VQA scene/ego tokens are absent from the base forward"
                )
            observation = prediction["language_feature"].float()
            ego_feature = prediction["ego_feature"].float()
            if observation.shape[1:] != (16, 256):
                raise RuntimeError(
                    f"unexpected current scene-token shape: {observation.shape}"
                )
            if ego_feature.shape[1:] != (1, 256):
                raise RuntimeError(
                    f"unexpected current ego-token shape: {ego_feature.shape}"
                )
            observation_mask = torch.ones(
                observation.shape[:2],
                dtype=torch.bool,
                device=observation.device,
            )
            status = ego_feature.squeeze(1)
        else:
            raise RuntimeError(
                f"unsupported private observation source: {observation_source}"
            )
        architecture = str(self._private_artifact["scorer_architecture"])
        if architecture == "IndependentProposalRanker":
            output = self.private_scorer(
                observation,
                status,
                proposals,
                observation_valid_mask=observation_mask,
            )
            mode = str(self._private_artifact["score_mode"])
            if mode == "direct":
                private_scores = output["utility"]
            elif mode == "coarse":
                private_scores = output["coarse_utility"]
            elif mode == "factor":
                private_scores = pdms_factor_log_utility(output["factor_logits"])
            else:
                raise RuntimeError(f"unsupported independent score mode: {mode}")
            shortlist_count = min(
                int(self._private_artifact["shortlist_size"]),
                proposals.shape[1],
            )
            shortlist = base_scores.topk(shortlist_count, dim=1).indices
            selection_scores = torch.full_like(private_scores, -1.0e4)
            selection_scores.scatter_(
                1, shortlist, private_scores.gather(1, shortlist)
            )
        else:
            factor_logits = torch.stack(
                [prediction["pred_logit"][key] for key in FACTOR_KEYS], dim=-1
            )
            m0_context = {}
            if self.private_scorer.residual_config.m0_candidate_fusion:
                if "scorer_candidate_features" not in prediction:
                    raise RuntimeError(
                        "released M0 candidate features are absent from online forward"
                    )
                m0_context["m0_candidate_features"] = prediction[
                    "scorer_candidate_features"
                ].float()
            if self.private_scorer.residual_config.m0_context_fusion:
                if "language_feature" not in prediction or "ego_feature" not in prediction:
                    raise RuntimeError(
                        "released M0 scene/ego context is absent from online forward"
                    )
                m0_context.update(
                    {
                        "m0_scene_features": prediction["language_feature"].float(),
                        "m0_ego_features": prediction["ego_feature"].float(),
                    }
                )
            output = self.private_scorer(
                observation,
                status,
                proposals,
                factor_logits,
                base_scores,
                observation_valid_mask=observation_mask,
                **m0_context,
            )
            selection_scores = output["selection_scores"]

        selected = selection_scores.argmax(dim=1)
        prediction["base_pdm_score"] = base_scores
        prediction["m0_private_scorer_output"] = output
        prediction["pdm_score"] = selection_scores
        prediction["trajectory"] = proposals[
            torch.arange(len(selected), device=selected.device), selected
        ]
        return prediction


def _private_vision_config(root: Path) -> Dict[str, object]:
    manifests = sorted(root.glob("*_shard_*-of-*/manifest.json"))
    if not manifests:
        raise RuntimeError(f"private M0 cache has no manifests: {root}")
    payloads = [json.loads(path.read_text()) for path in manifests]
    shard_counts = {int(payload["shard_count"]) for payload in payloads}
    if len(shard_counts) != 1 or len(payloads) != next(iter(shard_counts)):
        raise RuntimeError("private M0 cache manifests are incomplete")
    keys = (
        "m0_checkpoint_sha256",
        "camera_names",
        "max_dynamic_tiles",
        "max_crops_per_camera",
        "pool_grid",
        "visual_token_count",
        "visual_width",
        "visual_model_wrapper_chain",
    )
    config = {key: payloads[0][key] for key in keys}
    for payload in payloads:
        if any(payload[key] != config[key] for key in keys):
            raise RuntimeError("private M0 cache shards use different vision configs")
        if not bool(payload["current_observation_only"]):
            raise RuntimeError("private M0 cache is not current-observation-only")
        if bool(payload["future_or_evaluator_input"]):
            raise RuntimeError("private M0 cache declares future/evaluator input")
        if bool(payload["official_score_or_factor_input"]):
            raise RuntimeError("private M0 cache declares official score input")
        if bool(payload["proposal_input"]):
            raise RuntimeError("private M0 cache declares proposal input")
    if tuple(config["camera_names"]) != CAMERA_NAMES:
        raise RuntimeError("private M0 cache camera order mismatch")
    config["manifest_sha256"] = {
        str(path.relative_to(root)): _sha256(path) for path in manifests
    }
    return config


def build_m0_native_private_scorer_artifact(
    source: Mapping[str, object],
    *,
    source_path: Path,
    base_checkpoint: Path,
    private_observation_root: Optional[Path] = None,
    shortlist_size: int = 64,
) -> Dict[str, object]:
    """Package a held-out-selected ranker for real online M0 inference."""

    architecture = str(source.get("architecture"))
    if architecture not in {
        "IndependentProposalRanker",
        "M0PrivateResidualRanker",
    }:
        raise ValueError(f"unsupported source architecture: {architecture}")
    if not source_path.is_file() or not base_checkpoint.is_file():
        raise FileNotFoundError(source_path if not source_path.is_file() else base_checkpoint)
    if shortlist_size <= 0:
        raise ValueError("shortlist_size must be positive")
    base_sha = _sha256(base_checkpoint)
    source_schema = tuple(source.get("inference_input_schema", ()))
    source_fold = dict(source.get("fold_manifest", {}))
    observation_source = str(
        source_fold.get(
            "scorer_private_observation_source",
            "private_current_visual_token_cache"
            if private_observation_root is not None
            else "",
        )
    )
    if observation_source == "source_checkpoint_current_scene_tokens":
        if private_observation_root is not None:
            raise RuntimeError(
                "scene-token ranker must be packaged without a private visual cache"
            )
        if not source_schema or source_schema[0] != "m0_current_scene_tokens":
            raise RuntimeError("ranker artifact does not declare M0 scene-token input")
        source_lineage = list(source_fold.get("source_lineage", []))
        selection_source = str(
            source.get(
                "checkpoint_selection_source",
                source_fold.get("checkpoint_selection_source", ""),
            )
        )
        if selection_source:
            selected_lineage = [
                value
                for value in source_lineage
                if str(value.get("name", "")) == selection_source
            ]
            if len(selected_lineage) != 1:
                raise RuntimeError(
                    "scene-token artifact lacks one selected-source lineage entry"
                )
            source_hashes = {
                str(selected_lineage[0].get("checkpoint_sha256"))
            }
        else:
            # Legacy single-source artifacts predate an explicit source name.
            # Training-only auxiliary targets legitimately have no checkpoint
            # hash and must not be mistaken for an inference representation.
            source_hashes = {
                str(value["checkpoint_sha256"])
                for value in source_lineage
                if value.get("checkpoint_sha256") is not None
            }
        if source_hashes != {base_sha}:
            raise RuntimeError(
                "scene-token training source and packaged Base checkpoint differ"
            )
        vision_config = None
    elif observation_source == "private_current_visual_token_cache":
        if private_observation_root is None:
            raise RuntimeError("private visual ranker requires its cache lineage")
        vision_config = _private_vision_config(private_observation_root)
        if str(vision_config["m0_checkpoint_sha256"]) != base_sha:
            raise RuntimeError("private visual cache and Base checkpoint SHA256 differ")
    else:
        raise RuntimeError(
            f"unsupported scorer-private observation source: {observation_source!r}"
        )
    if architecture == "IndependentProposalRanker":
        score_mode = str(source["selection_mode"])
        if score_mode not in {"direct", "coarse", "factor"}:
            raise ValueError(f"unsupported independent score mode: {score_mode}")
        state_dict = source["state_dict"]
        private_config = dict(source["model_config"])
        residual_config = None
    else:
        score_mode = f"residual_{source['residual_config']['score_mode']}"
        state_dict = source["state_dict"]
        private_config = dict(source["private_config"])
        residual_config = dict(source["residual_config"])
    if observation_source == "private_current_visual_token_cache":
        if int(private_config["observation_dim"]) != int(
            vision_config["visual_width"]
        ):
            raise RuntimeError("ranker/cache visual width mismatch")
        if int(private_config["max_observation_tokens"]) != int(
            vision_config["visual_token_count"]
        ):
            raise RuntimeError("ranker/cache visual token-count mismatch")
        if int(private_config["status_dim"]) != 11:
            raise RuntimeError(
                "M0-native private visual ranker must use 11 current-state values"
            )
    else:
        if int(private_config["observation_dim"]) != 256:
            raise RuntimeError("M0 scene-token ranker must use width 256")
        if int(private_config["max_observation_tokens"]) != 16:
            raise RuntimeError("M0 scene-token ranker must use exactly 16 tokens")
        if int(private_config["status_dim"]) != 256:
            raise RuntimeError("M0 scene-token ranker must use the 256-D ego token")
    context_fusion = bool(
        residual_config is not None
        and residual_config.get("m0_context_fusion", False)
    )
    return {
        "artifact_type": M0NativePrivateScorerAgent.ARTIFACT_TYPE,
        "artifact_version": 3 if observation_source == "source_checkpoint_current_scene_tokens" else (2 if context_fusion else 1),
        "base_checkpoint_path": str(base_checkpoint.resolve()),
        "base_checkpoint_sha256": base_sha,
        "source_ranker_artifact_path": str(source_path.resolve()),
        "source_ranker_artifact_sha256": _sha256(source_path),
        "source_ranker_epoch": int(source["epoch"]),
        "source_ranker_validation": source.get("validation"),
        "source_ranker_refit_all_logs": bool(source.get("refit_all_logs", False)),
        "source_ranker_validation_performed": source.get(
            "validation_performed", True
        ),
        "source_ranker_refit_provenance": source.get("refit_provenance"),
        "scorer_architecture": architecture,
        "scorer_state_dict": {
            key: value.detach().cpu() for key, value in state_dict.items()
        },
        "private_config": private_config,
        "residual_config": residual_config,
        "score_mode": score_mode,
        "shortlist_size": int(shortlist_size),
        "private_observation_source": observation_source,
        "private_vision_config": vision_config,
        "inference_input_schema": (
            (
                "m0_current_scene_tokens",
                "m0_current_context_feature",
            )
            if observation_source == "source_checkpoint_current_scene_tokens"
            else (
                "m0_current_f0_l0_r0_b0_images",
                "m0_current_ego_navigation_status",
            )
        )
        + (("m0_released_scene_features", "m0_released_ego_features") if context_fusion else ())
        + (
            ("m0_released_candidate_features",)
            if residual_config is not None
            and residual_config.get("m0_candidate_fusion", False)
            else ()
        )
        + ("m0_proposals",)
        + (
            ("m0_base_factor_logits", "m0_base_scores")
            if architecture == "M0PrivateResidualRanker"
            else ("m0_base_topk_membership",)
        ),
        "future_or_evaluator_input": False,
        "official_score_input": False,
        "external_model_representation_or_weight_used": False,
        "drivor_representation_or_weight_used": False,
        "released_m0_context_fusion": context_fusion,
    }


__all__ = (
    "M0NativePrivateFeatureBuilder",
    "M0NativePrivateScorerAgent",
    "build_m0_native_private_scorer_artifact",
)
