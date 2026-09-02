from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import pickle
import gzip
import os
import threading

import torch
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate

from navsim.common.dataloader import SceneLoader
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

logger = logging.getLogger(__name__)


def _decode_image_path(path_tensor: torch.Tensor) -> str:
    """Decode the path representation emitted by DriveVLAFeatureBuilder."""
    if path_tensor.ndim > 1:
        path_tensor = path_tensor.squeeze(0)
    return "".join(chr(code) for code in path_tensor.tolist() if code != 0)


def drivevla_cached_collate(batch):
    """Collate worker-preprocessed images without requiring equal patch counts."""
    features = [dict(sample[0]) for sample in batch]
    pixel_values = [feature.pop("pixel_values") for feature in features]
    tile_metadata = [feature.pop("tile_metadata", None) for feature in features]
    collated_features = default_collate(features)

    # NAVSIM front-camera images normally all produce nine patches.  Stack that
    # common case into one pinned allocation and retain a list fallback for any
    # future dataset containing mixed aspect ratios.
    first_shape = pixel_values[0].shape
    if all(value.shape == first_shape for value in pixel_values):
        collated_features["pixel_values"] = torch.stack(pixel_values, dim=0)
    else:
        collated_features["pixel_values"] = pixel_values
    if any(metadata is not None for metadata in tile_metadata):
        if not all(metadata is not None for metadata in tile_metadata):
            raise ValueError("tile_metadata must be present for every sample")
        first_metadata_shape = tile_metadata[0].shape
        if all(metadata.shape == first_metadata_shape for metadata in tile_metadata):
            collated_features["tile_metadata"] = torch.stack(tile_metadata, dim=0)
        else:
            collated_features["tile_metadata"] = tile_metadata

    collated_targets = default_collate([sample[1] for sample in batch])
    if len(batch[0]) == 2:
        return collated_features, collated_targets
    if len(batch[0]) == 3:
        return collated_features, collated_targets, default_collate(
            [sample[2] for sample in batch]
        )
    raise ValueError(f"Unsupported cached sample width: {len(batch[0])}")


def load_feature_target_from_pickle(path: Path) -> Dict[str, torch.Tensor]:
    """Helper function to load pickled feature/target from path."""
    with gzip.open(path, "rb") as f:
        data_dict: Dict[str, torch.Tensor] = pickle.load(f)
    return data_dict


def dump_feature_target_to_pickle(path: Path, data_dict: Dict[str, torch.Tensor]) -> None:
    """Helper function to save feature/target to pickle."""
    # Use compresslevel = 1 to compress the size but also has fast write and read.
    # Publish atomically so an interrupted distributed worker cannot leave a
    # truncated ``.gz`` file that a resumed job mistakes for a valid cache.
    temporary_path = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    try:
        with gzip.open(temporary_path, "wb", compresslevel=1) as f:
            pickle.dump(data_dict, f)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


class CacheOnlyDataset(torch.utils.data.Dataset):
    """Dataset wrapper for feature/target datasets from cache only."""

    def __init__(
        self,
        cache_path: str,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        log_names: Optional[List[str]] = None,
        append_token_to_batch=False,
        preprocess_images: bool = False,
        preprocess_image_dtype: str = "bfloat16",
        pretokenize_inputs: bool = False,
        tokenizer=None,
    ):
        """
        Initializes the dataset module.
        :param cache_path: directory to cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :param log_names: optional list of log folder to consider, defaults to None
        """
        super().__init__()
        assert Path(cache_path).is_dir(), f"Cache path {cache_path} does not exist!"
        self._cache_path = Path(cache_path)

        if log_names is not None:
            self.log_names = [Path(log_name) for log_name in log_names if (self._cache_path / log_name).is_dir()]
        else:
            self.log_names = [log_name for log_name in self._cache_path.iterdir()]

        self._feature_builders = feature_builders
        self._target_builders = target_builders
        self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
            cache_path=self._cache_path,
            feature_builders=self._feature_builders,
            target_builders=self._target_builders,
            log_names=self.log_names,
        )
        self.tokens = list(self._valid_cache_paths.keys())
        self.append_token_to_batch = append_token_to_batch
        self.preprocess_images = preprocess_images
        self.pretokenize_inputs = pretokenize_inputs
        self.tokenizer = tokenizer
        if self.pretokenize_inputs and self.tokenizer is None:
            raise ValueError("pretokenize_inputs requires an initialized tokenizer")
        try:
            self.preprocess_image_dtype = getattr(torch, preprocess_image_dtype)
        except AttributeError as error:
            raise ValueError(
                f"Unsupported preprocess image dtype: {preprocess_image_dtype}"
            ) from error

    def __len__(self) -> int:
        """
        :return: number of samples to load
        """
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Loads and returns pair of feature and target dict from data.
        :param idx: index of sample to load.
        :return: tuple of feature and target dictionary
        """
        return self._load_scene_with_token(self.tokens[idx])

    @staticmethod
    def _load_valid_caches(
        cache_path: Path,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        log_names: List[Path],
    ) -> Dict[str, Path]:
        """
        Helper method to load valid cache paths.
        :param cache_path: directory of training cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :param log_names: list of log paths to load
        :return: dictionary of tokens and sample paths as keys / values
        """

        valid_cache_paths: Dict[str, Path] = {}

        for log_name in tqdm(log_names, desc="Loading Valid Caches"):
            log_path = cache_path / log_name
            for token_path in log_path.iterdir():
                # found_caches: List[bool] = []
                # for builder in feature_builders + target_builders:
                #     data_dict_path = token_path / (builder.get_unique_name() + ".gz")
                #     found_caches.append(data_dict_path.is_file())
                # if all(found_caches):
                valid_cache_paths[token_path.name] = token_path

        return valid_cache_paths

    def _load_scene_with_token(self, token: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Helper method to load sample tensors given token
        :param token: unique string identifier of sample
        :return: tuple of feature and target dictionaries
        """

        token_path = self._valid_cache_paths[token]

        features: Dict[str, torch.Tensor] = {}
        
        for builder in self._feature_builders:
            data_dict_path = token_path / (os.getenv('FEATURE_NAME',builder.get_unique_name()) + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            for key,value in data_dict.items():
                data_dict[key]=value.detach()
            features.update(data_dict)

        if self.preprocess_images:
            from navsim.agents.EpisodeDrive.utils.internvl_preprocess import load_image
            from navsim.agents.EpisodeDrive.utils.utils import build_drivevla_questions

            image_path_tensor = features.pop("image_path_tensor")
            image_path = _decode_image_path(image_path_tensor)
            # The backbone immediately casts the released FP32 preprocessing
            # result to BF16 on CUDA.  Casting here is bit-identical on this
            # platform and halves both pinned-memory footprint and H2D traffic.
            pixel_values, tile_metadata = load_image(
                image_path, return_tile_metadata=True
            )
            features["pixel_values"] = pixel_values.to(dtype=self.preprocess_image_dtype)
            features["tile_metadata"] = tile_metadata
            features["questions"] = build_drivevla_questions(
                features["history_trajectory"], features["high_command_one_hot"]
            )[0]
            if self.pretokenize_inputs:
                from navsim.agents.EpisodeDrive.drivevla_backbone import system_message
                from navsim.agents.EpisodeDrive.utils.internvl_tokenize import (
                    build_internvl_model_inputs,
                )

                model_inputs = build_internvl_model_inputs(
                    self.tokenizer,
                    [features["questions"]],
                    [features["pixel_values"].shape[0]],
                    system_message,
                )
                features["input_ids"] = model_inputs["input_ids"].squeeze(0)
                features["attention_mask"] = model_inputs["attention_mask"].squeeze(0)
                del features["questions"]

        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            data_dict_path = token_path / (os.getenv('TARGET_NAME',builder.get_unique_name()) + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            if 'token' not in data_dict:
                data_dict['token']=token
            targets.update(data_dict)

        if self.append_token_to_batch:
            return (features,targets,token)
        else:
            return (features, targets)


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scene_loader: SceneLoader,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        cache_path: Optional[str] = None,
        force_cache_computation: bool = False,
        append_token_to_batch: bool = False,
    ):
        super().__init__()
        self._scene_loader = scene_loader
        self._feature_builders = feature_builders
        self._target_builders = target_builders

        self._cache_path: Optional[Path] = Path(cache_path) if cache_path else None
        self._force_cache_computation = force_cache_computation
        self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
            self._cache_path, feature_builders, target_builders, self._scene_loader
        )
        self.append_token_to_batch = append_token_to_batch
        if self._cache_path is not None:
            self.cache_dataset()

    @staticmethod
    def _load_valid_caches(
        cache_path: Optional[Path],
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        scene_loader: SceneLoader,
    ) -> Dict[str, Path]:
        """
        Helper method to load valid cache paths.
        :param cache_path: directory of training cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :return: dictionary of tokens and sample paths as keys / values
        """

        valid_cache_paths: Dict[str, Path] = {}

        if (cache_path is not None) and cache_path.is_dir():
            # A cache worker only needs to inspect its own tokens. Scanning the
            # complete shared cache in every worker makes resume
            # O(workers * dataset_size) and creates heavy metadata contention.
            for log_name, tokens in scene_loader.get_tokens_list_per_log().items():
                for token in tokens:
                    token_path = cache_path / log_name / token
                    found_caches = [
                        (token_path / (builder.get_unique_name() + ".gz")).is_file()
                        for builder in feature_builders + target_builders
                    ]
                    if all(found_caches):
                        valid_cache_paths[token] = token_path

        return valid_cache_paths

    def _cache_scene_with_token(self, token: str) -> None:
        """
        Helper function to compute feature / targets and save in cache.
        :param token: unique identifier of scene to cache
        """

        scene = self._scene_loader.get_scene_from_token(token)
        agent_input = scene.get_agent_input()

        metadata = scene.scene_metadata
        token_path = self._cache_path / metadata.log_name / metadata.initial_token
        os.makedirs(token_path, exist_ok=True)

        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = builder.compute_features(agent_input)
            dump_feature_target_to_pickle(data_dict_path, data_dict)

        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = builder.compute_targets(scene)
            dump_feature_target_to_pickle(data_dict_path, data_dict)

        self._valid_cache_paths[token] = token_path

    def _load_scene_with_token(self, token: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Helper function to load feature / targets from cache.
        :param token:  unique identifier of scene to load
        :return: tuple of feature and target dictionaries
        """

        token_path = self._valid_cache_paths[token]

        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            features.update(data_dict)

        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            targets.update(data_dict)

        return (features, targets)

    def cache_dataset(self) -> None:
        """Caches complete dataset into cache folder."""

        assert self._cache_path is not None, "Dataset did not receive a cache path!"
        os.makedirs(self._cache_path, exist_ok=True)

        # determine tokens to cache
        if self._force_cache_computation:
            tokens_to_cache = self._scene_loader.tokens
        else:
            tokens_to_cache = set(self._scene_loader.tokens) - set(self._valid_cache_paths.keys())
            tokens_to_cache = list(tokens_to_cache)
            logger.info(
                f"""
                Starting caching of {len(tokens_to_cache)} tokens.
                Note: Caching tokens within the training loader is slow. Only use it with a small number of tokens.
                You can cache large numbers of tokens using the `run_dataset_caching.py` python script.
                """
            )

        for token in tqdm(tokens_to_cache, desc="Caching Dataset"):
            self._cache_scene_with_token(token)

    def __len__(self) -> None:
        """
        :return: number of samples to load
        """
        return len(self._scene_loader)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]:
        """
        Get features or targets either from cache or computed on-the-fly.
        :param idx: index of sample to load.
        :return: tuple of feature and target dictionary
        """

        token = self._scene_loader.tokens[idx]
        features: Dict[str, torch.Tensor] = {}
        targets: Dict[str, torch.Tensor] = {}

        if self._cache_path is not None:
            assert (
                token in self._valid_cache_paths.keys()
            ), f"The token {token} has not been cached yet, please call cache_dataset first!"

            features, targets = self._load_scene_with_token(token)
        else:
            scene = self._scene_loader.get_scene_from_token(self._scene_loader.tokens[idx])
            agent_input = self._scene_loader.get_agent_input_from_token(self._scene_loader.tokens[idx])
            for builder in self._feature_builders:
                features.update(builder.compute_features(agent_input))
            for builder in self._target_builders:
                targets.update(builder.compute_targets(scene))

        features["scenario_token"] = token
        if self.append_token_to_batch:
            return (features, targets, token)
        else:
            return (features, targets,)
