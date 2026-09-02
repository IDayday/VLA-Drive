# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import pickle
import hashlib
import json
import traceback
import uuid
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from dataclasses import asdict

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.distributed as dist
from hydra.utils import instantiate
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import PDMResults, SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import Dataset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_gpu"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_formal_candidate_bank(
    predictions: Dict[str, Dict[str, Any]],
    token_order: List[str],
    output_path: Path,
) -> Dict[str, Any]:
    """Persist current-frame model outputs once; official scoring is separate."""
    ordered_tokens = [token for token in token_order if token in predictions]
    if len(ordered_tokens) != len(predictions):
        raise RuntimeError("Candidate-bank token ordering is incomplete")
    component_names = (
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "time_to_collision_within_bound",
        "ego_progress",
        "driving_direction_compliance",
        "comfort",
    )
    component_probabilities = np.stack(
        [
            np.stack(
                [
                    predictions[token]["component_probabilities"][name]
                    for name in component_names
                ],
                axis=-1,
            )
            for token in ordered_tokens
        ]
    )
    arrays = {
        "tokens": np.asarray(ordered_tokens),
        "proposals": np.stack(
            [predictions[token]["proposals"] for token in ordered_tokens]
        ).astype(np.float32),
        "predicted_log_pdm": np.stack(
            [predictions[token]["predicted_log_pdm"] for token in ordered_tokens]
        ).astype(np.float32),
        "predicted_pdms": np.stack(
            [predictions[token]["predicted_pdms"] for token in ordered_tokens]
        ).astype(np.float32),
        "selected_indices": np.asarray(
            [predictions[token]["selected_index"] for token in ordered_tokens],
            dtype=np.int64,
        ),
        "component_probabilities": component_probabilities.astype(np.float32),
        "component_names": np.asarray(component_names),
        "planning_registers": np.stack(
            [predictions[token]["planning_registers"] for token in ordered_tokens]
        ).astype(np.float32),
        "tile_gate": np.asarray(
            [predictions[token]["tile_gate"] for token in ordered_tokens],
            dtype=np.float32,
        ),
        "semantic_gate": np.asarray(
            [predictions[token]["semantic_gate"] for token in ordered_tokens],
            dtype=np.float32,
        ),
        "inference_latency_seconds": np.asarray(
            [
                predictions[token]["inference_latency_seconds"]
                for token in ordered_tokens
            ],
            dtype=np.float64,
        ),
    }
    if arrays["proposals"].shape[1:] != (64, 8, 3):
        raise RuntimeError(
            "Formal candidate bank requires exactly [scene,64,8,3], got "
            f"{arrays['proposals'].shape}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(output_path)
    manifest = {
        "schema_version": 1,
        "candidate_bank": str(output_path.resolve()),
        "candidate_bank_sha256": _sha256(output_path),
        "scene_count": len(ordered_tokens),
        "candidate_count": 64,
        "poses_per_candidate": 8,
        "state_size": 3,
        "inference_uses_future_inputs": False,
        "official_candidate_scores_in_bank": False,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest

def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[pd.DataFrame]:
    """
    Helper function to run PDMS evaluation in.
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]
    model_trajectory = args[0]['model_trajectory']

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    # print(cfg)
    print("!!!!!!!!!!!!!!!!Path(cfg.metric_cache_path) ", Path(cfg.metric_cache_path))
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        load_image_path=cfg.load_image_path
    )

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    pdm_results: List[Dict[str, Any]] = []
    for idx, (token) in enumerate(tokens_to_evaluate):
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        score_row: Dict[str, Any] = {"token": token, "valid": True}
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            trajectory = model_trajectory[token]['trajectory']

            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            score_row.update(asdict(pdm_result))
        except Exception as e:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False

        pdm_results.append(score_row)

    return pdm_results

def dist_ready():
    return dist.is_available() and dist.is_initialized()

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """

    build_logger(cfg)
    if "seed" in cfg:
        pl.seed_everything(int(cfg.seed), workers=True)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    dump_root = os.path.join(os.getenv('SUBSCORE_PATH'), "navsim1_pdm_scores", cfg.experiment_name)
    os.makedirs(dump_root, exist_ok=True)
    dump_path = os.path.join(dump_root, f"{timestamp}.pkl")
    print(f'Subscore/Trajectories saved to {dump_path}')
    # gpu inference
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    scene_filter = instantiate(cfg.train_test_split.scene_filter)

    scene_loader_inference = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=cfg.load_image_path
    )
    dataset = Dataset(
        scene_loader=scene_loader_inference,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=None,
        force_cache_computation=False,
        append_token_to_batch=True,
    )
    dataloader = DataLoader(dataset, **cfg.dataloader.params, shuffle=False)
    trainer = pl.Trainer(**cfg.trainer.params)
    predictions = trainer.predict(
        AgentLightningModule(
            agent=agent,
            for_analysis=bool(cfg.get("candidate_analysis", False)),
        ),
        dataloader,
        return_predictions=True
    )
    
    if dist_ready():
        dist.barrier()
    
    world_size = dist.get_world_size() if dist_ready() else 1
    all_predictions = [None for _ in range(world_size)]

    if dist_ready():
        dist.all_gather_object(all_predictions, predictions)
    else:
        all_predictions = [predictions]

    rank = dist.get_rank() if dist_ready() else 0
    if rank != 0:
        return None

    merged_predictions = {}
    for proc_prediction in all_predictions:
        for d in proc_prediction:
            merged_predictions.update(d)

    pickle.dump(merged_predictions, open(dump_path, 'wb'))

    if bool(cfg.get("candidate_analysis", False)):
        candidate_path = cfg.get("candidate_artifact_path")
        if not candidate_path:
            raise ValueError(
                "candidate_analysis=true requires candidate_artifact_path"
            )
        manifest = save_formal_candidate_bank(
            merged_predictions,
            list(scene_loader_inference.tokens),
            Path(str(candidate_path)).expanduser().resolve(),
        )
        logger.info("Formal candidate bank: %s", json.dumps(manifest, sort_keys=True))

    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
            "model_trajectory": merged_predictions
        }
        for log_file, tokens_list in scene_loader_inference.get_tokens_list_per_log().items()
    ]

    worker = build_worker(cfg)
    score_rows: List[pd.DataFrame] = worker_map(worker, run_pdm_score, data_points)

    pdm_score_df = pd.DataFrame(score_rows)
    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Final average score of valid results: {pdm_score_df['score'].mean()}.
            Results are stored in: {save_path / f"{timestamp}.csv"}.

            All scores:
            {average_row}
        """
    )


if __name__ == "__main__":
    main()
