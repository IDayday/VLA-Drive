import os
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
os.environ.setdefault("OPENSCENE_DATA_ROOT", "/path/to/openscene-v1.1")
os.environ.setdefault("NAVSIM_EXP_ROOT", "exp")
os.environ.setdefault("HYDRA_FULL_ERROR", "1")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "/path/to/maps")


import logging
import hydra
from omegaconf import DictConfig

from nuplan.planning.script.builders.logging_builder import build_logger

from navsim.planning.metric_caching.train_caching import cache_data
from navsim.planning.script.builders.worker_pool_builder import build_worker

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/metric_caching"
CONFIG_NAME = "train_metric_caching"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for metric caching.
    :param cfg: omegaconf dictionary
    """
    # Configure logger
    build_logger(cfg)

    # Build worker
    worker = build_worker(cfg)

    # Precompute and cache all features
    logger.info("Starting Metric Caching...")
    if cfg.worker == "ray_distributed" and cfg.worker.use_distributed:
        raise AssertionError("ray in distributed mode will not work with this job")
    cache_data(cfg=cfg, worker=worker)


if __name__ == "__main__":
    main()
